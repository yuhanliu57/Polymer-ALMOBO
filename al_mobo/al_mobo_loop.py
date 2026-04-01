import datetime
import json
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import normalize, unnormalize

from dkl_surrogates import build_mobo_surrogate, fit_two_dkl_models
from ground_truth import fetch_ground_truth_auto
from prepare_data import load_unlabeled, load_word2vec, prepare_data_train


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_train_tensors(X_train, tc_z, mod_z, flip_modulus=True, pad_ratio=0.10):
    X = torch.as_tensor(X_train, dtype=torch.double)
    modulus = -mod_z if flip_modulus else mod_z
    Y = torch.as_tensor(np.column_stack([tc_z, modulus]), dtype=torch.double)

    x_min = X.min(dim=0).values
    x_max = X.max(dim=0).values
    pad = (x_max - x_min).clamp_min(1e-12) * pad_ratio
    bounds = torch.stack([x_min - pad, x_max + pad])
    return X, Y, bounds


def get_reference_point(Y, eps=4.0):
    return Y.min(dim=0).values - eps


def map_cand_to_indices(cand_norm, pool_norm, tol=1e-6):
    distances = torch.cdist(cand_norm, pool_norm)
    idx_list = []
    used = set()

    for i in range(distances.size(0)):
        order = torch.argsort(distances[i]).tolist()
        idx = next(j for j in order if j not in used)
        if distances[i, idx].item() >= tol:
            raise AssertionError(
                f"Nearest neighbor too far for candidate {i}: {distances[i, idx].item():.3e} (tol={tol})"
            )
        idx_list.append(idx)
        used.add(idx)

    return idx_list


def select_candidates_nehvi_greedy(mobo_model, train_X, bounds, ref_point, X_unlabeled_t, q=4):
    pool_norm = normalize(X_unlabeled_t, bounds)
    q = min(q, pool_norm.size(0))
    if q == 0:
        raise ValueError("Candidate pool is empty.")

    acqf = qNoisyExpectedHypervolumeImprovement(
        model=mobo_model,
        X_baseline=normalize(train_X, bounds),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=42),
        ref_point=ref_point.tolist() if torch.is_tensor(ref_point) else ref_point,
        prune_baseline=True,
    )
    cand_norm, _ = optimize_acqf_discrete(acq_function=acqf, choices=pool_norm, q=q, unique=True)
    idx_list = map_cand_to_indices(cand_norm, pool_norm)
    return unnormalize(cand_norm, bounds), idx_list


def compute_hv(train_Y, ref_point):
    Y_nd = train_Y[is_non_dominated(train_Y)]
    ref_point = ref_point if torch.is_tensor(ref_point) else torch.as_tensor(ref_point, dtype=train_Y.dtype)
    return NondominatedPartitioning(ref_point=ref_point, Y=Y_nd).compute_hypervolume().item()


def _iter_dir(models_root, iteration):
    return _ensure_dir(os.path.join(models_root, f"iter_{iteration:02d}"))


def save_iteration_snapshot(
    models_root,
    iteration,
    model_tc,
    model_mod,
    y_tc_scaler,
    y_mod_scaler,
    feature_args_tc,
    feature_args_mod,
    gp_args_tc,
    gp_args_mod,
    train_kwargs_tc,
    train_kwargs_mod,
    scaler_X,
    hv_value,
):
    it_dir = _iter_dir(models_root, iteration)

    torch.save(model_tc.state_dict(), os.path.join(it_dir, "tc_dkl_state.pt"))
    torch.save(model_mod.state_dict(), os.path.join(it_dir, "modulus_dkl_state.pt"))

    joblib.dump(
        {
            "scaler": y_tc_scaler,
            "feature_args": feature_args_tc,
            "gp_args": gp_args_tc,
            "train_kwargs": train_kwargs_tc,
        },
        os.path.join(it_dir, "tc_aux.pkl"),
    )
    joblib.dump(
        {
            "scaler": y_mod_scaler,
            "feature_args": feature_args_mod,
            "gp_args": gp_args_mod,
            "train_kwargs": train_kwargs_mod,
        },
        os.path.join(it_dir, "modulus_aux.pkl"),
    )
    joblib.dump(scaler_X, os.path.join(it_dir, "scaler_X.pkl"))

    with open(os.path.join(it_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "iteration": iteration,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "hv": float(hv_value),
                "files": [
                    "tc_dkl_state.pt",
                    "modulus_dkl_state.pt",
                    "tc_aux.pkl",
                    "modulus_aux.pkl",
                    "scaler_X.pkl",
                ],
            },
            f,
            indent=2,
        )


def copy_snapshot(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def export_all_candidates(df_candidates, output_file):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_candidates.to_csv(output_file, index=False)


def extract_pareto_solutions(df_candidates, output_csv, mod_col="Modulus", tc_col="TC"):
    objectives = torch.as_tensor(
        np.column_stack([-df_candidates[mod_col].values, df_candidates[tc_col].values]),
        dtype=torch.double,
    )
    pareto_df = df_candidates.loc[is_non_dominated(objectives).cpu().numpy()].copy()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pareto_df.to_csv(output_csv, index=False)
    return pareto_df


def run_mobo_loop(
    init_csv,
    unlabeled_csv,
    w2v_path,
    feature_args_tc,
    feature_args_mod,
    gp_args_tc,
    gp_args_mod,
    y_tc_scaler,
    y_mod_scaler,
    train_kwargs_tc,
    train_kwargs_mod,
    n_iter=60,
    q=4,
    models_root="models",
    results_root="results",
    truth_csv="data/ground_truth.csv",
    ref_point_eps=6.0,
):
    _ensure_dir(models_root)
    _ensure_dir(results_root)

    w2v_model = load_word2vec(w2v_path)
    X_train, df_train, scaler_X = prepare_data_train(init_csv, w2v_model)
    X_unlabeled, df_unlabeled = load_unlabeled(unlabeled_csv, w2v_model, scaler_X)

    feature_args_tc = {**feature_args_tc, "in_dim": int(X_train.shape[1])}
    feature_args_mod = {**feature_args_mod, "in_dim": int(X_train.shape[1])}

    tc_z = y_tc_scaler.transform(df_train[["TC"]]).ravel()
    mod_z = y_mod_scaler.transform(df_train[["Modulus"]]).ravel()

    model_tc, model_mod = fit_two_dkl_models(
        X_train,
        tc_z,
        mod_z,
        feature_args_tc,
        feature_args_mod,
        gp_args_tc,
        gp_args_mod,
        train_kwargs_tc,
        train_kwargs_mod,
    )
    mobo_model = build_mobo_surrogate(model_tc, model_mod)

    train_X, train_Y, bounds = get_train_tensors(X_train, tc_z, mod_z, flip_modulus=True)
    ref_point = get_reference_point(train_Y, eps=ref_point_eps)
    X_unlabeled_t = torch.as_tensor(X_unlabeled, dtype=torch.double)

    hvs = [compute_hv(train_Y, ref_point)]
    loop_history = {
        "iter": [0],
        "hv": [float(hvs[0])],
        "best_tc": [float(y_tc_scaler.inverse_transform([[tc_z.max()]])[0, 0])],
        "best_mod": [float(y_mod_scaler.inverse_transform([[mod_z.min()]])[0, 0])],
    }
    all_candidates = []

    print(
        f"[Iter  0] HV={hvs[0]:7.4f} | "
        f"TC_best={loop_history['best_tc'][-1]:.4f} | "
        f"Mod_best={loop_history['best_mod'][-1]:.4f}"
    )

    save_iteration_snapshot(
        models_root,
        0,
        model_tc,
        model_mod,
        y_tc_scaler,
        y_mod_scaler,
        feature_args_tc,
        feature_args_mod,
        gp_args_tc,
        gp_args_mod,
        train_kwargs_tc,
        train_kwargs_mod,
        scaler_X,
        hvs[0],
    )

    for iteration in range(1, n_iter + 1):
        if X_unlabeled_t.size(0) == 0:
            print("[INFO] Candidate pool is empty. Stopping early.")
            break

        t0 = time.monotonic()
        cand_X, idx_list = select_candidates_nehvi_greedy(
            mobo_model,
            train_X,
            bounds,
            ref_point,
            X_unlabeled_t,
            q=q,
        )

        rows = df_unlabeled.iloc[idx_list].reset_index(drop=True)
        tc_true, mod_true = fetch_ground_truth_auto(rows, truth_csv=truth_csv)
        rows["TC"] = tc_true
        rows["Modulus"] = mod_true
        rows["iteration"] = iteration
        all_candidates.append(rows)

        df_unlabeled = df_unlabeled.drop(index=idx_list).reset_index(drop=True)
        X_unlabeled = np.delete(X_unlabeled, idx_list, axis=0)
        X_unlabeled_t = torch.as_tensor(X_unlabeled, dtype=torch.double)

        tc_z_new = y_tc_scaler.transform(np.asarray(tc_true, dtype=float).reshape(-1, 1)).ravel()
        mod_z_new = y_mod_scaler.transform(np.asarray(mod_true, dtype=float).reshape(-1, 1)).ravel()

        X_train = np.vstack([X_train, cand_X.cpu().numpy()])
        tc_z = np.concatenate([tc_z, tc_z_new])
        mod_z = np.concatenate([mod_z, mod_z_new])

        model_tc, model_mod = fit_two_dkl_models(
            X_train,
            tc_z,
            mod_z,
            feature_args_tc,
            feature_args_mod,
            gp_args_tc,
            gp_args_mod,
            train_kwargs_tc,
            train_kwargs_mod,
        )
        mobo_model = build_mobo_surrogate(model_tc, model_mod)

        train_X, train_Y, bounds = get_train_tensors(X_train, tc_z, mod_z, flip_modulus=True)
        current_worst = train_Y.min(dim=0).values
        if not torch.all(ref_point < current_worst):
            raise AssertionError(
                f"Reference point {ref_point.tolist()} is not below current mins {current_worst.tolist()}."
            )

        hv = compute_hv(train_Y, ref_point)
        hvs.append(hv)
        loop_history["iter"].append(iteration)
        loop_history["hv"].append(float(hv))
        loop_history["best_tc"].append(float(y_tc_scaler.inverse_transform([[tc_z.max()]])[0, 0]))
        loop_history["best_mod"].append(float(y_mod_scaler.inverse_transform([[mod_z.min()]])[0, 0]))

        print(
            f"[Iter {iteration:2d}] HV={hv:7.4f} | ΔHV={hv - hvs[-2]:+.4f} | "
            f"TC_best={loop_history['best_tc'][-1]:.4f} | "
            f"Mod_best={loop_history['best_mod'][-1]:.4f} | "
            f"time={time.monotonic() - t0:5.1f}s"
        )

        save_iteration_snapshot(
            models_root,
            iteration,
            model_tc,
            model_mod,
            y_tc_scaler,
            y_mod_scaler,
            feature_args_tc,
            feature_args_mod,
            gp_args_tc,
            gp_args_mod,
            train_kwargs_tc,
            train_kwargs_mod,
            scaler_X,
            hv,
        )

    if all_candidates:
        df_candidates = pd.concat(all_candidates, ignore_index=True)
    else:
        df_candidates = pd.DataFrame(columns=["iteration", "PID", "SMILES", "TC", "Modulus"])

    np.save(os.path.join(results_root, "hv_trace.npy"), np.asarray(hvs, dtype=float))
    joblib.dump(scaler_X, os.path.join(results_root, "scaler_X.pkl"))

    with open(os.path.join(results_root, "loop_history.json"), "w", encoding="utf-8") as f:
        json.dump(loop_history, f, indent=2)

    export_all_candidates(df_candidates, os.path.join(results_root, "all_candidates.csv"))
    if not df_candidates.empty:
        extract_pareto_solutions(df_candidates, os.path.join(results_root, "final_pareto.csv"))

    best_iter = int(np.argmax(hvs))
    best_dir = os.path.join(models_root, f"iter_{best_iter:02d}")
    copy_snapshot(best_dir, os.path.join(models_root, "best"))

    with open(os.path.join(results_root, "s_best.json"), "w", encoding="utf-8") as f:
        json.dump({"best_iter": best_iter, "best_hv": float(hvs[best_iter]), "best_dir": best_dir}, f, indent=2)

    return mobo_model, hvs, scaler_X, train_Y, loop_history, df_candidates, best_iter, best_dir
