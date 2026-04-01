from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.settings import fast_pred_var
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

AL_MOBO_DIR = Path(__file__).resolve().parents[1] / "al_mobo"
if str(AL_MOBO_DIR) not in sys.path:
    sys.path.insert(0, str(AL_MOBO_DIR))

from dkl_surrogates import ExactDKLModel, FeatureNet, train_exact_dkl_full
from prepare_data import load_word2vec, prepare_data_train, smiles_to_embeddings


def _ensure_dir(path):
    if path and str(path).strip():
        Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _require_columns(df, required, source_name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {source_name}: {missing}")


def fit_target_scalers(init_csv):
    df = pd.read_csv(init_csv)
    _require_columns(df, {"TC", "Modulus"}, init_csv)
    y_tc_scaler = StandardScaler().fit(df["TC"].to_numpy(float).reshape(-1, 1))
    y_mod_scaler = StandardScaler().fit(df["Modulus"].to_numpy(float).reshape(-1, 1))
    return y_tc_scaler, y_mod_scaler


def gaussian_nll_mean(y, mu, sd):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.clip(np.asarray(sd, dtype=float), 1e-12, None)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sd**2) + 0.5 * ((y - mu) ** 2) / (sd**2)))


def _real_space_metrics(y_true_z, y_pred_z, scaler):
    y_true = scaler.inverse_transform(np.asarray(y_true_z).reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(np.asarray(y_pred_z).reshape(-1, 1)).ravel()
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def cross_validate_target(
    X_train,
    y_z,
    feature_args,
    gp_args,
    train_kwargs,
    n_splits=5,
    seed=42,
    flip_sign=False,
):
    y_z = np.asarray(y_z, dtype=float).ravel()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    y_true_z_all = []
    y_pred_z_all = []

    for train_idx, test_idx in kf.split(X_train):
        y_train_model = -y_z[train_idx] if flip_sign else y_z[train_idx]
        model = train_exact_dkl_full(
            X_train[train_idx],
            y_train_model,
            feature_args=feature_args,
            gp_args=gp_args,
            seed=seed,
            **train_kwargs,
        )
        model.eval()
        model.likelihood.eval()

        X_test_t = torch.as_tensor(X_train[test_idx], dtype=torch.double)
        with torch.no_grad(), fast_pred_var():
            post = model.likelihood(model(X_test_t))

        y_pred_model_z = post.mean.cpu().numpy().ravel()
        y_pred_z = -y_pred_model_z if flip_sign else y_pred_model_z

        y_true_z_all.append(y_z[test_idx])
        y_pred_z_all.append(y_pred_z)

    y_true_z = np.concatenate(y_true_z_all)
    y_pred_z = np.concatenate(y_pred_z_all)
    mse_z = mean_squared_error(y_true_z, y_pred_z)

    return {
        "y_true_z": y_true_z,
        "y_pred_z": y_pred_z,
        "r2_z": float(r2_score(y_true_z, y_pred_z)),
        "mse_z": float(mse_z),
        "rmse_z": float(np.sqrt(mse_z)),
    }


def run_cross_validation(
    init_csv,
    w2v_path,
    feature_args_tc,
    feature_args_mod,
    gp_args_tc,
    gp_args_mod,
    train_kwargs_tc,
    train_kwargs_mod,
    n_splits=5,
    seed=42,
):
    w2v_model = load_word2vec(w2v_path)
    X_train, df_train, scaler_X = prepare_data_train(init_csv, w2v_model)
    _require_columns(df_train, {"SMILES", "TC", "Modulus"}, init_csv)

    y_tc_scaler, y_mod_scaler = fit_target_scalers(init_csv)
    tc_z = y_tc_scaler.transform(df_train["TC"].to_numpy(float).reshape(-1, 1)).ravel()
    mod_z = y_mod_scaler.transform(df_train["Modulus"].to_numpy(float).reshape(-1, 1)).ravel()

    feature_args_tc = dict(feature_args_tc, in_dim=int(X_train.shape[1]))
    feature_args_mod = dict(feature_args_mod, in_dim=int(X_train.shape[1]))

    tc = cross_validate_target(
        X_train,
        tc_z,
        feature_args_tc,
        gp_args_tc,
        train_kwargs_tc,
        n_splits=n_splits,
        seed=seed,
        flip_sign=False,
    )
    mod = cross_validate_target(
        X_train,
        mod_z,
        feature_args_mod,
        gp_args_mod,
        train_kwargs_mod,
        n_splits=n_splits,
        seed=seed,
        flip_sign=True,
    )

    tc.update(_real_space_metrics(tc["y_true_z"], tc["y_pred_z"], y_tc_scaler))
    mod.update(_real_space_metrics(mod["y_true_z"], mod["y_pred_z"], y_mod_scaler))

    return {
        "X_train": X_train,
        "df_train": df_train,
        "scaler_X": scaler_X,
        "y_tc_scaler": y_tc_scaler,
        "y_mod_scaler": y_mod_scaler,
        "tc": tc,
        "mod": mod,
    }


def load_snapshot(snapshot_dir, target):
    if target not in {"tc", "mod"}:
        raise ValueError("target must be 'tc' or 'mod'.")

    snapshot_dir = Path(snapshot_dir)
    aux_file = snapshot_dir / ("tc_aux.pkl" if target == "tc" else "modulus_aux.pkl")
    state_file = snapshot_dir / ("tc_dkl_state.pt" if target == "tc" else "modulus_dkl_state.pt")
    scaler_X = joblib.load(snapshot_dir / "scaler_X.pkl")
    aux = joblib.load(aux_file)

    feature_args = dict(aux["feature_args"])
    gp_args = dict(aux["gp_args"])
    in_dim = int(feature_args["in_dim"])

    X_dummy = torch.zeros((1, in_dim), dtype=torch.double)
    y_dummy = torch.zeros(1, dtype=torch.double)
    feature_net = FeatureNet(**feature_args).to(dtype=torch.double)
    likelihood = GaussianLikelihood().to(dtype=torch.double)
    model = ExactDKLModel(X_dummy, y_dummy, likelihood, feature_net, **gp_args).to(dtype=torch.double)
    model.likelihood = likelihood
    model.load_state_dict(torch.load(state_file, map_location="cpu"))
    model.eval()
    model.likelihood.eval()

    return model, aux["scaler"], scaler_X, feature_args, gp_args


def rebuild_train_data_for_iter(iteration, init_csv, all_candidates_csv, w2v_model):
    df_init = pd.read_csv(init_csv)
    _require_columns(df_init, {"SMILES", "TC", "Modulus"}, init_csv)

    X_init = smiles_to_embeddings(df_init["SMILES"].to_numpy(), w2v_model)
    tc_init = df_init["TC"].to_numpy(float)
    mod_init = df_init["Modulus"].to_numpy(float)

    df_candidates = pd.read_csv(all_candidates_csv)
    _require_columns(df_candidates, {"iteration", "SMILES", "TC", "Modulus"}, all_candidates_csv)
    df_prev = df_candidates[df_candidates["iteration"] <= iteration].copy()

    if df_prev.empty:
        X_prev = np.zeros((0, X_init.shape[1]), dtype=float)
        tc_prev = np.zeros((0,), dtype=float)
        mod_prev = np.zeros((0,), dtype=float)
    else:
        X_prev = smiles_to_embeddings(df_prev["SMILES"].to_numpy(), w2v_model)
        tc_prev = df_prev["TC"].to_numpy(float)
        mod_prev = df_prev["Modulus"].to_numpy(float)

    X_raw = np.vstack([X_init, X_prev])
    tc_true = np.concatenate([tc_init, tc_prev])
    mod_true = np.concatenate([mod_init, mod_prev])

    return X_raw, tc_true, mod_true


def predict_iteration_batch(iteration, models_root, init_csv, all_candidates_csv, w2v_model):
    if iteration <= 0:
        return None

    snapshot_dir = Path(models_root) / f"iter_{iteration - 1:02d}"
    if not snapshot_dir.is_dir():
        return None

    df_candidates = pd.read_csv(all_candidates_csv)
    _require_columns(df_candidates, {"iteration", "SMILES", "TC", "Modulus"}, all_candidates_csv)
    batch = df_candidates[df_candidates["iteration"] == iteration].copy()
    if batch.empty:
        return None

    model_tc, y_tc_scaler, scaler_X_tc, _, _ = load_snapshot(snapshot_dir, "tc")
    model_mod, y_mod_scaler, scaler_X_mod, _, _ = load_snapshot(snapshot_dir, "mod")

    X_train_raw, tc_train, mod_train = rebuild_train_data_for_iter(
        iteration - 1,
        init_csv,
        all_candidates_csv,
        w2v_model,
    )

    X_train_tc = torch.as_tensor(scaler_X_tc.transform(X_train_raw), dtype=torch.double)
    X_train_mod = torch.as_tensor(scaler_X_mod.transform(X_train_raw), dtype=torch.double)
    tc_train_z = torch.as_tensor(y_tc_scaler.transform(tc_train.reshape(-1, 1)).ravel(), dtype=torch.double)
    mod_train_model_z = torch.as_tensor(-y_mod_scaler.transform(mod_train.reshape(-1, 1)).ravel(), dtype=torch.double)

    model_tc.set_train_data(inputs=X_train_tc, targets=tc_train_z, strict=False)
    model_mod.set_train_data(inputs=X_train_mod, targets=mod_train_model_z, strict=False)
    model_tc.eval()
    model_mod.eval()
    model_tc.likelihood.eval()
    model_mod.likelihood.eval()

    X_batch_raw = smiles_to_embeddings(batch["SMILES"].to_numpy(), w2v_model)
    X_batch_tc = torch.as_tensor(scaler_X_tc.transform(X_batch_raw), dtype=torch.double)
    X_batch_mod = torch.as_tensor(scaler_X_mod.transform(X_batch_raw), dtype=torch.double)

    with torch.no_grad(), fast_pred_var():
        post_tc = model_tc.likelihood(model_tc(X_batch_tc))
        mu_tc_z = post_tc.mean.cpu().numpy().ravel()
        sd_tc_z = post_tc.variance.sqrt().cpu().numpy().ravel()

        post_mod = model_mod.likelihood(model_mod(X_batch_mod))
        mu_mod_model_z = post_mod.mean.cpu().numpy().ravel()
        sd_mod_z = post_mod.variance.sqrt().cpu().numpy().ravel()

    mu_mod_z = -mu_mod_model_z

    y_tc = batch["TC"].to_numpy(float)
    y_mod = batch["Modulus"].to_numpy(float)
    y_tc_z = y_tc_scaler.transform(y_tc.reshape(-1, 1)).ravel()
    y_mod_z = y_mod_scaler.transform(y_mod.reshape(-1, 1)).ravel()

    mu_tc = y_tc_scaler.inverse_transform(mu_tc_z.reshape(-1, 1)).ravel()
    sd_tc = sd_tc_z * float(y_tc_scaler.scale_[0])
    mu_mod = y_mod_scaler.inverse_transform(mu_mod_z.reshape(-1, 1)).ravel()
    sd_mod = sd_mod_z * float(y_mod_scaler.scale_[0])

    return {
        "iteration": int(iteration),
        "n": int(len(batch)),
        "batch": batch,
        "y_tc": y_tc,
        "mu_tc": mu_tc,
        "sd_tc": sd_tc,
        "y_mod": y_mod,
        "mu_mod": mu_mod,
        "sd_mod": sd_mod,
        "y_tc_z": y_tc_z,
        "mu_tc_z": mu_tc_z,
        "sd_tc_z": sd_tc_z,
        "y_mod_z": y_mod_z,
        "mu_mod_z": mu_mod_z,
        "sd_mod_z": sd_mod_z,
    }


def predict_batch_with_snapshot(iteration, models_root, init_csv, all_candidates_csv, w2v_model):
    batch_out = predict_iteration_batch(iteration, models_root, init_csv, all_candidates_csv, w2v_model)
    if batch_out is None:
        return None

    return {
        "iteration": batch_out["iteration"],
        "n": batch_out["n"],
        "RMSE_TC": float(np.sqrt(np.mean((batch_out["y_tc"] - batch_out["mu_tc"]) ** 2))),
        "RMSE_Modulus": float(np.sqrt(np.mean((batch_out["y_mod"] - batch_out["mu_mod"]) ** 2))),
        "NLL_TC": gaussian_nll_mean(batch_out["y_tc"], batch_out["mu_tc"], batch_out["sd_tc"]),
        "NLL_Modulus": gaussian_nll_mean(batch_out["y_mod"], batch_out["mu_mod"], batch_out["sd_mod"]),
        "NLL_TC_Z": gaussian_nll_mean(batch_out["y_tc_z"], batch_out["mu_tc_z"], batch_out["sd_tc_z"]),
        "NLL_Modulus_Z": gaussian_nll_mean(batch_out["y_mod_z"], batch_out["mu_mod_z"], batch_out["sd_mod_z"]),
    }


def _weighted_moving_average(values, weights, window):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    out = np.zeros_like(values)

    for i in range(len(values)):
        lo = max(0, i - window + 1)
        values_i = values[lo : i + 1]
        weights_i = weights[lo : i + 1]
        weights_i = weights_i / np.maximum(weights_i.sum(), 1e-12)
        out[i] = float(np.dot(values_i, weights_i))

    return out


def _ewma(values, alpha):
    values = np.asarray(values, dtype=float)
    out = np.zeros_like(values)
    out[0] = values[0]

    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]

    return out


def _plot_learning_curve_panel(x, raw_tc, wma_tc, ewma_tc, raw_mod, wma_mod, ewma_mod, ylabel, save_path, wma_window, alpha_ewma):
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(x, raw_tc, marker="o", lw=0.8, alpha=0.20, label="TC (raw)")
    ax.plot(x, wma_tc, lw=2.2, label=f"TC (WMA W={wma_window})")
    ax.plot(x, ewma_tc, lw=2.0, ls="--", label=f"TC (EWMA α={alpha_ewma})")
    ax.plot(x, raw_mod, marker="s", lw=0.8, alpha=0.20, label="Modulus (raw)")
    ax.plot(x, wma_mod, lw=2.2, label=f"Modulus (WMA W={wma_window})")
    ax.plot(x, ewma_mod, lw=2.0, ls="--", label=f"Modulus (EWMA α={alpha_ewma})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.legend(ncols=2, frameon=False)
    fig.tight_layout()
    _ensure_dir(Path(save_path).parent)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def compute_and_plot_learning_curves(
    models_root,
    init_csv,
    all_candidates_csv,
    w2v_model,
    out_csv,
    out_rmse_png,
    out_nll_png,
    wma_window=5,
    alpha_ewma=0.35,
    out_csv_smoothed=None,
    use_zspace_for_nll=True,
):
    df_candidates = pd.read_csv(all_candidates_csv)
    _require_columns(df_candidates, {"iteration"}, all_candidates_csv)
    it_max = int(df_candidates["iteration"].max())

    rows = []
    for iteration in range(1, it_max + 1):
        row = predict_batch_with_snapshot(iteration, models_root, init_csv, all_candidates_csv, w2v_model)
        if row is not None:
            rows.append(row)

    metrics = pd.DataFrame(rows).sort_values("iteration")
    if metrics.empty:
        raise RuntimeError("No prospective batch metrics. Check snapshots and all_candidates.csv.")

    _ensure_dir(Path(out_csv).parent)
    metrics.to_csv(out_csv, index=False)

    n = metrics["n"].to_numpy()
    smoothed = {
        "iteration": metrics["iteration"].to_numpy(),
        "n": n,
        "RMSE_TC_raw": metrics["RMSE_TC"].to_numpy(),
        "RMSE_TC_WMA": _weighted_moving_average(metrics["RMSE_TC"].to_numpy(), n, wma_window),
        "RMSE_TC_EWMA": _ewma(metrics["RMSE_TC"].to_numpy(), alpha_ewma),
        "RMSE_Modulus_raw": metrics["RMSE_Modulus"].to_numpy(),
        "RMSE_Modulus_WMA": _weighted_moving_average(metrics["RMSE_Modulus"].to_numpy(), n, wma_window),
        "RMSE_Modulus_EWMA": _ewma(metrics["RMSE_Modulus"].to_numpy(), alpha_ewma),
    }

    nll_tc_col = "NLL_TC_Z" if use_zspace_for_nll else "NLL_TC"
    nll_mod_col = "NLL_Modulus_Z" if use_zspace_for_nll else "NLL_Modulus"
    smoothed[f"{nll_tc_col}_raw"] = metrics[nll_tc_col].to_numpy()
    smoothed[f"{nll_tc_col}_WMA"] = _weighted_moving_average(metrics[nll_tc_col].to_numpy(), n, wma_window)
    smoothed[f"{nll_tc_col}_EWMA"] = _ewma(metrics[nll_tc_col].to_numpy(), alpha_ewma)
    smoothed[f"{nll_mod_col}_raw"] = metrics[nll_mod_col].to_numpy()
    smoothed[f"{nll_mod_col}_WMA"] = _weighted_moving_average(metrics[nll_mod_col].to_numpy(), n, wma_window)
    smoothed[f"{nll_mod_col}_EWMA"] = _ewma(metrics[nll_mod_col].to_numpy(), alpha_ewma)

    smoothed = pd.DataFrame(smoothed)
    if out_csv_smoothed is None:
        out_csv_smoothed = str(Path(out_csv).with_name("prospective_learning_curves_smoothed.csv"))
    smoothed.to_csv(out_csv_smoothed, index=False)

    x = metrics["iteration"].to_numpy()
    _plot_learning_curve_panel(
        x,
        smoothed["RMSE_TC_raw"].to_numpy(),
        smoothed["RMSE_TC_WMA"].to_numpy(),
        smoothed["RMSE_TC_EWMA"].to_numpy(),
        smoothed["RMSE_Modulus_raw"].to_numpy(),
        smoothed["RMSE_Modulus_WMA"].to_numpy(),
        smoothed["RMSE_Modulus_EWMA"].to_numpy(),
        "Batch RMSE",
        out_rmse_png,
        wma_window,
        alpha_ewma,
    )
    _plot_learning_curve_panel(
        x,
        smoothed[f"{nll_tc_col}_raw"].to_numpy(),
        smoothed[f"{nll_tc_col}_WMA"].to_numpy(),
        smoothed[f"{nll_tc_col}_EWMA"].to_numpy(),
        smoothed[f"{nll_mod_col}_raw"].to_numpy(),
        smoothed[f"{nll_mod_col}_WMA"].to_numpy(),
        smoothed[f"{nll_mod_col}_EWMA"].to_numpy(),
        "Negative Log-Likelihood (lower is better)",
        out_nll_png,
        wma_window,
        alpha_ewma,
    )

    return metrics, smoothed
