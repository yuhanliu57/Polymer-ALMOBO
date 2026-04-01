from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cross_validation import predict_iteration_batch


def _ensure_dir(path):
    if path and str(path).strip():
        Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _require_columns(df, required, source_name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {source_name}: {missing}")


def ence_from_bins(y, mu, sd, nbins=8):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)

    n = len(y)
    if n == 0:
        return np.nan, pd.DataFrame(columns=["bin", "n", "RMSE", "RMS_sigma", "w", "norm_gap"])

    nbins = max(min(int(nbins), n), 2)
    order = np.argsort(sd)
    parts = np.array_split(order, nbins)

    rows = []
    ence = 0.0
    for k, idx in enumerate(parts, start=1):
        if idx.size == 0:
            continue

        error = y[idx] - mu[idx]
        rmse = float(np.sqrt(np.mean(error**2)))
        rms_sigma = float(np.sqrt(np.mean(sd[idx] ** 2)))
        weight = idx.size / n
        norm_gap = abs(rmse - rms_sigma) / (rms_sigma + 1e-12)

        rows.append(
            {
                "bin": k,
                "n": int(idx.size),
                "RMSE": rmse,
                "RMS_sigma": rms_sigma,
                "w": weight,
                "norm_gap": norm_gap,
            }
        )
        ence += weight * norm_gap

    return float(ence), pd.DataFrame(rows)


def predict_batch_for_uq(iteration, models_root, init_csv, all_candidates_csv, w2v_model):
    return predict_iteration_batch(iteration, models_root, init_csv, all_candidates_csv, w2v_model)


def _select_space(batch_out, space):
    if space == "z":
        return {
            "y_tc": batch_out["y_tc_z"],
            "mu_tc": batch_out["mu_tc_z"],
            "sd_tc": batch_out["sd_tc_z"],
            "y_mod": batch_out["y_mod_z"],
            "mu_mod": batch_out["mu_mod_z"],
            "sd_mod": batch_out["sd_mod_z"],
        }

    return {
        "y_tc": batch_out["y_tc"],
        "mu_tc": batch_out["mu_tc"],
        "sd_tc": batch_out["sd_tc"],
        "y_mod": batch_out["y_mod"],
        "mu_mod": batch_out["mu_mod"],
        "sd_mod": batch_out["sd_mod"],
    }


def compute_and_plot_uq_ence(
    models_root,
    init_csv,
    all_candidates_csv,
    w2v_model,
    nbins_iter,
    nbins_global,
    out_csv_iter,
    out_csv_bins_tc,
    out_csv_bins_mod,
    out_png_curve,
    out_png_reliability,
    rolling_window=1,
    draw_reliability=False,
    space="real",
):
    if space not in {"real", "z"}:
        raise ValueError("space must be 'real' or 'z'.")

    df_candidates = pd.read_csv(all_candidates_csv)
    _require_columns(df_candidates, {"iteration"}, all_candidates_csv)
    it_max = int(df_candidates["iteration"].max())

    batches = []
    for iteration in range(1, it_max + 1):
        batch_out = predict_batch_for_uq(iteration, models_root, init_csv, all_candidates_csv, w2v_model)
        if batch_out is not None:
            batches.append(batch_out)

    if not batches:
        raise RuntimeError("No batches found for ENCE.")

    raw_rows = []
    for batch_out in batches:
        data = _select_space(batch_out, space)
        ence_tc, _ = ence_from_bins(data["y_tc"], data["mu_tc"], data["sd_tc"], nbins=max(2, nbins_iter))
        ence_mod, _ = ence_from_bins(data["y_mod"], data["mu_mod"], data["sd_mod"], nbins=max(2, nbins_iter))
        raw_rows.append(
            {
                "iteration": batch_out["iteration"],
                "ENCE_TC": ence_tc,
                "ENCE_Modulus": ence_mod,
            }
        )

    raw_metrics = pd.DataFrame(raw_rows).sort_values("iteration")

    if rolling_window is None or rolling_window < 2:
        rolling_metrics = raw_metrics.copy()
    else:
        rolling_rows = []
        for iteration in range(1, it_max + 1):
            lo = max(1, iteration - rolling_window + 1)
            pool = [batch_out for batch_out in batches if lo <= batch_out["iteration"] <= iteration]
            if not pool:
                continue

            pooled_tc_y = []
            pooled_tc_mu = []
            pooled_tc_sd = []
            pooled_mod_y = []
            pooled_mod_mu = []
            pooled_mod_sd = []

            for batch_out in pool:
                data = _select_space(batch_out, space)
                pooled_tc_y.append(data["y_tc"])
                pooled_tc_mu.append(data["mu_tc"])
                pooled_tc_sd.append(data["sd_tc"])
                pooled_mod_y.append(data["y_mod"])
                pooled_mod_mu.append(data["mu_mod"])
                pooled_mod_sd.append(data["sd_mod"])

            ence_tc, _ = ence_from_bins(
                np.concatenate(pooled_tc_y),
                np.concatenate(pooled_tc_mu),
                np.concatenate(pooled_tc_sd),
                nbins=max(2, nbins_iter),
            )
            ence_mod, _ = ence_from_bins(
                np.concatenate(pooled_mod_y),
                np.concatenate(pooled_mod_mu),
                np.concatenate(pooled_mod_sd),
                nbins=max(2, nbins_iter),
            )
            rolling_rows.append(
                {
                    "iteration": iteration,
                    "ENCE_TC": ence_tc,
                    "ENCE_Modulus": ence_mod,
                    "window": int(rolling_window),
                }
            )

        rolling_metrics = pd.DataFrame(rolling_rows)

    _ensure_dir(Path(out_csv_iter).parent)
    raw_metrics.to_csv(str(out_csv_iter).replace(".csv", "_raw.csv"), index=False)
    rolling_metrics.to_csv(out_csv_iter, index=False)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(rolling_metrics["iteration"], rolling_metrics["ENCE_TC"], marker="o", lw=1.8, label=f"TC (rolling W={rolling_window})")
    ax.plot(rolling_metrics["iteration"], rolling_metrics["ENCE_Modulus"], marker="s", lw=1.8, ls="--", label=f"Modulus (rolling W={rolling_window})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ENCE (lower is better)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _ensure_dir(Path(out_png_curve).parent)
    fig.savefig(out_png_curve, bbox_inches="tight")
    plt.close(fig)

    if draw_reliability:
        pooled_tc_y = []
        pooled_tc_mu = []
        pooled_tc_sd = []
        pooled_mod_y = []
        pooled_mod_mu = []
        pooled_mod_sd = []

        for batch_out in batches:
            data = _select_space(batch_out, space)
            pooled_tc_y.append(data["y_tc"])
            pooled_tc_mu.append(data["mu_tc"])
            pooled_tc_sd.append(data["sd_tc"])
            pooled_mod_y.append(data["y_mod"])
            pooled_mod_mu.append(data["mu_mod"])
            pooled_mod_sd.append(data["sd_mod"])

        ence_tc_all, tab_tc = ence_from_bins(
            np.concatenate(pooled_tc_y),
            np.concatenate(pooled_tc_mu),
            np.concatenate(pooled_tc_sd),
            nbins=nbins_global,
        )
        ence_mod_all, tab_mod = ence_from_bins(
            np.concatenate(pooled_mod_y),
            np.concatenate(pooled_mod_mu),
            np.concatenate(pooled_mod_sd),
            nbins=nbins_global,
        )

        tab_tc.to_csv(out_csv_bins_tc, index=False)
        tab_mod.to_csv(out_csv_bins_mod, index=False)

        max_value = max(
            tab_tc["RMS_sigma"].max() if not tab_tc.empty else 1.0,
            tab_mod["RMS_sigma"].max() if not tab_mod.empty else 1.0,
            tab_tc["RMSE"].max() if not tab_tc.empty else 1.0,
            tab_mod["RMSE"].max() if not tab_mod.empty else 1.0,
        )
        x = np.linspace(0.0, max_value * 1.05, 200)

        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        ax.plot(x, x, ls=":", lw=1.2, label="Ideal")
        if not tab_tc.empty:
            ax.plot(tab_tc["RMS_sigma"], tab_tc["RMSE"], marker="o", lw=1.6, label=f"TC (ENCE={ence_tc_all:.3f})")
        if not tab_mod.empty:
            ax.plot(tab_mod["RMS_sigma"], tab_mod["RMSE"], marker="s", lw=1.6, ls="--", label=f"Modulus (ENCE={ence_mod_all:.3f})")
        ax.set_xlabel(r"RMS($\hat{\sigma}$) per bin")
        ax.set_ylabel("RMSE per bin")
        ax.legend(frameon=False)
        fig.tight_layout()
        _ensure_dir(Path(out_png_reliability).parent)
        fig.savefig(out_png_reliability, bbox_inches="tight")
        plt.close(fig)

    return rolling_metrics
