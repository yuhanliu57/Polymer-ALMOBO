from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.settings import fast_pred_var
from matplotlib.colors import PowerNorm, to_rgba
from sklearn.metrics import mean_squared_error, r2_score


def _ensure_dir(path):
    if path and str(path).strip():
        Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _save_or_show(fig, save_path=None, show=True, dpi=300):
    if save_path is not None:
        _ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def darken_color(color, factor=0.8):
    rgba = np.asarray(to_rgba(color), dtype=float)
    rgba[:3] *= factor
    return np.clip(rgba, 0.0, 1.0)


def _beautify_axes(ax, x0=True, y0=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if x0:
        _, x_hi = ax.get_xlim()
        ax.set_xlim(left=0.0, right=x_hi)
    if y0:
        _, y_hi = ax.get_ylim()
        ax.set_ylim(bottom=0.0, top=y_hi)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


def _inverse_target(values_z, scaler_y, flip_sign=False):
    values_z = np.asarray(values_z, dtype=float).reshape(-1, 1)
    if flip_sign:
        values_z = -values_z
    return scaler_y.inverse_transform(values_z).ravel()


def plot_full_parity(
    model,
    X_train,
    scaler_y,
    label,
    color,
    flip_sign=False,
    save_path=None,
    show=True,
    dpi=300,
):
    model.eval()
    model.likelihood.eval()
    model.feat.eval()

    X = torch.as_tensor(X_train, dtype=torch.double)
    with torch.no_grad(), fast_pred_var():
        post = model.likelihood(model(X))
        mu_model_z = post.mean.cpu().numpy().ravel()
        sd_z = post.variance.sqrt().cpu().numpy().ravel()

    y_pred = _inverse_target(mu_model_z, scaler_y, flip_sign=flip_sign)
    y_true = _inverse_target(model.train_targets.cpu().numpy(), scaler_y, flip_sign=flip_sign)
    y_std = sd_z * float(scaler_y.scale_[0])

    mse = mean_squared_error(y_true, y_pred)
    lim = [min(y_true.min(), y_pred.min()) * 0.95, max(y_true.max(), y_pred.max()) * 1.05]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=24, color=color, alpha=0.7, edgecolors="k", linewidth=0.5)
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="none", ecolor=darken_color(color, 0.6), alpha=0.4)
    ax.plot(lim, lim, ls="--", c="gray", lw=1.5)
    ax.annotate(
        f"R² = {r2_score(y_true, y_pred):.3f}\nMSE = {mse:.3f}",
        xy=(0.05, 0.90),
        xycoords="axes fraction",
        bbox={"boxstyle": "round", "fc": "white", "alpha": 0.6},
    )

    if "TC" in label:
        ax.set_xlabel("MD-calculated TC (W/m·K)")
        ax.set_ylabel("Predicted TC (W/m·K)")
        ax.set_title("TC Full_Data")
    else:
        ax.set_xlabel("MD-calculated Modulus (GPa)")
        ax.set_ylabel("Predicted Modulus (GPa)")
        ax.set_title("Modulus Full_Data")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    fig.tight_layout()
    _save_or_show(fig, save_path=save_path, show=show, dpi=dpi)


def plot_cv_parity(
    y_true_z,
    y_pred_z,
    scaler_y,
    color,
    x_label,
    y_label,
    fig_title,
    save_path=None,
    show=True,
    dpi=600,
):
    y_true = scaler_y.inverse_transform(np.asarray(y_true_z).reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(np.asarray(y_pred_z).reshape(-1, 1)).ravel()
    mse = mean_squared_error(y_true, y_pred)
    lim = [min(y_true.min(), y_pred.min()) * 0.95, max(y_true.max(), y_pred.max()) * 1.05]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=25, color=color, alpha=0.7, edgecolors="k", linewidth=0.5)
    ax.plot(lim, lim, ls="--", c="gray", lw=1.5)
    ax.annotate(
        f"CV R² = {r2_score(y_true, y_pred):.3f}\nCV MSE = {mse:.3f}",
        xy=(0.05, 0.90),
        xycoords="axes fraction",
        bbox={"boxstyle": "round", "fc": "white", "alpha": 0.6},
    )
    ax.set_title(fig_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    fig.tight_layout()
    _save_or_show(fig, save_path=save_path, show=show, dpi=dpi)


def plot_with_band(idx, mu, sd, color, title, ylim, y_label, save_path=None, show=True, dpi=300):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(idx, mu, ".", color=darken_color(color, 0.9), label="Prediction", ms=5, mew=0, zorder=2)
    ax.fill_between(idx, mu - sd, mu + sd, color=color, alpha=0.4, label="Uncertainty", zorder=1)
    ax.set_ylim(ylim)
    ax.set_xlabel("Polymer serial number")
    ax.set_ylabel(y_label)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path=save_path, show=show, dpi=dpi)


def batch_predict_and_plot(
    df,
    X_unlabeled,
    model,
    scaler_y,
    tag,
    color,
    ylim,
    y_label,
    flip_sign=False,
    output_path=None,
    save_path=None,
    show=True,
    dpi=300,
):
    model.eval()
    model.likelihood.eval()
    model.feat.eval()

    X = torch.as_tensor(X_unlabeled, dtype=torch.double)
    with torch.no_grad(), fast_pred_var():
        post = model.likelihood(model(X))
        mu_model_z = post.mean.cpu().numpy().ravel()
        sd_z = post.variance.sqrt().cpu().numpy().ravel()

    mu_real = _inverse_target(mu_model_z, scaler_y, flip_sign=flip_sign)
    std_real = sd_z * float(scaler_y.scale_[0])

    df_out = df.copy()
    df_out[f"{tag}_pred"] = mu_real
    df_out[f"{tag}_std"] = std_real
    if output_path is not None:
        _ensure_dir(Path(output_path).parent)
        df_out.to_csv(output_path, index=False)

    idx = np.arange(len(mu_real))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, mu_real, ".", color=darken_color(color, 0.9), label="Prediction", ms=5, mew=0, zorder=2)
    ax.fill_between(idx, mu_real - std_real, mu_real + std_real, color=color, alpha=0.4, label="Uncertainty", zorder=1)
    ax.set_ylim(ylim)
    ax.set_xlabel("Polymer serial number")
    ax.set_ylabel(y_label)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_or_show(fig, save_path=save_path, show=show, dpi=dpi)

    return df_out


def plot_candidates_over_iterations(df, ref_point_real, tc_scale=1.0, save_path=None, show=True, dpi=300):
    fig, ax = plt.subplots(figsize=(7, 5))
    vmin = int(df["iteration"].min())
    vmax = int(df["iteration"].max())
    norm = PowerNorm(gamma=0.65, vmin=vmin, vmax=vmax)

    scatter = ax.scatter(
        df["Modulus"],
        df["TC"] * tc_scale,
        c=df["iteration"],
        cmap="cividis",
        norm=norm,
        alpha=0.8,
        s=40,
        edgecolors="none",
    )

    Y = torch.as_tensor(np.column_stack([-df["Modulus"].to_numpy(), df["TC"].to_numpy()]), dtype=torch.double)
    mask = is_non_dominated(Y).cpu().numpy()
    pareto_df = df.loc[mask]

    ax.scatter(
        pareto_df["Modulus"],
        pareto_df["TC"] * tc_scale,
        marker="*",
        s=220,
        c="#A40000",
        edgecolors="white",
    )
    ax.scatter(
        float(ref_point_real[0]),
        float(ref_point_real[1]) * tc_scale,
        c="#1f3a93",
        marker="X",
        s=90,
        edgecolors="white",
    )

    ax.set_xlabel("Bulk Modulus (GPa)")
    ax.set_ylabel("Thermal Conductivity (W/m·K)" if tc_scale == 1.0 else f"{tc_scale:g} × Thermal Conductivity (W/m·K)")
    ax.grid(False)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Iteration")
    cbar.set_ticks(np.linspace(vmin, vmax, 7, dtype=int))

    fig.tight_layout()
    _save_or_show(fig, save_path=save_path, show=show, dpi=dpi)


def plot_snapshot(ax, df_candidates, iteration, ref_point_real, xlim=None, ylim=None, show_left_axis=True, show_legend=False):
    df_upto = df_candidates.loc[df_candidates["iteration"] <= iteration, ["Modulus", "TC"]].copy()
    ax.scatter(df_upto["Modulus"], df_upto["TC"], c="#BFBFBF", alpha=1.0, s=100, edgecolors="none")

    Y = torch.as_tensor(np.column_stack([-df_upto["Modulus"].to_numpy(), df_upto["TC"].to_numpy()]), dtype=torch.double)
    mask = is_non_dominated(Y).cpu().numpy()
    pareto_df = df_upto.loc[mask]

    ax.scatter(
        pareto_df["Modulus"],
        pareto_df["TC"],
        c="#A40000",
        marker="*",
        s=550,
        edgecolors="white",
        label="Pareto Front" if show_legend else None,
    )
    ax.scatter(
        float(ref_point_real[0]),
        float(ref_point_real[1]),
        c="#1f3a93",
        marker="X",
        s=225,
        edgecolors="white",
        label="Reference Point",
    )

    if not show_left_axis:
        ax.tick_params(axis="y", labelleft=False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(False)
    if show_legend:
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=18)


def plot_running_best_hv(hv_list, save_path=None, show=True, dpi=300, start_at_zero=False):
    hv = np.asarray(hv_list, dtype=float).ravel()
    if hv.ndim != 1 or hv.size < 1:
        raise ValueError("hv_list must be a 1D array with at least one element.")
    if not np.all(np.isfinite(hv)):
        raise ValueError("hv_list contains non-finite values.")

    running_best = np.maximum.accumulate(hv)
    denom = max(running_best[-1] - running_best[0], 1e-12)
    running_best_pct = 100.0 * (running_best - running_best[0]) / denom
    iteration = np.arange(len(running_best))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(iteration, running_best_pct, where="post")
    ax.plot(iteration, running_best_pct, "o", alpha=0.9)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best hypervolume (%)")
    if start_at_zero:
        _, y_hi = ax.get_ylim()
        ax.set_ylim(bottom=0.0, top=y_hi)
    _beautify_axes(ax, x0=True, y0=False)
    fig.tight_layout()
    _save_or_show(fig, save_path=save_path, show=show, dpi=dpi)


def plot_delta_hv_from_trace(hv_list, log=False, eps=1e-12, save_path=None, show=True, dpi=300):
    hv = np.asarray(hv_list, dtype=float).ravel()
    if hv.ndim != 1 or hv.size < 2:
        raise ValueError("hv_list must be a 1D array with at least two elements.")
    if not np.all(np.isfinite(hv)):
        raise ValueError("hv_list contains non-finite values.")
    if log and eps <= 0:
        raise ValueError("eps must be positive for log scaling.")

    running_best = np.maximum.accumulate(hv)
    delta_hv = np.maximum(np.diff(running_best), 0.0)
    iteration = np.arange(1, len(hv))

    fig, ax = plt.subplots(figsize=(7, 4))
    if log:
        ax.bar(iteration, np.log10(delta_hv + eps), width=0.8, align="center", color="#1f77b4", edgecolor="none")
        ax.set_ylabel(r"log$_{10}$(ΔHV + ε)")
    else:
        ax.bar(iteration, delta_hv, width=0.8, align="center", color="#1f77b4", edgecolor="none")
        ax.set_ylabel("Hypervolume improvement (ΔHV)")
    ax.set_xlabel("Iteration")
    _beautify_axes(ax, x0=True, y0=True)
    fig.tight_layout()
    _save_or_show(fig, save_path=save_path, show=show, dpi=dpi)


def export_hv_data_for_plots(hv_list, out_dir="results/hv_export", eps=1e-12):
    hv = np.asarray(hv_list, dtype=float).ravel()
    if hv.ndim != 1 or hv.size < 2 or not np.all(np.isfinite(hv)):
        raise ValueError("hv_list must be a 1D finite array with length >= 2.")

    running_best = np.maximum.accumulate(hv)
    iteration_rb = np.arange(len(hv))
    denom = max(running_best[-1] - running_best[0], 1e-12)
    running_best_pct = 100.0 * (running_best - running_best[0]) / denom
    iteration_delta = np.arange(1, len(hv))
    delta_hv = np.maximum(np.diff(running_best), 0.0)
    delta_hv_log = np.log10(delta_hv + eps)

    _ensure_dir(out_dir)
    np.savetxt(
        Path(out_dir) / "hv_running_best.csv",
        np.column_stack([iteration_rb, hv, running_best, running_best_pct]),
        delimiter=",",
        header="iteration,hv_abs,running_best_hv_abs,running_best_hv_pct",
        comments="",
    )
    np.savetxt(
        Path(out_dir) / "hv_delta_linear.csv",
        np.column_stack([iteration_delta, delta_hv]),
        delimiter=",",
        header="iteration,delta_hv",
        comments="",
    )
    np.savetxt(
        Path(out_dir) / "hv_delta_log.csv",
        np.column_stack([iteration_delta, delta_hv_log, np.full_like(delta_hv_log, eps, dtype=float)]),
        delimiter=",",
        header="iteration,delta_hv_log10_eps,eps",
        comments="",
    )


def plot_single_objective_history(best_mod_list, best_tc_list, save_path=None, show=True, dpi=300):
    iteration = np.arange(1, len(best_mod_list) + 1)

    fig, ax_left = plt.subplots(figsize=(7, 5))
    ax_left.plot(iteration, best_mod_list, "o--", c="blue")
    ax_left.set_xlabel("Iteration")
    ax_left.set_ylabel("Best Modulus (GPa)", color="blue")
    ax_left.tick_params(axis="y", labelcolor="blue")
    ax_left.grid(True)

    ax_right = ax_left.twinx()
    ax_right.plot(iteration, best_tc_list, "s--", c="red")
    ax_right.set_ylabel("Best TC (W/m·K)", color="red")
    ax_right.tick_params(axis="y", labelcolor="red")

    fig.tight_layout()
    _save_or_show(fig, save_path=save_path, show=show, dpi=dpi)
