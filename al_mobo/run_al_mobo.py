from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from al_mobo_loop import run_mobo_loop


def _prepare_target_scalers(init_csv):
    df = pd.read_csv(init_csv, usecols=["TC", "Modulus"])
    y_tc_scaler = StandardScaler().fit(df[["TC"]])
    y_mod_scaler = StandardScaler().fit(df[["Modulus"]])
    return y_tc_scaler, y_mod_scaler


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    init_csv = data_dir / "Initial_set.csv"
    unlabeled_csv = data_dir / "Unlabeled_database.csv"
    w2v_path = data_dir / "POLYINFO_PI1M.pkl"
    truth_csv = data_dir / "ground_truth.csv"

    feature_args_tc = dict(hidden_dims=(192, 128), latent_dim=16, dropout=0.1)
    feature_args_mod = dict(hidden_dims=(224, 192, 128, 32), latent_dim=12, dropout=0.05)

    gp_args_tc = {"kernel_type": "rq"}
    gp_args_mod = {"kernel_type": "matern", "nu": 1.5}

    train_kwargs_tc = dict(adam_lr=3e-4, adam_epochs=200, lbfgs_lr=0.8, lbfgs_iters=40, seed=42)
    train_kwargs_mod = dict(adam_lr=3e-4, adam_epochs=200, lbfgs_lr=0.8, lbfgs_iters=50, seed=42)

    y_tc_scaler, y_mod_scaler = _prepare_target_scalers(init_csv)

    run_mobo_loop(
        init_csv=str(init_csv),
        unlabeled_csv=str(unlabeled_csv),
        w2v_path=str(w2v_path),
        feature_args_tc=feature_args_tc,
        feature_args_mod=feature_args_mod,
        gp_args_tc=gp_args_tc,
        gp_args_mod=gp_args_mod,
        y_tc_scaler=y_tc_scaler,
        y_mod_scaler=y_mod_scaler,
        train_kwargs_tc=train_kwargs_tc,
        train_kwargs_mod=train_kwargs_mod,
        n_iter=60,
        q=4,
        models_root=str(project_root / "models"),
        results_root=str(project_root / "results"),
        truth_csv=str(truth_csv),
        ref_point_eps=6.0,
    )


if __name__ == "__main__":
    main()
