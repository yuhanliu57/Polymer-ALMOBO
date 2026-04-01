import pandas as pd


def fetch_ground_truth_auto(rows, truth_csv="data/ground_truth.csv"):
    if "PID" not in rows.columns:
        raise KeyError("Input rows must contain a PID column.")

    truth_df = pd.read_csv(truth_csv, usecols=["PID", "TC", "Modulus"])
    merged = rows[["PID"]].merge(truth_df, on="PID", how="left")

    missing = merged.loc[merged[["TC", "Modulus"]].isna().any(axis=1), "PID"].tolist()
    if missing:
        raise KeyError(f"Missing ground-truth entries for PID(s): {missing}")

    return merged["TC"].tolist(), merged["Modulus"].tolist()
