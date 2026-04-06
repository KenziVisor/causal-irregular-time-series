from __future__ import annotations

import pickle

import pandas as pd


CANONICAL_TS_COLUMNS = ["ts_id", "minute", "variable", "value"]
CANONICAL_OC_COLUMNS = ["ts_id", "length_of_stay", "in_hospital_mortality", "subset"]
REQUIRED_OC_COLUMNS = ["ts_id", "in_hospital_mortality"]


def _require_columns(df: pd.DataFrame, required_columns: list[str], df_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def build_canonical_ts(events_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse extracted MIMIC events into the PhysioNet-style time-series schema."""
    _require_columns(events_df, CANONICAL_TS_COLUMNS, "events_df")

    ts = events_df.loc[:, CANONICAL_TS_COLUMNS].copy()
    ts["ts_id"] = ts["ts_id"].astype(str)
    ts["minute"] = pd.to_numeric(ts["minute"], errors="raise").astype(int)
    ts["variable"] = ts["variable"].astype(str)
    ts["value"] = pd.to_numeric(ts["value"], errors="raise").astype(float)

    # Match the PhysioNet artifact's clean, duplicate-free long format.
    ts = ts.groupby(["ts_id", "minute", "variable"], as_index=False, sort=True)["value"].mean()
    return ts.loc[:, CANONICAL_TS_COLUMNS].reset_index(drop=True)


def build_canonical_oc(
    icu_full_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    valid_ts_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a minimal outcomes table keyed only by ts_id.

    Notes:
    - length_of_stay is derived in days from ICU in/out timestamps for compatibility
      with the PhysioNet processed pickle.
    - HADM_ID and SUBJECT_ID stay internal only and are intentionally not exported.
    """
    _require_columns(icu_full_df, ["ts_id", "HADM_ID", "INTIME", "OUTTIME"], "icu_full_df")
    _require_columns(admissions_df, ["HADM_ID", "HOSPITAL_EXPIRE_FLAG"], "admissions_df")

    icu = icu_full_df.loc[:, ["ts_id", "HADM_ID", "INTIME", "OUTTIME"]].copy()
    icu["ts_id"] = icu["ts_id"].astype(str)
    icu["INTIME"] = pd.to_datetime(icu["INTIME"])
    icu["OUTTIME"] = pd.to_datetime(icu["OUTTIME"])
    icu["length_of_stay"] = (icu["OUTTIME"] - icu["INTIME"]).dt.total_seconds() / (24 * 60 * 60)

    admissions = admissions_df.loc[:, ["HADM_ID", "HOSPITAL_EXPIRE_FLAG"]].drop_duplicates(subset=["HADM_ID"])
    oc = icu.loc[:, ["ts_id", "HADM_ID", "length_of_stay"]].merge(admissions, on="HADM_ID", how="left")
    oc = oc.rename(columns={"HOSPITAL_EXPIRE_FLAG": "in_hospital_mortality"})
    oc["in_hospital_mortality"] = pd.to_numeric(oc["in_hospital_mortality"], errors="raise")
    oc["subset"] = "mimic_iii"
    oc = oc.loc[:, CANONICAL_OC_COLUMNS].drop_duplicates(subset=["ts_id"])

    if valid_ts_ids is not None:
        valid_ts_ids = {str(ts_id) for ts_id in valid_ts_ids}
        oc = oc.loc[oc["ts_id"].isin(valid_ts_ids)]

    return oc.sort_values("ts_id").reset_index(drop=True)


def build_ts_ids(ts_df: pd.DataFrame) -> list[str]:
    _require_columns(ts_df, ["ts_id"], "ts_df")
    return sorted(ts_df["ts_id"].astype(str).unique().tolist())


def assert_physionet_compatible_output(
    ts: pd.DataFrame,
    oc: pd.DataFrame,
    ts_ids: list[str],
) -> None:
    payload = [ts, oc, ts_ids]
    if len(payload) != 3:
        raise AssertionError("Processed payload must contain exactly 3 objects: [ts, oc, ts_ids].")

    if not isinstance(ts, pd.DataFrame):
        raise TypeError("ts must be a pandas DataFrame.")
    if list(ts.columns) != CANONICAL_TS_COLUMNS:
        raise AssertionError(f"ts columns must be exactly {CANONICAL_TS_COLUMNS}, got {list(ts.columns)}.")
    if not pd.api.types.is_numeric_dtype(ts["minute"]):
        raise AssertionError("ts.minute must be numeric.")
    if not pd.api.types.is_numeric_dtype(ts["value"]):
        raise AssertionError("ts.value must be numeric.")

    if not isinstance(oc, pd.DataFrame):
        raise TypeError("oc must be a pandas DataFrame.")
    missing_oc_columns = [column for column in REQUIRED_OC_COLUMNS if column not in oc.columns]
    if missing_oc_columns:
        raise AssertionError(f"oc is missing required columns: {missing_oc_columns}.")
    if list(oc.columns) != CANONICAL_OC_COLUMNS:
        raise AssertionError(f"oc columns must be exactly {CANONICAL_OC_COLUMNS}, got {list(oc.columns)}.")
    forbidden_oc_columns = {"HADM_ID", "SUBJECT_ID", "TABLE"} & set(oc.columns)
    if forbidden_oc_columns:
        raise AssertionError(f"oc must not expose MIMIC-specific identifiers: {sorted(forbidden_oc_columns)}.")

    if ts_ids != sorted(ts_ids):
        raise AssertionError("ts_ids must be sorted.")

    ts_ids_from_ts = build_ts_ids(ts)
    if ts_ids != ts_ids_from_ts:
        raise AssertionError("ts_ids must equal sorted(ts.ts_id.unique()).")

    ts_id_set = set(ts_ids_from_ts)
    oc_id_set = set(oc["ts_id"].astype(str))
    if not oc_id_set.issubset(ts_id_set):
        raise AssertionError("All oc.ts_id values must be contained in ts_ids.")
    if set(ts["ts_id"].astype(str)) != ts_id_set:
        raise AssertionError("All ts.ts_id values must be represented in ts_ids.")


def serialize_processed_output(ts: pd.DataFrame, oc: pd.DataFrame, ts_ids: list[str], output_path: str) -> None:
    assert_physionet_compatible_output(ts, oc, ts_ids)
    with open(output_path, "wb") as file:
        pickle.dump([ts, oc, ts_ids], file)
