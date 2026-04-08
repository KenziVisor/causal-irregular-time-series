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


def canonicalize_stay_id_series(series: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    if series.empty:
        return out

    non_missing = series.notna()
    if not non_missing.any():
        return out

    trimmed = series.loc[non_missing].astype(str).str.strip()
    trimmed = trimmed.mask(trimmed == "", pd.NA)
    numeric = pd.to_numeric(trimmed, errors="coerce")
    integer_like = numeric.notna() & ((numeric % 1).abs() < 1e-9)

    normalized = trimmed.astype("object")
    normalized.loc[integer_like] = numeric.loc[integer_like].astype("Int64").astype(str)
    out.loc[non_missing] = normalized
    return out


def build_canonical_ts(events_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse extracted MIMIC events into the PhysioNet-style time-series schema."""
    _require_columns(events_df, CANONICAL_TS_COLUMNS, "events_df")

    ts = events_df.loc[:, CANONICAL_TS_COLUMNS].copy()
    ts["ts_id"] = canonicalize_stay_id_series(ts["ts_id"])
    if ts["ts_id"].isna().any():
        raise ValueError("events_df contains missing ts_id values after canonicalization.")
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
    icu["ts_id"] = canonicalize_stay_id_series(icu["ts_id"])
    if icu["ts_id"].isna().any():
        raise ValueError("icu_full_df contains missing ts_id values after canonicalization.")
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
        valid_ts_ids_series = canonicalize_stay_id_series(
            pd.Series(list(valid_ts_ids), dtype="object")
        )
        if valid_ts_ids_series.isna().any():
            raise ValueError("valid_ts_ids contains missing values after canonicalization.")
        valid_ts_ids = set(valid_ts_ids_series.tolist())
        oc = oc.loc[oc["ts_id"].isin(valid_ts_ids)]

    return oc.sort_values("ts_id").reset_index(drop=True)


def build_ts_ids(ts_df: pd.DataFrame) -> list[str]:
    _require_columns(ts_df, ["ts_id"], "ts_df")
    ts_ids = canonicalize_stay_id_series(ts_df["ts_id"])
    if ts_ids.isna().any():
        raise ValueError("ts_df contains missing ts_id values after canonicalization.")
    return sorted(ts_ids.unique().tolist())


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
    if oc.empty:
        raise AssertionError("oc must not be empty.")
    if list(oc.columns) != CANONICAL_OC_COLUMNS:
        raise AssertionError(f"oc columns must be exactly {CANONICAL_OC_COLUMNS}, got {list(oc.columns)}.")
    forbidden_oc_columns = {"HADM_ID", "SUBJECT_ID", "TABLE"} & set(oc.columns)
    if forbidden_oc_columns:
        raise AssertionError(f"oc must not expose MIMIC-specific identifiers: {sorted(forbidden_oc_columns)}.")
    if int(pd.to_numeric(oc["in_hospital_mortality"], errors="coerce").notna().sum()) == 0:
        raise AssertionError("oc.in_hospital_mortality must contain at least one non-missing value.")

    canonical_ts_ids = canonicalize_stay_id_series(pd.Series(list(ts_ids), dtype="object"))
    if canonical_ts_ids.isna().any():
        raise AssertionError("ts_ids must not contain missing values.")
    ts_ids = canonical_ts_ids.tolist()
    if ts_ids != sorted(ts_ids):
        raise AssertionError("ts_ids must be sorted.")

    ts_ids_from_ts = build_ts_ids(ts)
    if ts_ids != ts_ids_from_ts:
        raise AssertionError("ts_ids must equal sorted(ts.ts_id.unique()).")

    ts_id_set = set(ts_ids_from_ts)
    oc_ids = canonicalize_stay_id_series(oc["ts_id"])
    if oc_ids.isna().any():
        raise AssertionError("oc.ts_id must not contain missing values.")
    oc_id_set = set(oc_ids.tolist())
    overlap = oc_id_set & ts_id_set
    if not overlap:
        raise AssertionError("oc.ts_id has zero overlap with ts_ids after canonicalization.")
    if not oc_id_set.issubset(ts_id_set):
        raise AssertionError("All oc.ts_id values must be contained in ts_ids.")
    if oc_id_set != ts_id_set:
        raise AssertionError("Exported oc.ts_id values must exactly match ts_ids for the canonical MIMIC artifact.")
    if set(ts["ts_id"].astype(str)) != ts_id_set:
        raise AssertionError("All ts.ts_id values must be represented in ts_ids.")


def serialize_processed_output(ts: pd.DataFrame, oc: pd.DataFrame, ts_ids: list[str], output_path: str) -> None:
    assert_physionet_compatible_output(ts, oc, ts_ids)
    with open(output_path, "wb") as file:
        pickle.dump([ts, oc, ts_ids], file)
