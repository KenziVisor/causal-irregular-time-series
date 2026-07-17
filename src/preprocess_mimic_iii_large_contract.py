from __future__ import annotations

import math
import numbers
import pickle
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


CANONICAL_TS_COLUMNS = ["ts_id", "minute", "variable", "value"]
CANONICAL_OC_COLUMNS = ["ts_id", "length_of_stay", "in_hospital_mortality", "subset"]
REQUIRED_OC_COLUMNS = ["ts_id", "in_hospital_mortality"]
_CANONICAL_INTEGER_TEXT = re.compile(r"^(?:0|[1-9][0-9]*)(?:[.]0+)?$")
_MAX_EXACT_FLOAT_INTEGER = 2**53 - 1


def _require_columns(df: pd.DataFrame, required_columns: Sequence[str], df_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def canonicalize_mimic_id_scalar(value: object, *, field_name: str = "MIMIC identifier") -> str:
    """Return one lossless decimal representation for an ICU or admission identifier."""
    if value is None or value is pd.NA:
        raise ValueError(f"{field_name} contains a missing value.")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must not contain booleans.")

    if isinstance(value, numbers.Integral):
        integer = int(value)
        if integer < 0:
            raise ValueError(f"{field_name} must be a non-negative integer identifier.")
        return str(integer)

    if isinstance(value, numbers.Real):
        numeric = float(value)
        if math.isnan(numeric):
            raise ValueError(f"{field_name} contains a missing value.")
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name} must be finite.")
        if numeric < 0 or (numeric == 0 and math.copysign(1.0, numeric) < 0):
            raise ValueError(f"{field_name} must be a non-negative integer identifier.")
        if numeric > _MAX_EXACT_FLOAT_INTEGER:
            raise ValueError(
                f"{field_name} exceeds the exact IEEE-754 integer range. "
                "Provide the identifier as plain decimal text or an integer."
            )
        if not numeric.is_integer():
            raise ValueError(f"{field_name} must not contain fractional identifiers.")
        return str(int(numeric))

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} contains a missing value.")
        if _CANONICAL_INTEGER_TEXT.fullmatch(stripped) is None:
            raise ValueError(
                f"{field_name} must use plain, unpadded non-negative integer text "
                "with only an optional .0 suffix."
            )
        return stripped.split(".", maxsplit=1)[0]

    try:
        is_missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        is_missing = False
    if is_missing:
        raise ValueError(f"{field_name} contains a missing value.")
    raise TypeError(f"{field_name} contains an unsupported value type: {type(value).__name__}.")


def canonicalize_mimic_id_series(
    series: pd.Series,
    *,
    field_name: str | None = None,
) -> pd.Series:
    """Strictly canonicalize a complete MIMIC identifier series."""
    label = field_name or str(series.name or "MIMIC identifier")
    normalized: list[str] = []
    for row_index, value in series.items():
        try:
            normalized.append(canonicalize_mimic_id_scalar(value, field_name=label))
        except (TypeError, ValueError) as error:
            raise type(error)(f"{error} Invalid row index: {row_index!r}.") from error
    return pd.Series(normalized, index=series.index, dtype="object", name=series.name)


def canonicalize_stay_id_series(series: pd.Series) -> pd.Series:
    """Backward-compatible name for strict MIMIC identifier canonicalization."""
    return canonicalize_mimic_id_series(series, field_name=str(series.name or "ts_id"))


def canonicalize_binary_mortality_series(
    series: pd.Series,
    *,
    field_name: str = "HOSPITAL_EXPIRE_FLAG",
) -> pd.Series:
    """Validate a complete finite binary mortality series and return integer labels."""
    if series.empty:
        return pd.Series(index=series.index, dtype="int64", name=series.name)
    if series.isna().any():
        raise ValueError(f"{field_name} contains missing mortality labels.")
    if series.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(f"{field_name} must contain numeric 0/1 labels, not booleans.")

    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must contain numeric 0/1 mortality labels.") from error
    numeric_values = numeric.to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError(f"{field_name} must contain finite mortality labels.")
    if not numeric.isin([0, 1]).all():
        raise ValueError(f"{field_name} must contain only binary 0/1 mortality labels.")
    return numeric.astype("int64").rename(series.name)


def collapse_identical_rows_or_raise(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    frame_name: str,
) -> pd.DataFrame:
    """Collapse duplicate keys only when all declared values are identical."""
    key_columns = list(key_columns)
    value_columns = list(value_columns)
    _require_columns(frame, [*key_columns, *value_columns], frame_name)

    duplicate_rows = frame.loc[frame.duplicated(subset=key_columns, keep=False)]
    conflicting_groups = 0
    if not duplicate_rows.empty:
        grouping_key: str | list[str] = (
            key_columns[0] if len(key_columns) == 1 else key_columns
        )
        for _, group in duplicate_rows.groupby(grouping_key, sort=False, dropna=False):
            if len(group.loc[:, value_columns].drop_duplicates()) > 1:
                conflicting_groups += 1
    if conflicting_groups:
        raise ValueError(
            f"{frame_name} contains {conflicting_groups} duplicate-key group(s) "
            "with conflicting values."
        )
    return frame.drop_duplicates(subset=[*key_columns, *value_columns], keep="first").copy()


def canonicalize_cohort_id_series(
    values: pd.Series | Sequence[object],
    *,
    cohort_name: str,
    require_unique: bool = True,
) -> pd.Series:
    """Canonicalize a nonempty cohort and optionally reject semantic duplicates."""
    source = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype="object")
    normalized = canonicalize_mimic_id_series(
        source,
        field_name=f"{cohort_name}.ts_id",
    ).reset_index(drop=True)
    if normalized.empty:
        raise ValueError(f"{cohort_name} cohort must not be empty.")
    if require_unique:
        duplicate_mask = normalized.duplicated(keep=False)
        if duplicate_mask.any():
            raise ValueError(
                f"{cohort_name} contains duplicate ts_id values after canonicalization. "
                f"Duplicate row count={int(duplicate_mask.sum())}."
            )
    return normalized


def canonicalize_unique_id_frame(
    frame: pd.DataFrame,
    *,
    frame_name: str,
    id_column: str = "ts_id",
) -> pd.DataFrame:
    """Collapse semantically identical ID rows and reject conflicting duplicates."""
    _require_columns(frame, [id_column], frame_name)
    normalized = frame.copy()
    canonical_ids = canonicalize_cohort_id_series(
        normalized[id_column],
        cohort_name=frame_name,
        require_unique=False,
    )
    normalized[id_column] = canonical_ids.set_axis(normalized.index)

    value_columns = [column for column in normalized.columns if column != id_column]
    if not value_columns:
        return normalized.drop_duplicates(subset=[id_column], keep="first").reset_index(
            drop=True
        )
    try:
        normalized = collapse_identical_rows_or_raise(
            normalized,
            key_columns=[id_column],
            value_columns=value_columns,
            frame_name=frame_name,
        )
    except ValueError as error:
        raise ValueError(
            f"{frame_name} contains conflicting duplicate {id_column} rows."
        ) from error
    return normalized.reset_index(drop=True)


def assert_exact_id_cohort(
    reference_ids: pd.Series | Sequence[object],
    candidate_ids: pd.Series | Sequence[object],
    *,
    reference_name: str,
    candidate_name: str,
) -> None:
    """Require two unique ID cohorts to be identical, allowing only reordering."""
    reference = canonicalize_cohort_id_series(
        reference_ids,
        cohort_name=reference_name,
    )
    candidate = canonicalize_cohort_id_series(
        candidate_ids,
        cohort_name=candidate_name,
    )
    reference_set = set(reference.tolist())
    candidate_set = set(candidate.tolist())
    missing_count = len(reference_set - candidate_set)
    extra_count = len(candidate_set - reference_set)
    if missing_count or extra_count:
        raise ValueError(
            f"{candidate_name} cohort does not match {reference_name}: "
            f"missing_count={missing_count}; extra_count={extra_count}."
        )


def build_canonical_ts(events_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse extracted MIMIC events into the PhysioNet-style time-series schema."""
    _require_columns(events_df, CANONICAL_TS_COLUMNS, "events_df")

    ts = events_df.loc[:, CANONICAL_TS_COLUMNS].copy()
    ts["ts_id"] = canonicalize_mimic_id_series(ts["ts_id"], field_name="events_df.ts_id")
    ts["minute"] = pd.to_numeric(ts["minute"], errors="raise").astype(int)
    ts["variable"] = ts["variable"].astype(str)
    ts["value"] = pd.to_numeric(ts["value"], errors="raise").astype(float)

    # Match the PhysioNet artifact's clean, duplicate-free long format.
    ts = ts.groupby(["ts_id", "minute", "variable"], as_index=False, sort=True)["value"].mean()
    return ts.loc[:, CANONICAL_TS_COLUMNS].reset_index(drop=True)


def build_canonical_oc(
    icu_full_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    valid_ts_ids: Sequence[object] | None = None,
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
    icu["ts_id"] = canonicalize_mimic_id_series(icu["ts_id"], field_name="icu_full_df.ts_id")
    icu["HADM_ID"] = canonicalize_mimic_id_series(
        icu["HADM_ID"], field_name="icu_full_df.HADM_ID"
    )
    icu["INTIME"] = pd.to_datetime(icu["INTIME"], errors="raise")
    icu["OUTTIME"] = pd.to_datetime(icu["OUTTIME"], errors="raise")
    if icu[["INTIME", "OUTTIME"]].isna().any().any():
        raise ValueError("icu_full_df contains missing ICU timestamps.")
    icu = collapse_identical_rows_or_raise(
        icu,
        key_columns=["ts_id"],
        value_columns=["HADM_ID", "INTIME", "OUTTIME"],
        frame_name="icu_full_df",
    )
    icu["length_of_stay"] = (icu["OUTTIME"] - icu["INTIME"]).dt.total_seconds() / (24 * 60 * 60)
    if not np.isfinite(icu["length_of_stay"].to_numpy(dtype=float)).all():
        raise ValueError("icu_full_df contains non-finite ICU lengths of stay.")
    if (icu["length_of_stay"] < 0).any():
        raise ValueError("icu_full_df contains an ICU discharge before admission.")

    admissions = admissions_df.loc[:, ["HADM_ID", "HOSPITAL_EXPIRE_FLAG"]].copy()
    admissions["HADM_ID"] = canonicalize_mimic_id_series(
        admissions["HADM_ID"], field_name="admissions_df.HADM_ID"
    )
    admissions["HOSPITAL_EXPIRE_FLAG"] = canonicalize_binary_mortality_series(
        admissions["HOSPITAL_EXPIRE_FLAG"]
    )
    admissions = collapse_identical_rows_or_raise(
        admissions,
        key_columns=["HADM_ID"],
        value_columns=["HOSPITAL_EXPIRE_FLAG"],
        frame_name="admissions_df",
    )

    oc = icu.loc[:, ["ts_id", "HADM_ID", "length_of_stay"]].merge(
        admissions,
        on="HADM_ID",
        how="left",
        validate="many_to_one",
    )
    if len(oc) != len(icu):
        raise AssertionError("The admissions join changed the ICU cohort row count.")
    oc = oc.rename(columns={"HOSPITAL_EXPIRE_FLAG": "in_hospital_mortality"})
    oc["in_hospital_mortality"] = canonicalize_binary_mortality_series(
        oc["in_hospital_mortality"],
        field_name="oc.in_hospital_mortality",
    )
    oc["subset"] = "mimic_iii"
    oc = oc.loc[:, CANONICAL_OC_COLUMNS]

    if valid_ts_ids is not None:
        requested_ids = canonicalize_mimic_id_series(
            pd.Series(list(valid_ts_ids), dtype="object"),
            field_name="valid_ts_ids",
        )
        requested_set = set(requested_ids.tolist())
        if not requested_set:
            raise ValueError("valid_ts_ids must not be empty.")
        missing_count = len(requested_set - set(oc["ts_id"]))
        if missing_count:
            raise ValueError(
                "icu_full_df is missing "
                f"{missing_count} identifier(s) requested by valid_ts_ids."
            )
        oc = oc.loc[oc["ts_id"].isin(requested_set)]

    if oc.empty:
        raise ValueError("The canonical MIMIC outcome cohort must not be empty.")
    return oc.sort_values("ts_id").reset_index(drop=True)


def build_ts_ids(ts_df: pd.DataFrame) -> list[str]:
    _require_columns(ts_df, ["ts_id"], "ts_df")
    ts_ids = canonicalize_mimic_id_series(ts_df["ts_id"], field_name="ts_df.ts_id")
    unique_ids = sorted(ts_ids.unique().tolist())
    if not unique_ids:
        raise ValueError("The canonical MIMIC time-series cohort must not be empty.")
    return unique_ids


def assert_physionet_compatible_output(
    ts: pd.DataFrame,
    oc: pd.DataFrame,
    ts_ids: list[str],
) -> None:
    if not isinstance(ts, pd.DataFrame):
        raise TypeError("ts must be a pandas DataFrame.")
    if list(ts.columns) != CANONICAL_TS_COLUMNS:
        raise AssertionError(
            f"ts columns must be exactly {CANONICAL_TS_COLUMNS}, got {list(ts.columns)}."
        )
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
        raise AssertionError(
            f"oc columns must be exactly {CANONICAL_OC_COLUMNS}, got {list(oc.columns)}."
        )
    forbidden_oc_columns = {"HADM_ID", "SUBJECT_ID", "TABLE"} & set(oc.columns)
    if forbidden_oc_columns:
        raise AssertionError(
            f"oc must not expose MIMIC-specific identifiers: {sorted(forbidden_oc_columns)}."
        )
    try:
        canonicalize_binary_mortality_series(
            oc["in_hospital_mortality"],
            field_name="oc.in_hospital_mortality",
        )
    except ValueError as error:
        raise AssertionError(str(error)) from error

    try:
        canonical_ts_ids = canonicalize_mimic_id_series(
            pd.Series(list(ts_ids), dtype="object"),
            field_name="ts_ids",
        )
    except (TypeError, ValueError) as error:
        raise AssertionError(str(error)) from error
    canonical_ts_id_list = canonical_ts_ids.tolist()
    if canonical_ts_id_list != list(ts_ids):
        raise AssertionError("ts_ids must already use canonical identifier strings.")
    if len(canonical_ts_id_list) != len(set(canonical_ts_id_list)):
        raise AssertionError("ts_ids must not contain duplicates.")
    if canonical_ts_id_list != sorted(canonical_ts_id_list):
        raise AssertionError("ts_ids must be sorted.")

    try:
        canonical_ts_values = canonicalize_mimic_id_series(
            ts["ts_id"],
            field_name="ts.ts_id",
        )
        ts_ids_from_ts = build_ts_ids(ts)
    except (TypeError, ValueError) as error:
        raise AssertionError(str(error)) from error
    if canonical_ts_values.tolist() != ts["ts_id"].tolist():
        raise AssertionError("ts.ts_id must already use canonical identifier strings.")
    if canonical_ts_id_list != ts_ids_from_ts:
        raise AssertionError("ts_ids must equal sorted(ts.ts_id.unique()).")

    try:
        oc_ids = canonicalize_mimic_id_series(oc["ts_id"], field_name="oc.ts_id")
    except (TypeError, ValueError) as error:
        raise AssertionError(str(error)) from error
    if oc_ids.tolist() != oc["ts_id"].tolist():
        raise AssertionError("oc.ts_id must already use canonical identifier strings.")
    if oc_ids.duplicated().any():
        raise AssertionError("oc.ts_id must contain exactly one row per identifier.")

    ts_id_set = set(ts_ids_from_ts)
    oc_id_set = set(oc_ids.tolist())
    if oc_id_set != ts_id_set:
        raise AssertionError(
            "Exported oc.ts_id values must exactly match ts_ids for the canonical MIMIC artifact."
        )


def serialize_processed_output(ts: pd.DataFrame, oc: pd.DataFrame, ts_ids: list[str], output_path: str) -> None:
    assert_physionet_compatible_output(ts, oc, ts_ids)
    with open(output_path, "wb") as file:
        pickle.dump([ts, oc, ts_ids], file)
