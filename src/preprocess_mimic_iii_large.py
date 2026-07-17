from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess_mimic_iii_large_contract import (
    build_canonical_oc,
    build_canonical_ts,
    canonicalize_mimic_id_scalar,
    build_ts_ids,
    canonicalize_binary_mortality_series,
    canonicalize_mimic_id_series,
    collapse_identical_rows_or_raise,
    serialize_processed_output,
)


RAW_DATA_PATH = "../mimiciii"
OUTPUT_PATH = "../data/processed/mimic_iii_ts_oc_ids.pkl"


def log_stage(stage: int, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def log_memory(label: str) -> None:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss / (1024 * 1024)
        print(f"[memory] {label}: {rss:.1f} MB")
    except Exception:
        pass


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_remove(path: Path) -> None:
    if path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def normalize_raw_data_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def canonicalize_identifier_column(
    frame: pd.DataFrame,
    column: str,
    *,
    frame_name: str,
    allow_missing: bool = False,
) -> pd.DataFrame:
    if column not in frame.columns:
        raise KeyError(f"{frame_name} is missing required column: {column}")
    normalized = frame.copy()
    if not allow_missing:
        normalized[column] = canonicalize_mimic_id_series(
            normalized[column],
            field_name=f"{frame_name}.{column}",
        )
        return normalized

    missing = normalized[column].isna()
    canonical = pd.Series(pd.NA, index=normalized.index, dtype="object", name=column)
    if (~missing).any():
        canonical.loc[~missing] = canonicalize_mimic_id_series(
            normalized.loc[~missing, column],
            field_name=f"{frame_name}.{column}",
        )
    normalized[column] = canonical
    return normalized


def normalize_icu_cohort(icu: pd.DataFrame, *, frame_name: str = "icu") -> pd.DataFrame:
    required_columns = ["ICUSTAY_ID", "HADM_ID", "INTIME", "OUTTIME"]
    missing_columns = [column for column in required_columns if column not in icu.columns]
    if missing_columns:
        raise KeyError(f"{frame_name} is missing required columns: {missing_columns}")

    normalized = canonicalize_identifier_column(
        icu,
        "ICUSTAY_ID",
        frame_name=frame_name,
    )
    normalized = canonicalize_identifier_column(
        normalized,
        "HADM_ID",
        frame_name=frame_name,
    )
    normalized["INTIME"] = pd.to_datetime(normalized["INTIME"], errors="raise")
    normalized["OUTTIME"] = pd.to_datetime(normalized["OUTTIME"], errors="raise")
    if normalized[["INTIME", "OUTTIME"]].isna().any().any():
        raise ValueError(f"{frame_name} contains missing ICU timestamps.")

    normalized = collapse_identical_rows_or_raise(
        normalized,
        key_columns=["ICUSTAY_ID"],
        value_columns=[column for column in normalized.columns if column != "ICUSTAY_ID"],
        frame_name=frame_name,
    )
    if normalized.empty:
        raise ValueError(f"{frame_name} must not be empty.")
    return normalized.reset_index(drop=True)


def normalize_admissions(
    admissions: pd.DataFrame,
    *,
    frame_name: str = "admissions",
) -> pd.DataFrame:
    required_columns = ["HADM_ID", "HOSPITAL_EXPIRE_FLAG"]
    missing_columns = [column for column in required_columns if column not in admissions.columns]
    if missing_columns:
        raise KeyError(f"{frame_name} is missing required columns: {missing_columns}")

    normalized = canonicalize_identifier_column(
        admissions,
        "HADM_ID",
        frame_name=frame_name,
    )
    normalized["HOSPITAL_EXPIRE_FLAG"] = canonicalize_binary_mortality_series(
        normalized["HOSPITAL_EXPIRE_FLAG"],
        field_name=f"{frame_name}.HOSPITAL_EXPIRE_FLAG",
    )
    if "DEATHTIME" in normalized.columns:
        normalized["DEATHTIME"] = pd.to_datetime(normalized["DEATHTIME"], errors="raise")
    normalized = collapse_identical_rows_or_raise(
        normalized,
        key_columns=["HADM_ID"],
        value_columns=[column for column in normalized.columns if column != "HADM_ID"],
        frame_name=frame_name,
    )
    if normalized.empty:
        raise ValueError(f"{frame_name} must not be empty.")
    return normalized.reset_index(drop=True)


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Preprocess raw MIMIC-III ICU files.")
    parser.add_argument("--dataset-config-csv", default=None)
    parser.add_argument("--raw-data-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--max-debug-chunks", type=int, default=None)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve dataset config values and exit without loading data.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def maybe_run_validate_config_only(args):
    if not args.validate_config_only:
        return False
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_config import maybe_run_validate_config_only as validate

    validate("src/preprocess_mimic_iii_large.py", fixed_dataset="mimic")
    return True


def iter_csv_chunks(path: Path, usecols: list[str], chunksize: int, max_debug_chunks: int | None = None):
    count = 0
    identifier_dtypes = {
        column: "string"
        for column in usecols
        if column in {"ICUSTAY_ID", "HADM_ID"}
    }
    chunks = pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=usecols,
        dtype=identifier_dtypes,
    )
    for chunk in tqdm(chunks, desc=f"Reading {path.name}", unit="chunk"):
        yield chunk
        count += 1
        if max_debug_chunks is not None and count >= max_debug_chunks:
            break


def write_pickle_shard(path: Path, frame: pd.DataFrame) -> None:
    ensure_directory(path.parent)
    with path.open("wb") as handle:
        pickle.dump(frame, handle)


def read_pickle_shard(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        return pickle.load(handle)


def stream_filtered_shards(
    raw_path: Path,
    usecols: list[str],
    tmp_dir: Path,
    stem: str,
    predicate,
    chunksize: int,
    max_debug_chunks: int | None = None,
) -> list[Path]:
    shard_dir = tmp_dir / stem
    ensure_directory(shard_dir)
    for existing in shard_dir.glob("*.pkl"):
        existing.unlink()

    shard_paths: list[Path] = []
    for index, chunk in enumerate(iter_csv_chunks(raw_path, usecols, chunksize, max_debug_chunks=max_debug_chunks), start=0):
        filtered = predicate(chunk)
        if filtered is None or filtered.empty:
            continue
        shard_path = shard_dir / f"{stem}_{index:05d}.pkl"
        write_pickle_shard(shard_path, filtered)
        shard_paths.append(shard_path)
    return shard_paths


def candidate_identifier_membership_mask(
    series: pd.Series,
    canonical_ids: set[str],
) -> pd.Series:
    """Match lossless IDs while ignoring irrelevant invalid/null raw rows."""
    membership: list[bool] = []
    for value in series:
        try:
            canonical_id = canonicalize_mimic_id_scalar(
                value,
                field_name=str(series.name or "raw candidate identifier"),
            )
        except (TypeError, ValueError):
            membership.append(False)
            continue
        membership.append(canonical_id in canonical_ids)
    return pd.Series(membership, index=series.index, dtype=bool)


def build_chartevents_shards(raw_data_path: str | Path, icu: pd.DataFrame, chunksize: int, tmp_dir: Path, max_debug_chunks: int | None = None) -> list[Path]:
    raw_data_path = normalize_raw_data_path(raw_data_path)
    icu_ids = set(
        canonicalize_mimic_id_series(
            icu["ICUSTAY_ID"],
            field_name="icu.ICUSTAY_ID",
        )
    )

    def predicate(chunk: pd.DataFrame) -> pd.DataFrame:
        candidate_mask = candidate_identifier_membership_mask(
            chunk["ICUSTAY_ID"],
            icu_ids,
        )
        filtered = chunk.loc[candidate_mask].copy()
        filtered = canonicalize_identifier_column(
            filtered,
            "ICUSTAY_ID",
            frame_name="chartevents",
        )
        filtered = filtered.loc[filtered["ICUSTAY_ID"].isin(icu_ids)]
        filtered = canonicalize_identifier_column(
            filtered,
            "HADM_ID",
            frame_name="chartevents",
        )
        filtered = filtered.loc[filtered["ERROR"] != 1]
        filtered = filtered.loc[filtered["CHARTTIME"].notna()]
        filtered = filtered.loc[~(filtered["VALUE"].isna() & filtered["VALUENUM"].isna())]
        filtered = filtered.drop(columns=["ERROR"], errors="ignore")
        filtered["TABLE"] = "chart"
        return filtered.copy()

    return stream_filtered_shards(
        raw_data_path / "CHARTEVENTS.csv",
        ["HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUENUM", "VALUEUOM", "ERROR"],
        tmp_dir,
        "chartevents",
        predicate,
        chunksize,
        max_debug_chunks=max_debug_chunks,
    )


def build_labevents_shards(raw_data_path: str | Path, icu: pd.DataFrame, chunksize: int, tmp_dir: Path, max_debug_chunks: int | None = None) -> list[Path]:
    raw_data_path = normalize_raw_data_path(raw_data_path)
    hadm_ids = set(
        canonicalize_mimic_id_series(
            icu["HADM_ID"],
            field_name="icu.HADM_ID",
        )
    )

    def predicate(chunk: pd.DataFrame) -> pd.DataFrame:
        candidate_mask = candidate_identifier_membership_mask(
            chunk["HADM_ID"],
            hadm_ids,
        )
        filtered = chunk.loc[candidate_mask].copy()
        filtered = canonicalize_identifier_column(
            filtered,
            "HADM_ID",
            frame_name="labevents",
        )
        filtered = filtered.loc[filtered["HADM_ID"].isin(hadm_ids)]
        filtered = filtered.loc[filtered["CHARTTIME"].notna()]
        filtered = filtered.loc[~(filtered["VALUE"].isna() & filtered["VALUENUM"].isna())]
        filtered["ICUSTAY_ID"] = pd.NA
        filtered["TABLE"] = "lab"
        return filtered.copy()

    return stream_filtered_shards(
        raw_data_path / "LABEVENTS.csv",
        ["HADM_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUENUM", "VALUEUOM"],
        tmp_dir,
        "labevents",
        predicate,
        chunksize,
        max_debug_chunks=max_debug_chunks,
    )


def build_outputevents_shards(raw_data_path: str | Path, icu: pd.DataFrame, chunksize: int, tmp_dir: Path, max_debug_chunks: int | None = None) -> list[Path]:
    raw_data_path = normalize_raw_data_path(raw_data_path)
    icu_ids = set(
        canonicalize_mimic_id_series(
            icu["ICUSTAY_ID"],
            field_name="icu.ICUSTAY_ID",
        )
    )

    def predicate(chunk: pd.DataFrame) -> pd.DataFrame:
        candidate_mask = candidate_identifier_membership_mask(
            chunk["ICUSTAY_ID"],
            icu_ids,
        )
        filtered = chunk.loc[candidate_mask & chunk["VALUE"].notna()].copy()
        filtered = canonicalize_identifier_column(
            filtered,
            "ICUSTAY_ID",
            frame_name="outputevents",
        )
        filtered = filtered.loc[filtered["ICUSTAY_ID"].isin(icu_ids)]
        filtered["VALUENUM"] = filtered["VALUE"]
        filtered["VALUE"] = None
        filtered["TABLE"] = "output"
        return filtered.copy()

    return stream_filtered_shards(
        raw_data_path / "OUTPUTEVENTS.csv",
        ["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUEUOM"],
        tmp_dir,
        "outputevents",
        predicate,
        chunksize,
        max_debug_chunks=max_debug_chunks,
    )


def build_inputevents_shards(raw_data_path: str | Path, icu: pd.DataFrame, chunksize: int, tmp_dir: Path, max_debug_chunks: int | None = None) -> tuple[list[Path], list[Path]]:
    raw_data_path = normalize_raw_data_path(raw_data_path)
    icu_ids = set(
        canonicalize_mimic_id_series(
            icu["ICUSTAY_ID"],
            field_name="icu.ICUSTAY_ID",
        )
    )

    def cv_predicate(chunk: pd.DataFrame) -> pd.DataFrame:
        candidate_mask = candidate_identifier_membership_mask(
            chunk["ICUSTAY_ID"],
            icu_ids,
        )
        filtered = chunk.loc[candidate_mask & chunk["AMOUNT"].notna()].copy()
        filtered = canonicalize_identifier_column(
            filtered,
            "ICUSTAY_ID",
            frame_name="inputevents_cv",
        )
        filtered = filtered.loc[filtered["ICUSTAY_ID"].isin(icu_ids)]
        filtered["TABLE"] = "input_cv"
        filtered["CHARTTIME"] = pd.to_datetime(filtered["CHARTTIME"])
        filtered["VALUENUM"] = filtered["AMOUNT"]
        filtered["VALUEUOM"] = filtered["AMOUNTUOM"]
        filtered["VALUE"] = None
        return filtered.copy()

    cv_paths = stream_filtered_shards(
        raw_data_path / "INPUTEVENTS_CV.csv",
        ["ICUSTAY_ID", "ITEMID", "CHARTTIME", "AMOUNT", "AMOUNTUOM"],
        tmp_dir,
        "inputevents_cv",
        cv_predicate,
        chunksize,
        max_debug_chunks=max_debug_chunks,
    )

    def mv_predicate(chunk: pd.DataFrame) -> pd.DataFrame:
        candidate_mask = candidate_identifier_membership_mask(
            chunk["ICUSTAY_ID"],
            icu_ids,
        )
        filtered = chunk.loc[candidate_mask].copy()
        filtered = canonicalize_identifier_column(
            filtered,
            "ICUSTAY_ID",
            frame_name="inputevents_mv",
        )
        filtered = filtered.loc[filtered["ICUSTAY_ID"].isin(icu_ids)]
        filtered["TABLE"] = "input_mv"
        filtered["CHARTTIME"] = pd.to_datetime(filtered["ENDTIME"])
        filtered["VALUENUM"] = filtered["AMOUNT"]
        filtered["VALUEUOM"] = filtered["AMOUNTUOM"]
        filtered["VALUE"] = None
        return filtered.copy()

    mv_paths = stream_filtered_shards(
        raw_data_path / "INPUTEVENTS_MV.csv",
        ["ICUSTAY_ID", "ITEMID", "STARTTIME", "ENDTIME", "AMOUNT", "AMOUNTUOM"],
        tmp_dir,
        "inputevents_mv",
        mv_predicate,
        chunksize,
        max_debug_chunks=max_debug_chunks,
    )
    return cv_paths, mv_paths


def add_feature_rows(frame: pd.DataFrame, fragments: list[pd.DataFrame]) -> None:
    frame["CHARTTIME"] = pd.to_datetime(frame["CHARTTIME"])

    def append_subset(subset: pd.DataFrame, name: str, lower: float | None = None, upper: float | None = None) -> None:
        if subset.empty:
            return
        subset = subset.copy()
        subset["NAME"] = name
        subset["VALUEUOM"] = None
        subset["VALUE"] = None
        if lower is not None:
            subset = subset.loc[(subset["VALUENUM"] >= lower)]
        if upper is not None:
            subset = subset.loc[(subset["VALUENUM"] <= upper)]
        if subset.empty:
            return
        fragments.append(subset[["HADM_ID", "ICUSTAY_ID", "CHARTTIME", "VALUENUM", "TABLE", "NAME"]])

    # chart-derived features
    bp_item_ids = [8368, 220051, 225310, 8555, 8441, 220180, 8502, 8440, 8503, 8504, 8507, 8506, 224643, 227242, 51, 220050, 225309, 6701, 455, 220179, 3313, 3315, 442, 3317, 3323, 3321, 224167, 227243, 52, 220052, 225312, 224, 6702, 224322, 456, 220181, 3312, 3314, 3316, 3322, 3320, 443]
    bp = frame.loc[frame["ITEMID"].isin(bp_item_ids)]
    bp = bp.loc[(bp["VALUENUM"] >= 0) & (bp["VALUENUM"] <= 375)]
    append_subset(bp, "BP")

    chart = frame.loc[frame["TABLE"] == "chart"]
    gcs_components = {
        "GCS_eye": [184, 220739],
        "GCS_motor": [454, 223901],
        "GCS_verbal": [723, 223900],
    }
    for name, item_ids in gcs_components.items():
        append_subset(chart.loc[chart["ITEMID"].isin(item_ids)], name)

    hr = frame.loc[frame["ITEMID"].isin([211, 220045])]
    append_subset(hr, "HR", 0, 390)

    rr = frame.loc[frame["ITEMID"].isin([618, 220210, 3603, 224689, 614, 651, 224422, 615, 224690, 619, 224688, 227860, 227918])]
    append_subset(rr, "RR", 0, 330)

    temp = frame.loc[frame["ITEMID"].isin([3655, 677, 676, 223762, 223761, 678, 679, 3654])]
    if not temp.empty:
        temp = temp.copy()
        temp.loc[temp["ITEMID"].isin([223761, 678, 679, 3654]), "VALUENUM"] = (temp.loc[temp["ITEMID"].isin([223761, 678, 679, 3654]), "VALUENUM"] - 32) * 5 / 9
        append_subset(temp, "Temperature", 14.2, 47)

    weight = frame.loc[frame["ITEMID"].isin([224639, 226512, 226846, 763, 226531])]
    if not weight.empty:
        weight = weight.copy()
        weight.loc[weight["ITEMID"].isin([226531]), "VALUENUM"] = weight.loc[weight["ITEMID"].isin([226531]), "VALUENUM"] * 0.453592
        append_subset(weight, "Weight", 0, 300)

    height = frame.loc[frame["ITEMID"].isin([1394, 226707, 226730])]
    if not height.empty:
        height = height.copy()
        height.loc[height["ITEMID"].isin([1394, 226707]), "VALUENUM"] = height.loc[height["ITEMID"].isin([1394, 226707]), "VALUENUM"] * 2.54
        append_subset(height, "Height", 0, 275)

    fio2 = frame.loc[frame["ITEMID"].isin([3420, 223835, 3422, 189, 727, 190])]
    if not fio2.empty:
        fio2 = fio2.copy()
        fio2.loc[fio2["VALUENUM"] > 1.0, "VALUENUM"] = fio2.loc[fio2["VALUENUM"] > 1.0, "VALUENUM"] / 100
        append_subset(fio2, "FiO2", 0.2, 1)


def validate_icustay_hadm_mapping(icu: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ICUSTAY_ID", "HADM_ID"}
    missing_columns = sorted(required_columns - set(icu.columns))
    if missing_columns:
        raise KeyError(f"ICU mapping is missing required columns: {missing_columns}")

    mapping = canonicalize_identifier_column(
        icu[["ICUSTAY_ID", "HADM_ID"]],
        "ICUSTAY_ID",
        frame_name="ICU mapping",
    )
    mapping = canonicalize_identifier_column(
        mapping,
        "HADM_ID",
        frame_name="ICU mapping",
    )
    try:
        mapping = collapse_identical_rows_or_raise(
            mapping,
            key_columns=["ICUSTAY_ID"],
            value_columns=["HADM_ID"],
            frame_name="ICU mapping",
        )
    except ValueError as error:
        raise ValueError(
            "ICU mapping contains conflicting duplicate ICUSTAY_ID -> HADM_ID values."
        ) from error
    return mapping.reset_index(drop=True)


def link_fragment_hadm_ids(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    mapping = validate_icustay_hadm_mapping(mapping)
    linked_input = canonicalize_identifier_column(
        frame,
        "ICUSTAY_ID",
        frame_name=source,
    )
    source_has_hadm = "HADM_ID" in linked_input.columns
    if source_has_hadm:
        linked_input = canonicalize_identifier_column(
            linked_input,
            "HADM_ID",
            frame_name=source,
        )

    linked = linked_input.merge(
        mapping.rename(columns={"HADM_ID": "_MAPPED_HADM_ID"}),
        on="ICUSTAY_ID",
        how="left",
        validate="many_to_one",
    )
    if len(linked) != len(linked_input):
        raise AssertionError(f"{source} mapping changed the event-fragment row count.")

    unmapped = linked["_MAPPED_HADM_ID"].isna()
    if unmapped.any():
        raise ValueError(
            f"{source} contains {int(unmapped.sum())} row(s) with unmapped ICUSTAY_ID values."
        )
    if source_has_hadm:
        conflicts = linked["HADM_ID"] != linked["_MAPPED_HADM_ID"]
        if conflicts.any():
            raise ValueError(
                f"{source} contains {int(conflicts.sum())} row(s) whose HADM_ID "
                "conflicts with the canonical ICU mapping."
            )
        return linked.drop(columns=["_MAPPED_HADM_ID"])

    return linked.rename(columns={"_MAPPED_HADM_ID": "HADM_ID"})


def collect_event_fragments(chartevents_paths: list[Path], labevents_paths: list[Path], outputevents_paths: list[Path], inputevents_paths: list[Path], icu: pd.DataFrame) -> pd.DataFrame:
    fragments: list[pd.DataFrame] = []
    icustay_hadm_mapping = validate_icustay_hadm_mapping(icu)
    for path in chartevents_paths:
        frame = read_pickle_shard(path)
        if frame.empty:
            continue
        frame = link_fragment_hadm_ids(frame, icustay_hadm_mapping, "chartevents")
        add_feature_rows(frame, fragments)

    for path in labevents_paths:
        frame = read_pickle_shard(path)
        if frame.empty:
            continue
        frame = canonicalize_identifier_column(
            frame,
            "HADM_ID",
            frame_name="labevents",
        )
        frame = canonicalize_identifier_column(
            frame,
            "ICUSTAY_ID",
            frame_name="labevents",
            allow_missing=True,
        )
        add_feature_rows(frame, fragments)

    for path in outputevents_paths:
        frame = read_pickle_shard(path)
        if frame.empty:
            continue
        frame = link_fragment_hadm_ids(frame, icustay_hadm_mapping, "outputevents")
        frame["CHARTTIME"] = pd.to_datetime(frame["CHARTTIME"])
        frame = frame.loc[(frame["VALUENUM"] >= 0)]
        if frame.empty:
            continue
        frame["NAME"] = "Output"
        fragments.append(frame[["HADM_ID", "ICUSTAY_ID", "CHARTTIME", "VALUENUM", "TABLE", "NAME"]])

    for path in inputevents_paths:
        frame = read_pickle_shard(path)
        if frame.empty:
            continue
        frame = link_fragment_hadm_ids(frame, icustay_hadm_mapping, "inputevents")
        frame["CHARTTIME"] = pd.to_datetime(frame["CHARTTIME"])
        frame = frame.loc[(frame["VALUENUM"] >= 0)]
        if frame.empty:
            continue
        frame["NAME"] = "Input"
        fragments.append(frame[["HADM_ID", "ICUSTAY_ID", "CHARTTIME", "VALUENUM", "TABLE", "NAME"]])

    if not fragments:
        return pd.DataFrame(columns=["HADM_ID", "ICUSTAY_ID", "CHARTTIME", "VALUENUM", "TABLE", "NAME"])
    return pd.concat(fragments, ignore_index=True)


def assign_missing_icustays(events: pd.DataFrame, icu: pd.DataFrame) -> pd.DataFrame:
    icu = normalize_icu_cohort(icu, frame_name="icu")
    interval_map: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]] = {}
    for row in icu[["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]].itertuples(index=False):
        interval_map.setdefault(row.HADM_ID, []).append(
            (row.ICUSTAY_ID, row.INTIME, row.OUTTIME)
        )

    events = canonicalize_identifier_column(
        events,
        "HADM_ID",
        frame_name="events",
    )
    events = canonicalize_identifier_column(
        events,
        "ICUSTAY_ID",
        frame_name="events",
        allow_missing=True,
    )
    events["CHARTTIME"] = pd.to_datetime(events["CHARTTIME"], errors="raise")
    if events["CHARTTIME"].isna().any():
        raise ValueError("events contains missing CHARTTIME values.")

    missing_idx = events["ICUSTAY_ID"].isna()
    ambiguous_rows = 0
    if missing_idx.any():
        assigned: list[object] = []
        for row in events.loc[missing_idx].itertuples(index=False):
            matching_ids = {
                icustay_id
                for icustay_id, intime, outtime in interval_map.get(row.HADM_ID, [])
                if intime <= row.CHARTTIME <= outtime
            }
            if len(matching_ids) > 1:
                ambiguous_rows += 1
                assigned.append(pd.NA)
            elif matching_ids:
                assigned.append(next(iter(matching_ids)))
            else:
                assigned.append(pd.NA)
        events.loc[missing_idx, "ICUSTAY_ID"] = assigned
    if ambiguous_rows:
        raise ValueError(
            f"events contains {ambiguous_rows} row(s) matching multiple ICU stays."
        )

    resolved = events.loc[events["ICUSTAY_ID"].notna()].copy()
    if resolved.empty:
        return resolved

    mapping = validate_icustay_hadm_mapping(icu)
    checked = resolved.merge(
        mapping.rename(columns={"HADM_ID": "_MAPPED_HADM_ID"}),
        on="ICUSTAY_ID",
        how="left",
        validate="many_to_one",
    )
    if len(checked) != len(resolved):
        raise AssertionError("ICU assignment validation changed the event row count.")
    unmapped = checked["_MAPPED_HADM_ID"].isna()
    if unmapped.any():
        raise ValueError(
            f"events contains {int(unmapped.sum())} row(s) with unmapped ICUSTAY_ID values."
        )
    conflicts = checked["HADM_ID"] != checked["_MAPPED_HADM_ID"]
    if conflicts.any():
        raise ValueError(
            f"events contains {int(conflicts.sum())} row(s) whose HADM_ID "
            "conflicts with the canonical ICU mapping."
        )
    return checked.drop(columns=["_MAPPED_HADM_ID"])


def build_canonical_payload(
    events: pd.DataFrame,
    final_icu: pd.DataFrame,
    admissions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if events.empty:
        raise ValueError("events must not be empty when building the canonical MIMIC cohort.")
    final_icu = normalize_icu_cohort(final_icu, frame_name="final_icu")
    admissions = normalize_admissions(admissions, frame_name="admissions")

    events = canonicalize_identifier_column(
        events,
        "ICUSTAY_ID",
        frame_name="events",
    )
    if "HADM_ID" in events.columns:
        events = link_fragment_hadm_ids(
            events,
            validate_icustay_hadm_mapping(final_icu),
            "events",
        )
    events = events.rename(
        columns={
            "rel_charttime": "minute",
            "NAME": "variable",
            "VALUENUM": "value",
            "ICUSTAY_ID": "ts_id",
        }
    ).copy()
    final_icu = final_icu.rename(columns={"ICUSTAY_ID": "ts_id"}).copy()
    final_icu["ts_id"] = canonicalize_mimic_id_series(
        final_icu["ts_id"],
        field_name="final_icu.ts_id",
    )

    data_age = final_icu[["ts_id", "AGE"]].copy()
    data_age["variable"] = "Age"
    data_age = data_age.rename(columns={"AGE": "value"})
    data_gen = final_icu[["ts_id", "GENDER"]].copy()
    data_gen.loc[data_gen["GENDER"] == "M", "GENDER"] = 0
    data_gen.loc[data_gen["GENDER"] == "F", "GENDER"] = 1
    data_gen["variable"] = "Gender"
    data_gen = data_gen.rename(columns={"GENDER": "value"})
    static_data = pd.concat([data_age, data_gen], ignore_index=True)
    static_data["minute"] = 0
    static_data = static_data[["ts_id", "minute", "variable", "value"]]

    events = pd.concat(
        [static_data, events[["ts_id", "minute", "variable", "value"]]],
        ignore_index=True,
    ).drop_duplicates()
    ts = build_canonical_ts(events)
    ts_ids = build_ts_ids(ts)
    oc = build_canonical_oc(final_icu, admissions, valid_ts_ids=ts_ids)
    return ts, oc, ts_ids


def main(argv: Iterable[str] | None = None):
    global RAW_DATA_PATH, OUTPUT_PATH
    args = parse_args(argv)
    if maybe_run_validate_config_only(args):
        return

    RAW_DATA_PATH = normalize_raw_data_path(args.raw_data_path or RAW_DATA_PATH)
    OUTPUT_PATH = args.output_path or OUTPUT_PATH
    print("=== Starting MIMIC-III preprocessing ===")
    print(f"Raw data root: {os.path.abspath(RAW_DATA_PATH)}")
    print(f"Output artifact: {os.path.abspath(OUTPUT_PATH)}")
    print(f"Chunksize: {args.chunksize}")
    print(f"Temporary directory: {args.tmp_dir or 'auto'}")
    print(f"Keep intermediates: {bool(args.keep_intermediates)}")
    if args.max_debug_chunks is not None:
        print(f"Max debug chunks: {args.max_debug_chunks}")

    tmp_dir = Path(args.tmp_dir or os.path.join(os.path.dirname(OUTPUT_PATH) or ".", "preprocess_mimic_tmp"))
    if tmp_dir.exists():
        maybe_remove(tmp_dir)
    ensure_directory(tmp_dir)
    log_memory("before load")

    log_stage(1, "Loading ICU stays and patient demographics")
    icu = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "ICUSTAYS.csv"),
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"],
        dtype={"HADM_ID": "string", "ICUSTAY_ID": "string"},
    )
    icu = normalize_icu_cohort(icu, frame_name="ICUSTAYS")

    pat = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "PATIENTS.csv"),
        usecols=["SUBJECT_ID", "DOB", "DOD", "GENDER"],
    )
    pat = collapse_identical_rows_or_raise(
        pat,
        key_columns=["SUBJECT_ID"],
        value_columns=["DOB", "DOD", "GENDER"],
        frame_name="PATIENTS",
    )
    icu_row_count = len(icu)
    icu = icu.merge(pat, on="SUBJECT_ID", how="left", validate="many_to_one")
    if len(icu) != icu_row_count:
        raise AssertionError("The patient-demographics join changed the ICU cohort row count.")
    icu["DOB"] = pd.to_datetime(icu["DOB"], errors="raise")
    icu["AGE"] = icu["INTIME"].map(lambda x: x.year) - icu["DOB"].map(lambda x: x.year)
    icu = icu.loc[icu["AGE"] >= 18].copy()
    if icu.empty:
        raise ValueError("No adult ICU stays remain after demographic filtering.")
    print(f"      Adult ICU stays retained: {len(icu):,}")
    log_memory("after ICU/patient load")

    log_stage(2, "Reading CHARTEVENTS shards")
    chartevents_paths = build_chartevents_shards(RAW_DATA_PATH, icu, args.chunksize, tmp_dir, args.max_debug_chunks)
    log_stage(3, "Reading LABEVENTS shards")
    labevents_paths = build_labevents_shards(RAW_DATA_PATH, icu, args.chunksize, tmp_dir, args.max_debug_chunks)
    log_stage(4, "Reading OUTPUTEVENTS shards")
    outputevents_paths = build_outputevents_shards(RAW_DATA_PATH, icu, args.chunksize, tmp_dir, args.max_debug_chunks)
    log_stage(5, "Reading INPUTEVENTS shards")
    inputevents_cv_paths, inputevents_mv_paths = build_inputevents_shards(RAW_DATA_PATH, icu, args.chunksize, tmp_dir, args.max_debug_chunks)
    log_memory("after shard filtering")

    log_stage(6, "Extracting event fragments")
    events = collect_event_fragments(chartevents_paths, labevents_paths, outputevents_paths, inputevents_cv_paths + inputevents_mv_paths, icu)
    print(f"      Rows accumulated after shard extraction: {len(events):,}")

    log_stage(7, "Aligning events to ICU stays")
    events = assign_missing_icustays(events, icu)
    if events.empty:
        raise ValueError("No events could be assigned to an ICU stay.")

    event_row_count = len(events)
    events = events.merge(
        icu[["HADM_ID", "ICUSTAY_ID", "INTIME"]],
        on=["HADM_ID", "ICUSTAY_ID"],
        how="left",
        validate="many_to_one",
    )
    if len(events) != event_row_count:
        raise AssertionError("The ICU timeline join changed the event row count.")
    events["rel_charttime"] = (
        events["CHARTTIME"] - events["INTIME"]
    ).dt.total_seconds() // 60
    if events["rel_charttime"].isna().any():
        raise ValueError("Some events could not be aligned to an ICU admission time.")
    events = events.drop(columns=["INTIME"])

    icu = icu.loc[
        (icu["OUTTIME"] - icu["INTIME"]) >= pd.Timedelta(24, "h")
    ].copy()
    admissions = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "ADMISSIONS.csv"),
        usecols=["HADM_ID", "DEATHTIME", "HOSPITAL_EXPIRE_FLAG"],
        dtype={"HADM_ID": "string"},
    )
    admissions = normalize_admissions(admissions, frame_name="ADMISSIONS")
    icu_row_count = len(icu)
    icu = icu.merge(
        admissions[["HADM_ID", "DEATHTIME"]],
        on="HADM_ID",
        how="left",
        validate="many_to_one",
    )
    if len(icu) != icu_row_count:
        raise AssertionError("The admissions join changed the ICU cohort row count.")
    icu = icu.loc[
        ((icu["DEATHTIME"] - icu["INTIME"]) >= pd.Timedelta(24, "h"))
        | icu["DEATHTIME"].isna()
    ].copy()
    events = events.loc[events["ICUSTAY_ID"].isin(set(icu["ICUSTAY_ID"]))]
    events = events.loc[events["rel_charttime"] < 24 * 60].copy()
    if events.empty:
        raise ValueError("No events remain after the canonical 24-hour cohort filters.")
    final_icu = icu.loc[icu["ICUSTAY_ID"].isin(set(events["ICUSTAY_ID"]))].copy()
    if final_icu.empty:
        raise ValueError("The canonical MIMIC ICU cohort must not be empty.")

    print(f"      Events retained after timeline alignment: {len(events):,}")
    print(f"      ICU stays retained after cohort filters: {len(final_icu):,}")

    log_stage(8, "Building canonical ts/oc payload")
    ts, oc, ts_ids = build_canonical_payload(events, final_icu, admissions)

    log_stage(9, "Validating and saving the processed MIMIC artifact")
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    temp_output = Path(OUTPUT_PATH).with_suffix(".tmp.pkl")
    serialize_processed_output(ts, oc, ts_ids, str(temp_output))
    if temp_output.exists():
        shutil.move(str(temp_output), OUTPUT_PATH)

    print(f"Saved processed MIMIC artifact to: {os.path.abspath(OUTPUT_PATH)} | ts rows={len(ts):,} | oc rows={len(oc):,} | stays={len(ts_ids):,}")

    if not args.keep_intermediates:
        maybe_remove(tmp_dir)


if __name__ == "__main__":
    main()
