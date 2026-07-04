from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess_mimic_iii_large_contract import (
    build_canonical_oc,
    build_canonical_ts,
    build_ts_ids,
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
    for chunk in tqdm(pd.read_csv(path, chunksize=chunksize, usecols=usecols), desc=f"Reading {path.name}", unit="chunk"):
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


def build_chartevents_shards(raw_data_path: Path, icu: pd.DataFrame, chunksize: int, tmp_dir: Path, max_debug_chunks: int | None = None) -> list[Path]:
    icu_ids = set(icu["ICUSTAY_ID"].dropna().astype(int).tolist())

    def predicate(chunk: pd.DataFrame) -> pd.DataFrame:
        filtered = chunk.loc[chunk["ICUSTAY_ID"].isin(icu_ids)]
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


def build_labevents_shards(raw_data_path: Path, icu: pd.DataFrame, chunksize: int, tmp_dir: Path, max_debug_chunks: int | None = None) -> list[Path]:
    hadm_ids = set(icu["HADM_ID"].dropna().astype(int).tolist())

    def predicate(chunk: pd.DataFrame) -> pd.DataFrame:
        filtered = chunk.loc[chunk["HADM_ID"].isin(hadm_ids)]
        filtered = filtered.loc[filtered["CHARTTIME"].notna()]
        filtered = filtered.loc[~(filtered["VALUE"].isna() & filtered["VALUENUM"].isna())]
        filtered["ICUSTAY_ID"] = np.nan
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


def build_outputevents_shards(raw_data_path: Path, icu: pd.DataFrame, chunksize: int, tmp_dir: Path, max_debug_chunks: int | None = None) -> list[Path]:
    icu_ids = set(icu["ICUSTAY_ID"].dropna().astype(int).tolist())

    def predicate(chunk: pd.DataFrame) -> pd.DataFrame:
        filtered = chunk.loc[chunk["VALUE"].notna()]
        filtered = filtered.loc[filtered["ICUSTAY_ID"].isin(icu_ids)]
        filtered["VALUENUM"] = filtered["VALUE"]
        filtered["VALUE"] = None
        filtered["TABLE"] = "output"
        filtered["ICUSTAY_ID"] = filtered["ICUSTAY_ID"].astype(int)
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


def build_inputevents_shards(raw_data_path: Path, icu: pd.DataFrame, chunksize: int, tmp_dir: Path, max_debug_chunks: int | None = None) -> tuple[list[Path], list[Path]]:
    icu_ids = set(icu["ICUSTAY_ID"].dropna().astype(int).tolist())

    def cv_predicate(chunk: pd.DataFrame) -> pd.DataFrame:
        filtered = chunk.loc[chunk["AMOUNT"].notna()]
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
        filtered = chunk.loc[chunk["ICUSTAY_ID"].isin(icu_ids)]
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


def collect_event_fragments(chartevents_paths: list[Path], labevents_paths: list[Path], outputevents_paths: list[Path], inputevents_paths: list[Path]) -> pd.DataFrame:
    fragments: list[pd.DataFrame] = []
    for path in chartevents_paths:
        frame = read_pickle_shard(path)
        if frame.empty:
            continue
        add_feature_rows(frame, fragments)

    for path in labevents_paths:
        frame = read_pickle_shard(path)
        if frame.empty:
            continue
        add_feature_rows(frame, fragments)

    for path in outputevents_paths:
        frame = read_pickle_shard(path)
        if frame.empty:
            continue
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
    icu = icu.copy()
    icu["INTIME"] = pd.to_datetime(icu["INTIME"])
    icu["OUTTIME"] = pd.to_datetime(icu["OUTTIME"])
    interval_map: dict[int, list[tuple[int, pd.Timestamp, pd.Timestamp]]] = {}
    for row in icu[["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]].itertuples(index=False):
        interval_map.setdefault(int(row.HADM_ID), []).append((int(row.ICUSTAY_ID), row.INTIME, row.OUTTIME))

    events = events.copy()
    events["CHARTTIME"] = pd.to_datetime(events["CHARTTIME"])
    missing_idx = events["ICUSTAY_ID"].isna()
    if missing_idx.any():
        assigned: list[Any] = []
        for row in events.loc[missing_idx].itertuples(index=False):
            matched = np.nan
            for icustay_id, intime, outtime in interval_map.get(int(row.HADM_ID), []):
                if intime <= row.CHARTTIME <= outtime:
                    matched = icustay_id
                    break
            assigned.append(matched)
        events.loc[missing_idx, "ICUSTAY_ID"] = assigned
    return events.loc[events["ICUSTAY_ID"].notna()].copy()


def main(argv: Iterable[str] | None = None):
    global RAW_DATA_PATH, OUTPUT_PATH
    args = parse_args(argv)
    if maybe_run_validate_config_only(args):
        return

    RAW_DATA_PATH = args.raw_data_path or RAW_DATA_PATH
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
    icu = pd.read_csv(os.path.join(RAW_DATA_PATH, "ICUSTAYS.csv"), usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"])
    icu = icu.loc[icu["INTIME"].notna()]
    icu = icu.loc[icu["OUTTIME"].notna()]
    pat = pd.read_csv(os.path.join(RAW_DATA_PATH, "PATIENTS.csv"), usecols=["SUBJECT_ID", "DOB", "DOD", "GENDER"])
    icu = icu.merge(pat, on="SUBJECT_ID", how="left")
    icu["INTIME"] = pd.to_datetime(icu["INTIME"])
    icu["DOB"] = pd.to_datetime(icu["DOB"])
    icu["AGE"] = icu["INTIME"].map(lambda x: x.year) - icu["DOB"].map(lambda x: x.year)
    icu = icu.loc[icu["AGE"] >= 18]
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
    events = collect_event_fragments(chartevents_paths, labevents_paths, outputevents_paths, inputevents_cv_paths + inputevents_mv_paths)
    print(f"      Rows accumulated after shard extraction: {len(events):,}")

    log_stage(7, "Aligning events to ICU stays")
    events = assign_missing_icustays(events, icu)
    if events.empty:
        events = pd.DataFrame(columns=["HADM_ID", "ICUSTAY_ID", "CHARTTIME", "VALUENUM", "TABLE", "NAME"])
    else:
        events = events.merge(icu[["HADM_ID", "ICUSTAY_ID", "INTIME"]], on=["HADM_ID", "ICUSTAY_ID"], how="left")
        events["rel_charttime"] = (events["CHARTTIME"] - events["INTIME"]).dt.total_seconds() // 60
        events = events.loc[events["rel_charttime"].notna()]
        events = events.drop(columns=["INTIME"])

    icu = icu.loc[(icu["OUTTIME"] - icu["INTIME"]) >= pd.Timedelta(24, "h")]
    admissions = pd.read_csv(os.path.join(RAW_DATA_PATH, "ADMISSIONS.csv"), usecols=["HADM_ID", "DEATHTIME"])
    icu = icu.merge(admissions, on="HADM_ID", how="left")
    icu["DEATHTIME"] = pd.to_datetime(icu["DEATHTIME"])
    icu = icu.loc[((icu["DEATHTIME"] - icu["INTIME"]) >= pd.Timedelta(24, "h")) | icu["DEATHTIME"].isna()]
    if not events.empty:
        events = events.loc[events["ICUSTAY_ID"].isin(icu["ICUSTAY_ID"])]
        events = events.loc[events["rel_charttime"] < 24 * 60]
    final_icu = icu.loc[icu["ICUSTAY_ID"].isin(events["ICUSTAY_ID"]) if not events.empty else icu.index]

    print(f"      Events retained after timeline alignment: {len(events):,}")
    print(f"      ICU stays retained after cohort filters: {len(final_icu):,}")

    log_stage(8, "Building canonical ts/oc payload")
    events = events.rename(columns={"rel_charttime": "minute", "NAME": "variable", "VALUENUM": "value", "ICUSTAY_ID": "ts_id"})
    final_icu = final_icu.rename(columns={"ICUSTAY_ID": "ts_id"})
    final_icu["ts_id"] = final_icu["ts_id"].astype(str)
    data_age = final_icu[["ts_id", "AGE"]].copy()
    data_age["variable"] = "Age"
    data_age = data_age.rename(columns={"AGE": "value"})
    data_gen = final_icu[["ts_id", "GENDER"]].copy()
    data_gen.loc[data_gen["GENDER"] == "M", "GENDER"] = 0
    data_gen.loc[data_gen["GENDER"] == "F", "GENDER"] = 1
    data_gen["variable"] = "Gender"
    data_gen = data_gen.rename(columns={"GENDER": "value"})
    data = pd.concat([data_age, data_gen], ignore_index=True)
    data["minute"] = 0
    data = data[["ts_id", "minute", "variable", "value"]]
    events = pd.concat([data, events[["ts_id", "minute", "variable", "value"]]], ignore_index=True)
    events = events.drop_duplicates()

    ts = build_canonical_ts(events)
    ts_ids = build_ts_ids(ts)
    oc = build_canonical_oc(final_icu.rename(columns={"ts_id": "ICUSTAY_ID"}), admissions, valid_ts_ids=ts_ids)

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
