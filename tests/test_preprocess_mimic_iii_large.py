from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import preprocess_mimic_iii_large as preprocess  # noqa: E402
from preprocess_mimic_iii_large_contract import (  # noqa: E402
    CANONICAL_OC_COLUMNS,
    CANONICAL_TS_COLUMNS,
)
from tagging_latent_variables_mimiciii import PICKLE_GCS_COMPONENTS  # noqa: E402


def _event_shard(icustay_id: int, table: str, value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ICUSTAY_ID": [icustay_id],
            "ITEMID": [1],
            "CHARTTIME": ["2020-01-01 01:00:00"],
            "VALUENUM": [value],
            "VALUEUOM": ["mL"],
            "TABLE": [table],
        }
    )


def _write_shard(path: Path, frame: pd.DataFrame) -> Path:
    preprocess.write_pickle_shard(path, frame)
    return path


def test_shard_builder_normalizes_relative_string_raw_path(monkeypatch, tmp_path: Path) -> None:
    caller = tmp_path / "caller"
    raw_root = tmp_path / "raw"
    caller.mkdir()
    raw_root.mkdir()
    monkeypatch.chdir(caller)

    captured_paths: list[Path] = []

    def fake_stream_filtered_shards(
        raw_path,
        usecols,
        tmp_dir,
        stem,
        predicate,
        chunksize,
        max_debug_chunks=None,
    ):
        captured_paths.append(raw_path)
        return []

    monkeypatch.setattr(preprocess, "stream_filtered_shards", fake_stream_filtered_shards)
    icu = pd.DataFrame({"ICUSTAY_ID": pd.Series(dtype="int64")})

    result = preprocess.build_outputevents_shards("../raw", icu, 10, tmp_path / "tmp")

    assert result == []
    assert captured_paths == [(raw_root / "OUTPUTEVENTS.csv").resolve()]
    assert isinstance(captured_paths[0], Path)
    assert captured_paths[0].is_absolute()


def test_collect_event_fragments_links_output_and_input_hadm_ids(tmp_path: Path) -> None:
    output_path = _write_shard(tmp_path / "output.pkl", _event_shard(10, "output", 2.0))
    input_path = _write_shard(tmp_path / "input.pkl", _event_shard(20, "input_cv", 3.0))
    icu_mapping = pd.DataFrame(
        {
            "ICUSTAY_ID": [10, 20],
            "HADM_ID": [100, 200],
        }
    )

    events = preprocess.collect_event_fragments(
        [],
        [],
        [output_path],
        [input_path],
        icu_mapping,
    )

    assert len(events) == 2
    assert list(events[["ICUSTAY_ID", "HADM_ID", "NAME"]].itertuples(index=False, name=None)) == [
        ("10", "100", "Output"),
        ("20", "200", "Input"),
    ]


def test_collect_event_fragments_rejects_unmapped_icustay(tmp_path: Path) -> None:
    output_path = _write_shard(tmp_path / "output.pkl", _event_shard(30, "output"))
    icu_mapping = pd.DataFrame({"ICUSTAY_ID": [10], "HADM_ID": [100]})

    with pytest.raises(ValueError, match="unmapped ICUSTAY_ID"):
        preprocess.collect_event_fragments([], [], [output_path], [], icu_mapping)


def test_collect_event_fragments_rejects_duplicate_icustay_mapping(tmp_path: Path) -> None:
    output_path = _write_shard(tmp_path / "output.pkl", _event_shard(10, "output"))
    duplicate_mapping = pd.DataFrame(
        {
            "ICUSTAY_ID": [10, 10],
            "HADM_ID": [100, 101],
        }
    )

    with pytest.raises(ValueError, match="conflicting duplicate ICUSTAY_ID"):
        preprocess.collect_event_fragments([], [], [output_path], [], duplicate_mapping)


def test_build_canonical_payload_preserves_mortality_and_identifier_contract() -> None:
    events = pd.DataFrame(
        {
            "ICUSTAY_ID": [20, 10],
            "rel_charttime": [15, 5],
            "NAME": ["RR", "HR"],
            "VALUENUM": [18.0, 80.0],
        }
    )
    final_icu = pd.DataFrame(
        {
            "ICUSTAY_ID": [10, 20],
            "HADM_ID": [100, 200],
            "INTIME": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "OUTTIME": pd.to_datetime(["2020-01-03", "2020-02-04"]),
            "AGE": [65, 70],
            "GENDER": ["M", "F"],
        }
    )
    admissions = pd.DataFrame(
        {
            "HADM_ID": [100, 200],
            "DEATHTIME": [pd.NaT, pd.Timestamp("2020-02-03")],
            "HOSPITAL_EXPIRE_FLAG": [1, 0],
        }
    )

    ts, oc, ts_ids = preprocess.build_canonical_payload(events, final_icu, admissions)

    assert list(ts.columns) == CANONICAL_TS_COLUMNS
    assert list(oc.columns) == CANONICAL_OC_COLUMNS
    assert ts_ids == ["10", "20"]
    assert set(ts["ts_id"]) == set(oc["ts_id"]) == set(ts_ids)
    assert oc.set_index("ts_id")["in_hospital_mortality"].to_dict() == {"10": 1, "20": 0}


@pytest.mark.parametrize("missing_column", ["ICUSTAY_ID", "HADM_ID"])
def test_mapping_requires_both_identifier_columns(missing_column: str) -> None:
    mapping = pd.DataFrame({"ICUSTAY_ID": [10], "HADM_ID": [100]}).drop(
        columns=[missing_column]
    )
    with pytest.raises(KeyError, match="missing required columns"):
        preprocess.validate_icustay_hadm_mapping(mapping)


@pytest.mark.parametrize(
    ("column", "value"),
    [("ICUSTAY_ID", None), ("HADM_ID", pd.NA)],
)
def test_mapping_rejects_null_identifiers(column: str, value: object) -> None:
    mapping = pd.DataFrame({"ICUSTAY_ID": [10], "HADM_ID": [100]})
    mapping.loc[0, column] = value
    with pytest.raises(ValueError, match="missing value"):
        preprocess.validate_icustay_hadm_mapping(mapping)


def test_identical_duplicate_mapping_collapses_after_validation() -> None:
    mapping = pd.DataFrame(
        {
            "ICUSTAY_ID": [10, "10.0"],
            "HADM_ID": [100, "100.0"],
        }
    )
    validated = preprocess.validate_icustay_hadm_mapping(mapping)
    assert validated.to_dict("records") == [{"ICUSTAY_ID": "10", "HADM_ID": "100"}]


def test_linkage_preserves_multirow_count_and_normalizes_mixed_ids() -> None:
    frame = pd.DataFrame(
        {
            "ICUSTAY_ID": [10.0, "20"],
            "VALUE": [1.0, 2.0],
        }
    )
    mapping = pd.DataFrame(
        {
            "ICUSTAY_ID": ["10.0", 20],
            "HADM_ID": [100.0, "200"],
        }
    )

    linked = preprocess.link_fragment_hadm_ids(frame, mapping, "synthetic")

    assert len(linked) == len(frame)
    assert linked["ICUSTAY_ID"].tolist() == ["10", "20"]
    assert linked["HADM_ID"].tolist() == ["100", "200"]
    assert linked["VALUE"].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("invalid_id", [10.5, "abc", "010", "1e1"])
def test_mapping_rejects_lossy_or_ambiguous_ids(invalid_id: object) -> None:
    mapping = pd.DataFrame({"ICUSTAY_ID": [invalid_id], "HADM_ID": [100]})
    with pytest.raises(ValueError):
        preprocess.validate_icustay_hadm_mapping(mapping)


def test_source_hadm_conflict_is_not_overwritten() -> None:
    frame = pd.DataFrame({"ICUSTAY_ID": [10], "HADM_ID": [999]})
    mapping = pd.DataFrame({"ICUSTAY_ID": [10], "HADM_ID": [100]})
    with pytest.raises(ValueError, match="conflicts with the canonical ICU mapping"):
        preprocess.link_fragment_hadm_ids(frame, mapping, "chartevents")


def test_identical_admissions_collapse_and_conflicts_fail() -> None:
    identical = pd.DataFrame(
        {
            "HADM_ID": [100, "100.0"],
            "DEATHTIME": [pd.NaT, pd.NaT],
            "HOSPITAL_EXPIRE_FLAG": [1, "1.0"],
        }
    )
    normalized = preprocess.normalize_admissions(identical)
    assert len(normalized) == 1
    assert normalized.loc[0, "HADM_ID"] == "100"
    assert normalized.loc[0, "HOSPITAL_EXPIRE_FLAG"] == 1

    conflicting = identical.copy()
    conflicting.loc[1, "HOSPITAL_EXPIRE_FLAG"] = 0
    with pytest.raises(ValueError, match="conflicting"):
        preprocess.normalize_admissions(conflicting)


def _duplicate_icu_rows(second_hadm: object = "100.0") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ICUSTAY_ID": [10, "10.0"],
            "HADM_ID": [100, second_hadm],
            "INTIME": ["2020-01-01", "2020-01-01"],
            "OUTTIME": ["2020-01-03", "2020-01-03"],
            "AGE": [65, 65],
            "GENDER": ["F", "F"],
        }
    )


def test_identical_icu_cohort_rows_collapse_and_conflicts_fail() -> None:
    normalized = preprocess.normalize_icu_cohort(_duplicate_icu_rows())
    assert len(normalized) == 1
    assert normalized.loc[0, "ICUSTAY_ID"] == "10"

    with pytest.raises(ValueError, match="conflicting"):
        preprocess.normalize_icu_cohort(_duplicate_icu_rows(second_hadm=101))


def test_build_canonical_payload_rejects_empty_event_cohort() -> None:
    events = pd.DataFrame(
        columns=["ICUSTAY_ID", "rel_charttime", "NAME", "VALUENUM"]
    )
    admissions = pd.DataFrame(
        {"HADM_ID": [100], "HOSPITAL_EXPIRE_FLAG": [0]}
    )
    with pytest.raises(ValueError, match="events must not be empty"):
        preprocess.build_canonical_payload(events, _duplicate_icu_rows().iloc[[0]], admissions)


def test_normalize_raw_data_path_covers_absolute_relative_tilde_and_nonexistent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    caller = tmp_path / "caller"
    home = tmp_path / "home"
    caller.mkdir()
    home.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setenv("HOME", str(home))

    absolute = tmp_path / "absolute"
    assert preprocess.normalize_raw_data_path(absolute) == absolute.resolve()
    assert preprocess.normalize_raw_data_path("../relative") == (tmp_path / "relative").resolve()
    assert preprocess.normalize_raw_data_path("~/raw") == (home / "raw").resolve()
    nonexistent = tmp_path / "does-not-exist"
    assert preprocess.normalize_raw_data_path(nonexistent) == nonexistent.resolve()
    assert not nonexistent.exists()


def test_add_feature_rows_restores_exact_gcs_tagger_inventory() -> None:
    historical_mapping = {
        184: "GCS_eye",
        220739: "GCS_eye",
        454: "GCS_motor",
        223901: "GCS_motor",
        723: "GCS_verbal",
        223900: "GCS_verbal",
    }
    chart_times = pd.date_range("2020-01-01", periods=len(historical_mapping), freq="min")
    frame = pd.DataFrame(
        {
            "HADM_ID": [100] * len(historical_mapping),
            "ICUSTAY_ID": [10] * len(historical_mapping),
            "ITEMID": list(historical_mapping),
            "CHARTTIME": chart_times,
            "VALUENUM": [4, 3, 6, 5, 5, 4],
            "TABLE": ["chart"] * len(historical_mapping),
        }
    )
    lab_collision = frame.iloc[[0]].copy()
    lab_collision["CHARTTIME"] = pd.Timestamp("2020-01-02")
    lab_collision["TABLE"] = "lab"
    frame = pd.concat([frame, lab_collision], ignore_index=True)

    fragments: list[pd.DataFrame] = []
    preprocess.add_feature_rows(frame, fragments)
    emitted = pd.concat(fragments, ignore_index=True)

    expected_by_time = dict(zip(chart_times, historical_mapping.values()))
    actual_by_time = dict(zip(emitted["CHARTTIME"], emitted["NAME"]))
    assert actual_by_time == expected_by_time
    assert set(emitted["NAME"]) == set(PICKLE_GCS_COMPONENTS)
    assert len(emitted) == len(historical_mapping)
    assert set(emitted["TABLE"]) == {"chart"}


def _run_chartevents_predicate(
    monkeypatch,
    tmp_path: Path,
    chunk: pd.DataFrame,
) -> pd.DataFrame:
    filtered_chunks: list[pd.DataFrame] = []

    def fake_stream_filtered_shards(
        raw_path,
        usecols,
        tmp_dir,
        stem,
        predicate,
        chunksize,
        max_debug_chunks=None,
    ):
        filtered_chunks.append(predicate(chunk))
        return []

    monkeypatch.setattr(
        preprocess,
        "stream_filtered_shards",
        fake_stream_filtered_shards,
    )
    result = preprocess.build_chartevents_shards(
        tmp_path / "raw",
        pd.DataFrame({"ICUSTAY_ID": [10]}),
        chunksize=10,
        tmp_dir=tmp_path / "tmp",
    )

    assert result == []
    return filtered_chunks[0]


def test_chartevents_filter_ignores_irrelevant_null_and_malformed_ids(
    monkeypatch,
    tmp_path: Path,
) -> None:
    chunk = pd.DataFrame(
        {
            "HADM_ID": [100, "abc"],
            "ICUSTAY_ID": [10, None],
            "ITEMID": [211, 211],
            "CHARTTIME": ["2020-01-01", "2020-01-02"],
            "VALUE": [None, None],
            "VALUENUM": [80.0, 81.0],
            "VALUEUOM": ["bpm", "bpm"],
            "ERROR": [0, 0],
        }
    )

    filtered = _run_chartevents_predicate(monkeypatch, tmp_path, chunk)

    assert filtered[["ICUSTAY_ID", "HADM_ID"]].to_dict("records") == [
        {"ICUSTAY_ID": "10", "HADM_ID": "100"}
    ]


def test_chartevents_filter_rejects_selected_row_missing_hadm_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    chunk = pd.DataFrame(
        {
            "HADM_ID": [None],
            "ICUSTAY_ID": [10],
            "ITEMID": [211],
            "CHARTTIME": ["2020-01-01"],
            "VALUE": [None],
            "VALUENUM": [80.0],
            "VALUEUOM": ["bpm"],
            "ERROR": [0],
        }
    )

    with pytest.raises(ValueError, match="missing value"):
        _run_chartevents_predicate(monkeypatch, tmp_path, chunk)


def test_candidate_id_membership_is_lossless_above_float_precision() -> None:
    exact_large_id = str(2**53 + 1)
    candidates = pd.Series(
        [exact_large_id, float(2**53 + 1), None, "0010"],
        name="ICUSTAY_ID",
        dtype="object",
    )

    mask = preprocess.candidate_identifier_membership_mask(
        candidates,
        {exact_large_id, "10"},
    )

    assert mask.tolist() == [True, False, False, False]
