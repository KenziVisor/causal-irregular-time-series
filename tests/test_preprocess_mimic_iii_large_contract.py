from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess_mimic_iii_large_contract import (  # noqa: E402
    CANONICAL_OC_COLUMNS,
    CANONICAL_TS_COLUMNS,
    assert_physionet_compatible_output,
    assert_exact_id_cohort,
    build_canonical_oc,
    build_canonical_ts,
    build_ts_ids,
    canonicalize_binary_mortality_series,
    canonicalize_mimic_id_series,
    canonicalize_unique_id_frame,
)


def _icu_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_id": [101, 202, 303],
            "HADM_ID": [11, 22, 33],
            "INTIME": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "OUTTIME": pd.to_datetime(
                [
                    "2020-01-03 00:00:00",
                    "2020-01-02 12:00:00",
                    "2020-01-04 00:00:00",
                ]
            ),
        }
    )


def _admission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "HADM_ID": [11, 22, 33],
            "HOSPITAL_EXPIRE_FLAG": [1, 0, 0],
        }
    )


def test_synthetic_main_contract_is_collected_by_pytest() -> None:
    events = pd.DataFrame(
        {
            "ts_id": [202, 101, 101, 101],
            "minute": [10, 0, 0, 5],
            "variable": ["Temperature", "HR", "HR", "RR"],
            "value": [37.0, 80.0, 82.0, 18.0],
            "TABLE": ["chart", "chart", "lab", "chart"],
        }
    )

    ts = build_canonical_ts(events)
    ts_ids = build_ts_ids(ts)
    oc = build_canonical_oc(_icu_rows(), _admission_rows(), valid_ts_ids=ts_ids)

    assert list(ts.columns) == CANONICAL_TS_COLUMNS
    assert list(oc.columns) == CANONICAL_OC_COLUMNS
    assert ts_ids == ["101", "202"]
    averaged_hr = ts.loc[
        (ts["ts_id"] == "101")
        & (ts["minute"] == 0)
        & (ts["variable"] == "HR"),
        "value",
    ].iloc[0]
    assert averaged_hr == 81.0
    assert set(oc["ts_id"]) == {"101", "202"}
    assert "HADM_ID" not in oc.columns
    assert "SUBJECT_ID" not in oc.columns
    assert oc.loc[oc["ts_id"] == "101", "subset"].iloc[0] == "mimic_iii"
    assert oc.loc[oc["ts_id"] == "202", "length_of_stay"].iloc[0] == 0.5
    assert_physionet_compatible_output(ts, oc, ts_ids)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (10, "10"),
        (np.int64(10), "10"),
        (10.0, "10"),
        ("10", "10"),
        ("10.0", "10"),
        (" 10.00 ", "10"),
    ],
)
def test_identifier_contract_accepts_exact_integer_representations(
    value: object,
    expected: str,
) -> None:
    result = canonicalize_mimic_id_series(pd.Series([value], dtype="object"))
    assert result.tolist() == [expected]


def test_identifier_contract_accepts_nonnull_nullable_integer_series() -> None:
    values = pd.Series([10, 11], dtype="Int64", name="ICUSTAY_ID")
    assert canonicalize_mimic_id_series(values).tolist() == ["10", "11"]


@pytest.mark.parametrize(
    "value",
    [
        None,
        pd.NA,
        np.nan,
        True,
        np.bool_(False),
        10.5,
        np.inf,
        float(2**53 + 1),
        -1,
        -0.0,
        "",
        "abc",
        "10.5",
        "1e1",
        "010",
        "+10",
        "-1",
        "NaN",
        "inf",
    ],
)
def test_identifier_contract_rejects_missing_fractional_and_ambiguous_values(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonicalize_mimic_id_series(pd.Series([value], dtype="object"))


def test_identical_mapping_admission_and_cohort_duplicates_collapse() -> None:
    icu = pd.DataFrame(
        {
            "ts_id": [10, "10.0"],
            "HADM_ID": [100, "100.0"],
            "INTIME": ["2020-01-01", "2020-01-01"],
            "OUTTIME": ["2020-01-03", "2020-01-03"],
        }
    )
    admissions = pd.DataFrame(
        {
            "HADM_ID": [100, "100.0"],
            "HOSPITAL_EXPIRE_FLAG": [1, "1.0"],
        }
    )

    oc = build_canonical_oc(icu, admissions, valid_ts_ids=[10.0])

    assert len(oc) == 1
    assert oc.loc[0, "ts_id"] == "10"
    assert oc.loc[0, "in_hospital_mortality"] == 1


def test_conflicting_admissions_fail_without_first_row_selection() -> None:
    admissions = pd.DataFrame(
        {
            "HADM_ID": [11, "11.0"],
            "HOSPITAL_EXPIRE_FLAG": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="conflicting"):
        build_canonical_oc(_icu_rows().iloc[[0]], admissions)


def test_conflicting_duplicate_cohort_rows_fail() -> None:
    icu = pd.DataFrame(
        {
            "ts_id": [10, "10.0"],
            "HADM_ID": [100, 101],
            "INTIME": ["2020-01-01", "2020-01-01"],
            "OUTTIME": ["2020-01-03", "2020-01-03"],
        }
    )
    admissions = pd.DataFrame(
        {
            "HADM_ID": [100, 101],
            "HOSPITAL_EXPIRE_FLAG": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="conflicting"):
        build_canonical_oc(icu, admissions)


@pytest.mark.parametrize("mortality", [None, np.nan, np.inf, -1, 2, "dead"])
def test_mortality_contract_requires_complete_finite_binary_values(
    mortality: object,
) -> None:
    with pytest.raises(ValueError):
        canonicalize_binary_mortality_series(
            pd.Series([mortality], dtype="object")
        )


def test_missing_admission_mortality_fails() -> None:
    admissions = pd.DataFrame(
        {"HADM_ID": [999], "HOSPITAL_EXPIRE_FLAG": [0]}
    )
    with pytest.raises(ValueError, match="missing mortality"):
        build_canonical_oc(_icu_rows().iloc[[0]], admissions)


def test_requested_cohort_must_exist_in_icu_rows() -> None:
    with pytest.raises(ValueError, match="missing 1 identifier"):
        build_canonical_oc(
            _icu_rows().iloc[[0]],
            _admission_rows(),
            valid_ts_ids=["101", "999"],
        )


def test_empty_canonical_cohorts_fail() -> None:
    empty_events = pd.DataFrame(columns=CANONICAL_TS_COLUMNS)
    with pytest.raises(ValueError, match="must not be empty"):
        build_ts_ids(build_canonical_ts(empty_events))

    empty_icu = pd.DataFrame(columns=["ts_id", "HADM_ID", "INTIME", "OUTTIME"])
    with pytest.raises(ValueError, match="must not be empty"):
        build_canonical_oc(empty_icu, _admission_rows())


def test_output_assertion_rejects_duplicate_outcome_ids() -> None:
    events = pd.DataFrame(
        {"ts_id": [101], "minute": [0], "variable": ["HR"], "value": [80.0]}
    )
    ts = build_canonical_ts(events)
    oc = build_canonical_oc(
        _icu_rows().iloc[[0]],
        _admission_rows(),
        valid_ts_ids=["101"],
    )
    duplicate_oc = pd.concat([oc, oc], ignore_index=True)

    with pytest.raises(AssertionError, match="exactly one row"):
        assert_physionet_compatible_output(ts, duplicate_oc, ["101"])


def test_unique_id_frame_collapses_only_exact_semantic_duplicates() -> None:
    frame = pd.DataFrame(
        {"ts_id": ["10", "10.0"], "LAT_A": [1, 1]}
    )

    normalized = canonicalize_unique_id_frame(
        frame,
        frame_name="latent tags",
    )

    assert normalized.to_dict("records") == [{"ts_id": "10", "LAT_A": 1}]
    id_only = canonicalize_unique_id_frame(
        pd.DataFrame({"ts_id": ["10", "10.00"]}),
        frame_name="ID-only cohort",
    )
    assert id_only["ts_id"].tolist() == ["10"]


def test_unique_id_frame_rejects_conflicting_semantic_duplicates() -> None:
    frame = pd.DataFrame(
        {"ts_id": ["10", "10.0"], "LAT_A": [0, 1]}
    )

    with pytest.raises(ValueError, match="conflicting duplicate ts_id"):
        canonicalize_unique_id_frame(frame, frame_name="latent tags")


def test_exact_id_cohort_allows_reordering_but_not_set_drift() -> None:
    assert_exact_id_cohort(
        ["10", "11.0"],
        [11, 10.0],
        reference_name="processed outcomes",
        candidate_name="latent tags",
    )

    with pytest.raises(
        ValueError,
        match=r"missing_count=1; extra_count=1",
    ):
        assert_exact_id_cohort(
            ["10", "11"],
            ["10", "12"],
            reference_name="processed outcomes",
            candidate_name="latent tags",
        )


def test_exact_id_cohort_rejects_semantic_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate ts_id"):
        assert_exact_id_cohort(
            ["10"],
            ["10", "10.0"],
            reference_name="processed outcomes",
            candidate_name="latent tags",
        )
