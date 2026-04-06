from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocess_mimic_iii_large_contract import (  # noqa: E402
    CANONICAL_OC_COLUMNS,
    CANONICAL_TS_COLUMNS,
    assert_physionet_compatible_output,
    build_canonical_oc,
    build_canonical_ts,
    build_ts_ids,
)


def main() -> None:
    # This synthetic check validates only the compatibility layer, not the full
    # runtime behavior on real MIMIC-III files.
    events = pd.DataFrame(
        {
            "ts_id": [202, 101, 101, 101],
            "minute": [10, 0, 0, 5],
            "variable": ["Temperature", "HR", "HR", "RR"],
            "value": [37.0, 80.0, 82.0, 18.0],
            "TABLE": ["chart", "chart", "lab", "chart"],
        }
    )
    icu_full = pd.DataFrame(
        {
            "ts_id": [101, 202, 303],
            "HADM_ID": [11, 22, 33],
            "INTIME": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"]
            ),
            "OUTTIME": pd.to_datetime(
                ["2020-01-03 00:00:00", "2020-01-02 12:00:00", "2020-01-04 00:00:00"]
            ),
        }
    )
    admissions = pd.DataFrame(
        {
            "HADM_ID": [11, 22, 33],
            "HOSPITAL_EXPIRE_FLAG": [1, 0, 0],
        }
    )

    ts = build_canonical_ts(events)
    ts_ids = build_ts_ids(ts)
    oc = build_canonical_oc(icu_full, admissions, valid_ts_ids=ts_ids)

    assert list(ts.columns) == CANONICAL_TS_COLUMNS
    assert list(oc.columns) == CANONICAL_OC_COLUMNS
    assert ts_ids == ["101", "202"]
    assert ts.loc[(ts["ts_id"] == "101") & (ts["minute"] == 0) & (ts["variable"] == "HR"), "value"].iloc[0] == 81.0
    assert set(oc["ts_id"]) == {"101", "202"}
    assert "HADM_ID" not in oc.columns
    assert "SUBJECT_ID" not in oc.columns
    assert oc.loc[oc["ts_id"] == "101", "subset"].iloc[0] == "mimic_iii"
    assert oc.loc[oc["ts_id"] == "202", "length_of_stay"].iloc[0] == 0.5

    assert_physionet_compatible_output(ts, oc, ts_ids)
    print("Synthetic preprocess_mimic_iii_large compatibility checks passed.")


if __name__ == "__main__":
    main()
