from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from split_predicted_latent_tags import split_predicted_latent_tags  # noqa: E402


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "combined.csv",
        tmp_path / "probabilities.csv",
        tmp_path / "tags.csv",
    )


def _write_input(tmp_path: Path, content: str) -> tuple[Path, Path, Path]:
    input_path, prob_path, tag_path = _paths(tmp_path)
    input_path.write_text(content, encoding="utf-8")
    return input_path, prob_path, tag_path


def _read_rows(path: Path) -> list[list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_valid_split_canonicalizes_ts_ids_and_preserves_values(tmp_path: Path) -> None:
    input_path, prob_path, tag_path = _write_input(
        tmp_path,
        "ts_id,LAT_A_prob,LAT_B_prob,LAT_A_tag,LAT_B_tag\n"
        "10.0,0.1,0.5,0,1\n"
        "11,1,0,1,0\n",
    )

    prob_columns, tag_columns, row_count = split_predicted_latent_tags(
        input_path,
        prob_path,
        tag_path,
    )

    assert prob_columns == ["LAT_A_prob", "LAT_B_prob"]
    assert tag_columns == ["LAT_A_tag", "LAT_B_tag"]
    assert row_count == 2
    assert _read_rows(prob_path) == [
        ["ts_id", "LAT_A_prob", "LAT_B_prob"],
        ["10", "0.1", "0.5"],
        ["11", "1", "0"],
    ]
    assert _read_rows(tag_path) == [
        ["ts_id", "LAT_A_tag", "LAT_B_tag"],
        ["10", "0", "1"],
        ["11", "1", "0"],
    ]


@pytest.mark.parametrize(
    "content",
    [
        "ts_id,LAT_A_prob,LAT_A_tag\n,0.2,0\n",
        "ts_id,LAT_A_prob,LAT_A_tag\n 10 ,0.2,0\n",
        "ts_id,LAT_A_prob,LAT_A_tag\n10,0.2,0\n10,0.3,0\n",
        "ts_id,LAT_A_prob,LAT_A_tag\n10,0.2,0\n10.0,0.3,0\n",
        "ts_id,LAT_A_prob,LAT_A_tag\nabc,0.2,0\n",
        "ts_id,LAT_A_prob,LAT_A_tag\n10.5,0.2,0\n",
        "ts_id,LAT_A_prob,LAT_A_tag\n0010,0.2,0\n",
        "ts_id,LAT_A_prob,LAT_A_tag\n1e1,0.2,0\n",
    ],
)
def test_invalid_and_duplicate_ids_fail_before_outputs(
    tmp_path: Path,
    content: str,
) -> None:
    input_path, prob_path, tag_path = _write_input(tmp_path, content)

    with pytest.raises(ValueError, match="ts_id"):
        split_predicted_latent_tags(input_path, prob_path, tag_path)

    assert not prob_path.exists()
    assert not tag_path.exists()


@pytest.mark.parametrize(
    "header",
    [
        "ts_id",
        "ts_id,LAT_A_prob",
        "ts_id,LAT_A,LAT_A_tag",
        "ts_id,LAT_A_prob,LAT_B_tag",
        "ts_id,LAT_A_prob,LAT_B_prob,LAT_B_tag,LAT_A_tag",
        "ts_id,LAT_A_prob,LAT_A_prob,LAT_A_tag,LAT_A_tag",
    ],
)
def test_probability_and_tag_columns_must_pair_exactly(
    tmp_path: Path,
    header: str,
) -> None:
    input_path, prob_path, tag_path = _write_input(
        tmp_path,
        f"{header}\n10,0.2,0\n",
    )
    with pytest.raises(ValueError):
        split_predicted_latent_tags(input_path, prob_path, tag_path)


@pytest.mark.parametrize("probability", ["", "abc", "nan", "inf", "-0.1", "1.1"])
def test_invalid_probabilities_fail(
    tmp_path: Path,
    probability: str,
) -> None:
    input_path, prob_path, tag_path = _write_input(
        tmp_path,
        f"ts_id,LAT_A_prob,LAT_A_tag\n10,{probability},0\n",
    )
    with pytest.raises(ValueError, match="[Pp]robability"):
        split_predicted_latent_tags(input_path, prob_path, tag_path)


@pytest.mark.parametrize("tag", ["", "abc", "nan", "inf", "-1", "2", "0.5"])
def test_invalid_binary_tags_fail(tmp_path: Path, tag: str) -> None:
    input_path, prob_path, tag_path = _write_input(
        tmp_path,
        f"ts_id,LAT_A_prob,LAT_A_tag\n10,0.2,{tag}\n",
    )
    with pytest.raises(ValueError, match="[Tt]ag"):
        split_predicted_latent_tags(input_path, prob_path, tag_path)


@pytest.mark.parametrize(
    ("probability", "tag"),
    [("0.49", "1"), ("0.5", "0"), ("1", "0"), ("0", "1")],
)
def test_tags_must_match_half_probability_threshold(
    tmp_path: Path,
    probability: str,
    tag: str,
) -> None:
    input_path, prob_path, tag_path = _write_input(
        tmp_path,
        f"ts_id,LAT_A_prob,LAT_A_tag\n10,{probability},{tag}\n",
    )
    with pytest.raises(ValueError, match="Threshold-inconsistent"):
        split_predicted_latent_tags(input_path, prob_path, tag_path)


def test_empty_prediction_cohort_fails_before_outputs(tmp_path: Path) -> None:
    input_path, prob_path, tag_path = _write_input(
        tmp_path,
        "ts_id,LAT_A_prob,LAT_A_tag\n",
    )
    with pytest.raises(ValueError, match="no prediction rows"):
        split_predicted_latent_tags(input_path, prob_path, tag_path)
    assert not prob_path.exists()
    assert not tag_path.exists()
