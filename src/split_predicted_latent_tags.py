from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from preprocess_mimic_iii_large_contract import canonicalize_mimic_id_scalar

if "--validate-config-only" in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_config import maybe_run_validate_config_only

    maybe_run_validate_config_only(
        "src/split_predicted_latent_tags.py",
        default_dataset="physionet",
    )

DATASET_MODEL = "physionet"
INPUT_CSV = Path("../../data/predicted_latent_tags_230326.csv")
TAG_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a combined predicted-latents CSV into probability and tag tables."
    )
    parser.add_argument(
        "--model",
        choices=["physionet", "mimic"],
        default=DATASET_MODEL,
        help=f"Dataset selector for input defaults. Default: {DATASET_MODEL}",
    )
    parser.add_argument(
        "--dataset-config-csv",
        default=None,
        help=(
            "Path to the dataset global-variables CSV. If omitted, use the default "
            "config for --model."
        ),
    )
    parser.add_argument("--input-csv", default=None)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Resolve dataset config values and exit without loading data.",
    )
    return parser.parse_args()


def default_output_paths(input_csv: Path) -> tuple[Path, Path]:
    prob_output = input_csv.with_name(f"{input_csv.stem}_probabilities.csv")
    tag_output = input_csv.with_name(f"{input_csv.stem}_absolute_tags.csv")
    return prob_output, tag_output


def write_csv(output_path: Path, header: list[str], rows: list[list[str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _parse_probability(value: str, *, row_number: int, column: str) -> float:
    if not value.strip():
        raise ValueError(
            f"Missing probability at row {row_number}, column {column!r}."
        )
    try:
        probability = float(value)
    except ValueError as error:
        raise ValueError(
            f"Malformed probability at row {row_number}, column {column!r}."
        ) from error
    if not math.isfinite(probability):
        raise ValueError(
            f"Non-finite probability at row {row_number}, column {column!r}."
        )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Probability outside [0, 1] at row {row_number}, column {column!r}."
        )
    return probability


def _parse_binary_tag(value: str, *, row_number: int, column: str) -> int:
    if not value.strip():
        raise ValueError(f"Missing binary tag at row {row_number}, column {column!r}.")
    try:
        numeric_tag = float(value)
    except ValueError as error:
        raise ValueError(
            f"Malformed binary tag at row {row_number}, column {column!r}."
        ) from error
    if not math.isfinite(numeric_tag) or numeric_tag not in {0.0, 1.0}:
        raise ValueError(
            f"Invalid binary tag at row {row_number}, column {column!r}; expected 0 or 1."
        )
    return int(numeric_tag)


def _validate_paired_columns(header: list[str]) -> tuple[list[str], list[str]]:
    if not header:
        raise ValueError("Input CSV has an empty header row.")
    if len(header) != len(set(header)):
        raise ValueError("Input CSV contains duplicate column names.")
    if header[0] != "ts_id":
        raise ValueError(
            f"Expected the first column to be 'ts_id', got {header[0]!r} instead."
        )

    non_key_columns = header[1:]
    if len(non_key_columns) == 0 or len(non_key_columns) % 2 != 0:
        raise ValueError(
            "Expected a non-zero, even number of non-key columns after 'ts_id'."
        )

    midpoint = len(non_key_columns) // 2
    prob_columns = non_key_columns[:midpoint]
    tag_columns = non_key_columns[midpoint:]
    if any(not column.endswith("_prob") for column in prob_columns):
        raise ValueError("Every probability column must end with '_prob'.")
    if any(not column.endswith("_tag") for column in tag_columns):
        raise ValueError("Every tag column must end with '_tag'.")

    prob_bases = [column.removesuffix("_prob") for column in prob_columns]
    tag_bases = [column.removesuffix("_tag") for column in tag_columns]
    if prob_bases != tag_bases:
        raise ValueError(
            "Probability and tag columns must form exact, ordered base-name pairs."
        )
    return prob_columns, tag_columns


def split_predicted_latent_tags(
    input_csv: Path,
    prob_output: Path,
    tag_output: Path,
    *,
    threshold: float = TAG_THRESHOLD,
) -> tuple[list[str], list[str], int]:
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Tag threshold must be finite and within [0, 1].")

    print(f"[2/4] Reading combined latent CSV from: {input_csv.resolve()}")
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input CSV is empty: {input_csv}") from exc

        prob_columns, tag_columns = _validate_paired_columns(header)
        midpoint = len(prob_columns)
        prob_rows: list[list[str]] = []
        tag_rows: list[list[str]] = []
        seen_ts_ids: set[str] = set()

        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"Row {row_number} has {len(row)} columns, expected {len(header)}."
                )

            raw_ts_id = row[0]
            if raw_ts_id != raw_ts_id.strip():
                raise ValueError(
                    f"Row {row_number} contains a whitespace-padded ts_id."
                )
            try:
                ts_id = canonicalize_mimic_id_scalar(
                    raw_ts_id,
                    field_name="input prediction ts_id",
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Row {row_number} contains an invalid ts_id. {error}"
                ) from error
            if ts_id in seen_ts_ids:
                raise ValueError(
                    "Input CSV contains a semantic duplicate ts_id after "
                    f"canonicalization at row {row_number}."
                )
            seen_ts_ids.add(ts_id)

            probability_values = row[1 : 1 + midpoint]
            tag_values = row[1 + midpoint :]
            for prob_column, tag_column, prob_text, tag_text in zip(
                prob_columns,
                tag_columns,
                probability_values,
                tag_values,
            ):
                probability = _parse_probability(
                    prob_text,
                    row_number=row_number,
                    column=prob_column,
                )
                tag = _parse_binary_tag(
                    tag_text,
                    row_number=row_number,
                    column=tag_column,
                )
                expected_tag = int(probability >= threshold)
                if tag != expected_tag:
                    raise ValueError(
                        f"Threshold-inconsistent tag at row {row_number} for "
                        f"paired columns {prob_column!r}/{tag_column!r}."
                    )

            prob_rows.append([ts_id, *probability_values])
            tag_rows.append([ts_id, *tag_values])
            if len(prob_rows) % 50000 == 0:
                print(f"      Processed {len(prob_rows):,} data rows")

    processed_rows = len(prob_rows)
    if processed_rows == 0:
        raise ValueError("Input CSV contains no prediction rows.")

    print(
        f"[3/4] Writing split outputs ({processed_rows:,} rows | "
        f"{len(prob_columns)} probability columns | {len(tag_columns)} tag columns)"
    )
    write_csv(prob_output, ["ts_id", *prob_columns], prob_rows)
    write_csv(tag_output, ["ts_id", *tag_columns], tag_rows)

    return prob_columns, tag_columns, processed_rows


def main() -> None:
    args = parse_args()
    input_csv_value = args.input_csv if args.input_csv is not None else str(INPUT_CSV)
    if input_csv_value is None:
        raise ValueError(
            "Provide --input-csv because this repo does not define a safe checked-in "
            "default combined predicted-latents CSV."
        )

    input_csv = Path(str(input_csv_value))

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print("=== Splitting predicted latent tags ===")
    print(f"[1/4] Runtime configuration: model={args.model} | input={input_csv.resolve()}")
    prob_output, tag_output = default_output_paths(input_csv)
    prob_columns, tag_columns, row_count = split_predicted_latent_tags(
        input_csv=input_csv,
        prob_output=prob_output,
        tag_output=tag_output,
    )

    print("[4/4] Finished writing split CSV files")
    print(f"Saved probabilities table to: {prob_output}")
    print(f"Saved absolute tags table to: {tag_output}")
    print(f"Rows processed: {row_count:,}")
    print(f"Probability columns: {len(prob_columns)}")
    print(f"Absolute tag columns: {len(tag_columns)}")


if __name__ == "__main__":
    main()
