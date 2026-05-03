from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


DATASET_CHOICES = {"physionet", "mimic"}
MODEL_TYPE_CHOICES = {"CausalForest", "LinearDML"}

COMPACT_CONFIG_FIELDS = [
    "PREFERRED_ENV_NAME",
    "SEED",
    "DATASET_NAME",
    "ID_COL",
    "ALT_ID_COL",
    "OUTCOME_COL",
    "GRAPH_OUTCOME_NODE",
    "TREATMENTS",
    "LATENT_ORDER",
    "BACKGROUND_FEATURE_COLUMNS",
    "EFFECT_MODIFIER_COLUMNS",
    "MODEL_TYPE",
    "TRIALS",
    "DOWN_SAMPLE",
    "USE_EXPANDED_SAFE_CONFOUNDERS",
    "SAVE_CONTOUR_PLOT",
    "MATCH_WITH_REPLACEMENT",
    "REQUIRE_BINARY_CONF",
]

KNOWN_LIST_KEYS = {
    "TREATMENTS",
    "LATENT_ORDER",
    "BACKGROUND_FEATURE_COLUMNS",
    "EFFECT_MODIFIER_COLUMNS",
}

KNOWN_BOOL_KEYS = {
    "DOWN_SAMPLE",
    "USE_EXPANDED_SAFE_CONFOUNDERS",
    "SAVE_CONTOUR_PLOT",
    "MATCH_WITH_REPLACEMENT",
    "REQUIRE_BINARY_CONF",
}

KNOWN_INT_KEYS = {
    "SEED",
    "TRIALS",
}

KNOWN_FLOAT_KEYS: set[str] = set()
REQUIRED_CANONICAL_KEYS = list(COMPACT_CONFIG_FIELDS)
MIMIC_ONLY_KEYS: list[str] = []

SCRIPT_CONFIG_CONTRACTS: dict[str, dict[str, object]] = {
    "main.py": {
        "required_keys": ["MODEL_TYPE", "TRIALS"],
        "alias_groups": {},
    },
    "src/cate_estimation.py": {
        "required_keys": [
            "OUTCOME_COL",
            "GRAPH_OUTCOME_NODE",
            "TREATMENTS",
            "BACKGROUND_FEATURE_COLUMNS",
            "EFFECT_MODIFIER_COLUMNS",
            "SEED",
            "DOWN_SAMPLE",
            "USE_EXPANDED_SAFE_CONFOUNDERS",
            "MODEL_TYPE",
        ],
        "alias_groups": {},
    },
    "src/matching_causal_effect.py": {
        "required_keys": [
            "OUTCOME_COL",
            "GRAPH_OUTCOME_NODE",
            "TREATMENTS",
            "BACKGROUND_FEATURE_COLUMNS",
            "SEED",
            "DOWN_SAMPLE",
            "USE_EXPANDED_SAFE_CONFOUNDERS",
            "MATCH_WITH_REPLACEMENT",
            "REQUIRE_BINARY_CONF",
        ],
        "alias_groups": {},
    },
    "src/mortality_prediction_using_latents.py": {
        "required_keys": ["OUTCOME_COL", "SEED"],
        "alias_groups": {},
    },
    "src/analyze_cate_results.py": {
        "required_keys": [
            "PREFERRED_ENV_NAME",
            "OUTCOME_COL",
            "BACKGROUND_FEATURE_COLUMNS",
            "SEED",
            "SAVE_CONTOUR_PLOT",
        ],
        "alias_groups": {},
    },
    "src/permutations_test.py": {
        "required_keys": ["OUTCOME_COL", "TRIALS", "MODEL_TYPE", "SEED"],
        "alias_groups": {},
    },
    "src/split_predicted_latent_tags.py": {
        "required_keys": [],
        "alias_groups": {},
    },
    "src/physionet2012_causal_graph.py": {
        "required_keys": [],
        "alias_groups": {},
    },
    "src/mimiciii_causal_graph.py": {
        "required_keys": [],
        "alias_groups": {},
    },
    "src/tagging_latent_variables_physionet.py": {
        "required_keys": ["LATENT_ORDER"],
        "alias_groups": {},
    },
    "src/tagging_latent_variables_mimiciii.py": {
        "required_keys": ["LATENT_ORDER"],
        "alias_groups": {},
    },
    "src/decision_trees_plot.py": {
        "required_keys": ["LATENT_ORDER"],
        "alias_groups": {},
    },
    "src/preprocess_physionet_2012.py": {
        "required_keys": [],
        "alias_groups": {},
    },
    "src/preprocess_mimic_iii_large.py": {
        "required_keys": [],
        "alias_groups": {},
    },
}

_TRUE_VALUES = {"true", "yes", "on", "1"}
_FALSE_VALUES = {"false", "no", "off", "0"}
_INT_RE = re.compile(r"^[+-]?(0|[1-9]\d*)$")
_FLOAT_RE = re.compile(
    r"^[+-]?((\d+\.\d*)|(\.\d+)|(\d+e[+-]?\d+)|(\d+\.\d*e[+-]?\d+)|(\.\d+e[+-]?\d+))$",
    re.IGNORECASE,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_dataset(dataset: str) -> str:
    normalized = str(dataset).strip().lower()
    if normalized not in DATASET_CHOICES:
        raise ValueError(
            f"Invalid dataset {dataset!r}. Expected one of {sorted(DATASET_CHOICES)}."
        )
    return normalized


def default_config_path(dataset: str) -> Path:
    dataset = _normalize_dataset(dataset)
    filename = {
        "physionet": "physionet-global-variables.csv",
        "mimic": "mimic-global-variables.csv",
    }[dataset]
    return repo_root() / "configs" / filename


def _context_message(dataset: str, config_csv_path: str | Path, key: str) -> str:
    return (
        f"dataset={dataset!r}, config file={Path(config_csv_path)}, "
        f"config key={key!r}"
    )


def _parse_value(raw_value: str, *, dataset: str, path: Path, key: str) -> Any:
    value = raw_value.strip()

    if value.startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid JSON value in dataset config "
                f"({_context_message(dataset, path, key)}): {exc}"
            ) from exc

    lowered = value.lower()
    if key in KNOWN_BOOL_KEYS and lowered in (_TRUE_VALUES | _FALSE_VALUES):
        return lowered in _TRUE_VALUES

    if _INT_RE.match(value):
        try:
            return int(value)
        except ValueError:
            pass

    if _FLOAT_RE.match(value):
        try:
            return float(value)
        except ValueError:
            pass

    return value


def _validate_header(dataset: str, path: Path, columns: list[str]) -> None:
    if columns == COMPACT_CONFIG_FIELDS:
        return

    expected = set(COMPACT_CONFIG_FIELDS)
    actual = set(columns)
    missing = [key for key in COMPACT_CONFIG_FIELDS if key not in actual]
    extra = [key for key in columns if key not in expected]
    order_mismatch = not missing and not extra and columns != COMPACT_CONFIG_FIELDS

    details = []
    if missing:
        details.append(f"missing columns={missing}")
    if extra:
        details.append(f"unexpected columns={extra}")
    if order_mismatch:
        details.append("columns are present but not in the compact contract order")

    raise ValueError(
        "Dataset config CSV must use the compact column contract exactly "
        f"({_context_message(dataset, path, '<header>')}): "
        + "; ".join(details)
    )


def _validate_compact_config(config: dict[str, object]) -> None:
    dataset = _config_dataset(config)
    path = _config_path(config)

    empty_allowed = {"ALT_ID_COL"}
    missing_values = [
        key
        for key in COMPACT_CONFIG_FIELDS
        if key not in empty_allowed and not _has_config_value(config.get(key))
    ]
    if missing_values:
        raise ValueError(
            "Dataset config CSV has empty required compact fields "
            f"({_context_message(dataset, path, '<required>')}): {missing_values}"
        )

    model_type = get_config_scalar(config, "MODEL_TYPE", None)
    if model_type not in MODEL_TYPE_CHOICES:
        raise ValueError(
            "Invalid MODEL_TYPE in dataset config "
            f"({_context_message(dataset, path, 'MODEL_TYPE')}): {model_type!r}. "
            f"Allowed values: {sorted(MODEL_TYPE_CHOICES)}"
        )

    for key in sorted(KNOWN_BOOL_KEYS):
        if _has_config_value(config.get(key)):
            get_config_bool(config, key)

    for key in sorted(KNOWN_INT_KEYS):
        if _has_config_value(config.get(key)):
            get_config_int(config, key)


def load_dataset_config(
    dataset: str,
    config_csv_path: str | Path | None = None,
) -> dict[str, object]:
    dataset = _normalize_dataset(dataset)
    path = (
        default_config_path(dataset)
        if config_csv_path is None
        else Path(config_csv_path).expanduser()
    )
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            "Dataset config CSV does not exist "
            f"({_context_message(dataset, path, '<config_csv_path>')})."
        )
    if not path.is_file():
        raise FileNotFoundError(
            "Dataset config path is not a file "
            f"({_context_message(dataset, path, '<config_csv_path>')})."
        )

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(
                "Dataset config CSV is missing a header row "
                f"({_context_message(dataset, path, '<header>')})."
            )

        columns = [field.strip() for field in reader.fieldnames if field and field.strip()]
        _validate_header(dataset, path, columns)
        values_by_key: dict[str, list[str]] = {key: [] for key in columns}

        for row in reader:
            normalized_row = {
                (key.strip() if key else ""): ("" if value is None else str(value).strip())
                for key, value in row.items()
            }
            if not any(normalized_row.get(key, "") for key in columns):
                continue
            for key in columns:
                value = normalized_row.get(key, "")
                if value:
                    values_by_key[key].append(value)

    config: dict[str, object] = {
        "__dataset__": dataset,
        "__config_csv_path__": str(path),
    }
    for key, raw_values in values_by_key.items():
        if not raw_values:
            config[key] = None
            continue

        if key in KNOWN_LIST_KEYS:
            parsed_items: list[Any] = []
            for raw in raw_values:
                parsed = _parse_value(raw, dataset=dataset, path=path, key=key)
                if isinstance(parsed, list):
                    parsed_items.extend(parsed)
                else:
                    parsed_items.append(parsed)
            config[key] = parsed_items
            continue

        if len(raw_values) == 1:
            config[key] = _parse_value(raw_values[0], dataset=dataset, path=path, key=key)
            continue

        config[key] = [
            _parse_value(raw, dataset=dataset, path=path, key=key)
            for raw in raw_values
        ]

    _validate_compact_config(config)
    return config


def _config_dataset(config: dict[str, object]) -> str:
    return str(config.get("__dataset__", "<unknown>"))


def _config_path(config: dict[str, object]) -> str:
    return str(config.get("__config_csv_path__", "<unknown>"))


def get_config_value(config: dict[str, object], key: str, default: Any = None) -> Any:
    value = config.get(key)
    return default if value is None else value


def _has_config_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return any(_has_config_value(item) for item in value)
    return True


def get_first_available(
    config: dict[str, object],
    keys: list[str],
    default: Any = None,
) -> Any:
    for key in keys:
        value = config.get(key)
        if _has_config_value(value):
            return value
    return default


def get_first_available_with_key(
    config: dict[str, object],
    keys: list[str],
    default: Any = None,
) -> tuple[Any, str | None]:
    for key in keys:
        value = config.get(key)
        if _has_config_value(value):
            return value, key
    return default, None


def get_config_required(config: dict[str, object], key: str) -> Any:
    value = config.get(key)
    if value is None:
        raise KeyError(
            "Missing required dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)})."
        )
    return value


def get_config_list(
    config: dict[str, object],
    key: str,
    default: list[Any] | None = None,
) -> list[Any] | None:
    value = get_config_value(config, key, default)
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def get_config_scalar(config: dict[str, object], key: str, default: Any = None) -> Any:
    value = get_config_value(config, key, default)
    if isinstance(value, list):
        raise TypeError(
            "Expected scalar dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got list."
        )
    return value


def _parse_bool(value: Any, *, config: dict[str, object], key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    raise TypeError(
        "Expected boolean dataset config value "
        f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
        f"got {value!r}."
    )


def get_config_bool(
    config: dict[str, object],
    key: str,
    default: bool | None = None,
) -> bool | None:
    value = get_config_scalar(config, key, default)
    if value is None:
        return None
    return _parse_bool(value, config=config, key=key)


def get_config_int(
    config: dict[str, object],
    key: str,
    default: int | None = None,
) -> int | None:
    value = get_config_scalar(config, key, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(
            "Expected integer dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got bool."
        )
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Expected integer dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got {value!r}."
        ) from exc
    if not numeric_value.is_integer():
        raise ValueError(
            "Expected integral dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got {value!r}."
        )
    return int(numeric_value)


def get_config_float(
    config: dict[str, object],
    key: str,
    default: float | None = None,
) -> float | None:
    value = get_config_scalar(config, key, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(
            "Expected float dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got bool."
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Expected float dataset config value "
            f"({_context_message(_config_dataset(config), _config_path(config), key)}), "
            f"got {value!r}."
        ) from exc


def resolve_with_precedence(
    cli_value: Any,
    config: dict[str, object],
    key: str,
    fallback: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value
    config_value = config.get(key)
    if config_value is not None:
        return config_value
    return fallback


def normalize_script_name(script_name: str | Path) -> str:
    value = str(script_name).replace("\\", "/")
    if value.startswith("./"):
        value = value[2:]
    if "/src/" in value:
        value = "src/" + value.rsplit("/src/", 1)[1]
    elif value.endswith("/main.py"):
        value = "main.py"
    return value


def validate_script_config(
    script_name: str | Path,
    config: dict[str, object],
) -> dict[str, dict[str, object]]:
    normalized_script = normalize_script_name(script_name)
    if normalized_script not in SCRIPT_CONFIG_CONTRACTS:
        raise KeyError(f"No dataset config contract is defined for {normalized_script!r}.")

    contract = SCRIPT_CONFIG_CONTRACTS[normalized_script]
    resolved: dict[str, dict[str, object]] = {}
    missing: list[str] = []

    for key in list(contract.get("required_keys", [])):
        value, source_key = get_first_available_with_key(config, [str(key)])
        if source_key is None:
            missing.append(str(key))
        else:
            resolved[str(key)] = {
                "value": value,
                "source_key": source_key,
                "candidate_keys": [str(key)],
            }

    alias_groups = dict(contract.get("alias_groups", {}))
    for name, keys in alias_groups.items():
        candidate_keys = [str(key) for key in keys]
        value, source_key = get_first_available_with_key(config, candidate_keys)
        if source_key is None:
            missing.append(f"{name} ({', '.join(candidate_keys)})")
        else:
            resolved[str(name)] = {
                "value": value,
                "source_key": source_key,
                "candidate_keys": candidate_keys,
            }

    if missing:
        raise KeyError(
            "Missing required config values for "
            f"{normalized_script}: {', '.join(missing)}"
        )

    return resolved


def _format_summary_value(value: Any, limit: int = 120) -> str:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def print_resolved_config_summary(
    script_name: str | Path,
    config: dict[str, object],
    resolved: dict[str, dict[str, object]],
) -> None:
    normalized_script = normalize_script_name(script_name)
    used_fields = list(SCRIPT_CONFIG_CONTRACTS.get(normalized_script, {}).get("required_keys", []))
    print("VALIDATE-CONFIG-ONLY PASS")
    print(f"  dataset: {_config_dataset(config)}")
    print(f"  script: {normalized_script}")
    print(f"  config_csv: {_config_path(config)}")
    print(
        "  compact_fields_used: "
        + (", ".join(str(field) for field in used_fields) if used_fields else "(none)")
    )
    for name in sorted(resolved):
        item = resolved[name]
        value = _format_summary_value(item["value"])
        print(f"  {name}: {value}  [source={item['source_key']}]")


def maybe_run_validate_config_only(
    script_name: str,
    *,
    fixed_dataset: str | None = None,
    default_dataset: str | None = None,
    argv: list[str] | None = None,
) -> None:
    argv = list(sys.argv if argv is None else argv)
    if "--validate-config-only" not in argv:
        return

    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--dataset-config-csv", default=None)
    parser.add_argument("--model", choices=sorted(DATASET_CHOICES), default=None)
    parser.add_argument("--dataset", choices=sorted(DATASET_CHOICES), default=None)
    args, _ = parser.parse_known_args(argv[1:])

    dataset = fixed_dataset or args.dataset or args.model or default_dataset
    if dataset is None:
        raise SystemExit(
            "Validation mode requires --dataset/--model or a fixed script dataset."
        )

    try:
        config = load_dataset_config(dataset, args.dataset_config_csv)
        resolved = validate_script_config(script_name, config)
        print_resolved_config_summary(script_name, config, resolved)
    except Exception as exc:
        print(f"VALIDATE-CONFIG-ONLY FAIL: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
    raise SystemExit(0)
