from __future__ import annotations

import numbers
from collections.abc import Mapping
import random
from typing import Any

import numpy as np


def configure_deterministic_runtime(
    seed: int,
    *,
    torch_module: Any | None = None,
    seed_cuda: bool = True,
) -> None:
    """Seed supported RNGs and enable deterministic Torch behavior when supplied."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise TypeError("seed must be an integer, not a boolean or non-integer value.")
    seed_value = int(seed)
    if not 0 <= seed_value <= 2**32 - 1:
        raise ValueError("seed must be within NumPy's unsigned 32-bit seed range.")

    random.seed(seed_value)
    np.random.seed(seed_value)

    if torch_module is None:
        return

    manual_seed = getattr(torch_module, "manual_seed", None)
    if not callable(manual_seed):
        raise TypeError("torch_module must expose a callable manual_seed.")
    manual_seed(seed_value)

    cuda = getattr(torch_module, "cuda", None)
    cuda_is_available = getattr(cuda, "is_available", None)
    cuda_manual_seed_all = getattr(cuda, "manual_seed_all", None)
    if (
        seed_cuda
        and callable(cuda_is_available)
        and cuda_is_available()
        and callable(cuda_manual_seed_all)
    ):
        cuda_manual_seed_all(seed_value)

    use_deterministic_algorithms = getattr(
        torch_module,
        "use_deterministic_algorithms",
        None,
    )
    if callable(use_deterministic_algorithms):
        try:
            use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            use_deterministic_algorithms(True)

    backends = getattr(torch_module, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False


DEVICE_REQUEST_CHOICES = {"auto", "cpu", "cuda"}
MORTALITY_DEVICE_ENV_VAR = "CLINICAUSE_MORTALITY_DEVICE"


def normalize_device_request(value: object, *, field_name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in DEVICE_REQUEST_CHOICES:
        raise ValueError(
            f"{field_name} must be one of {sorted(DEVICE_REQUEST_CHOICES)}; "
            f"got {value!r}."
        )
    return normalized


def resolve_device_request(
    cli_value: object | None,
    configured_value: object | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    environment_variable: str = MORTALITY_DEVICE_ENV_VAR,
) -> tuple[str, str]:
    """Resolve CLI > environment > config > auto and return value plus evidence."""
    environment = environ if environ is not None else {}
    if cli_value is not None:
        raw_value = cli_value
        source = "cli"
    elif environment_variable in environment:
        raw_value = environment[environment_variable]
        source = f"environment:{environment_variable}"
    elif configured_value is not None:
        raw_value = configured_value
        source = "config:MORTALITY_DEVICE"
    else:
        raw_value = "auto"
        source = "default:auto"
    return normalize_device_request(raw_value, field_name="mortality device"), source


def select_torch_device(device_request: str, *, torch_module: Any) -> str:
    request = normalize_device_request(
        device_request,
        field_name="mortality device request",
    )
    if request == "cpu":
        return "cpu"
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    cuda_available = bool(callable(is_available) and is_available())
    if request == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was explicitly requested for mortality prediction, but Torch "
            "reports that CUDA is unavailable."
        )
    if request == "auto":
        return "cuda" if cuda_available else "cpu"
    return request


def resolve_routed_mortality_device(
    model_type: str,
    requested_device: str,
) -> tuple[str, str]:
    """Prevent a background mortality job from sharing CUDA with CausalPFN."""
    request = normalize_device_request(
        requested_device,
        field_name="routed mortality device",
    )
    if model_type == "CausalPFN":
        if request == "cuda":
            raise ValueError(
                "CausalPFN cannot overlap with a CUDA mortality job on the same "
                "routed GPU; use --mortality-device cpu."
            )
        if request == "auto":
            return "cpu", "causalpfn_overlap_guard"
    return request, "requested"
