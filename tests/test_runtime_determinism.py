from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runtime_determinism import configure_deterministic_runtime  # noqa: E402
from runtime_determinism import resolve_device_request  # noqa: E402
from runtime_determinism import resolve_routed_mortality_device  # noqa: E402
from runtime_determinism import select_torch_device  # noqa: E402


class _FakeCuda:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    @staticmethod
    def is_available() -> bool:
        return True

    def manual_seed_all(self, seed: int) -> None:
        self.seeds.append(seed)


class _FakeTorch:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.deterministic_calls: list[tuple[bool, bool]] = []
        self.cuda = _FakeCuda()
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=True)
        )

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def use_deterministic_algorithms(
        self,
        enabled: bool,
        *,
        warn_only: bool,
    ) -> None:
        self.deterministic_calls.append((enabled, warn_only))


def test_deterministic_runtime_repeats_python_and_numpy_sequences() -> None:
    configure_deterministic_runtime(0)
    first = (random.random(), float(np.random.random()))

    configure_deterministic_runtime(0)
    second = (random.random(), float(np.random.random()))

    assert first == second


def test_deterministic_runtime_configures_torch_and_cuda() -> None:
    fake_torch = _FakeTorch()

    configure_deterministic_runtime(7, torch_module=fake_torch)

    assert fake_torch.seeds == [7]
    assert fake_torch.cuda.seeds == [7]
    assert fake_torch.deterministic_calls == [(True, True)]
    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cudnn.benchmark is False


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_deterministic_runtime_rejects_invalid_seeds(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        configure_deterministic_runtime(seed)  # type: ignore[arg-type]


def test_device_request_precedence_and_evidence() -> None:
    assert resolve_device_request(
        "cpu",
        "cuda",
        environ={"CLINICAUSE_MORTALITY_DEVICE": "cuda"},
    ) == ("cpu", "cli")
    assert resolve_device_request(
        None,
        "cpu",
        environ={"CLINICAUSE_MORTALITY_DEVICE": "cuda"},
    ) == ("cuda", "environment:CLINICAUSE_MORTALITY_DEVICE")
    assert resolve_device_request(None, "cpu", environ={}) == (
        "cpu",
        "config:MORTALITY_DEVICE",
    )
    assert resolve_device_request(None, None, environ={}) == (
        "auto",
        "default:auto",
    )


def test_torch_device_selection_and_explicit_cuda_failure() -> None:
    fake_torch = _FakeTorch()
    assert select_torch_device("auto", torch_module=fake_torch) == "cuda"
    assert select_torch_device("cpu", torch_module=fake_torch) == "cpu"

    fake_torch.cuda.is_available = lambda: False
    assert select_torch_device("auto", torch_module=fake_torch) == "cpu"
    with pytest.raises(RuntimeError, match="explicitly requested"):
        select_torch_device("cuda", torch_module=fake_torch)


def test_cpu_deterministic_setup_does_not_seed_cuda() -> None:
    fake_torch = _FakeTorch()
    configure_deterministic_runtime(
        9,
        torch_module=fake_torch,
        seed_cuda=False,
    )
    assert fake_torch.seeds == [9]
    assert fake_torch.cuda.seeds == []


def test_causalpfn_route_forces_cpu_and_rejects_cuda_overlap() -> None:
    assert resolve_routed_mortality_device("CausalPFN", "auto") == (
        "cpu",
        "causalpfn_overlap_guard",
    )
    assert resolve_routed_mortality_device("CausalPFN", "cpu") == (
        "cpu",
        "requested",
    )
    with pytest.raises(ValueError, match="cannot overlap"):
        resolve_routed_mortality_device("CausalPFN", "cuda")
    assert resolve_routed_mortality_device("CausalForest", "auto") == (
        "auto",
        "requested",
    )
