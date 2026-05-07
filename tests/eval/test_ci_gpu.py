"""Unit tests for the GPU-accelerated stats primitives.

Verifies:
    * pick_device() resolves correctly on the running machine
    * pick_device() hard-errors on an explicit unavailable backend
    * bootstrap_mean_ci(device='cpu') matches ci.bootstrap_ci(values, mean)
      bit-for-bit on point + CI endpoints (vectorized numpy reduces the
      same draw stream as the Python loop, given the same seed)
    * bootstrap_mean_ci with weights matches the explicit weighted mean
      lambda used in metrics_non_nested._per_field_row
    * paired_bootstrap_diff(device='cpu') matches ci.paired_bootstrap_diff
    * independent_bootstrap_diff(device='cpu') matches the cross_dataset
      hand-rolled loop
    * GPU vs CPU agreement within Monte-Carlo noise (skipped when no GPU)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from digital_registrar_research.benchmarks.eval import ci, ci_gpu


_HAS_TORCH: bool
try:  # pragma: no cover - environment probe
    import torch as _torch
    _HAS_TORCH = True
    _HAS_CUDA = bool(getattr(_torch, "cuda", None) and _torch.cuda.is_available())
    _HAS_MPS = bool(getattr(_torch.backends, "mps", None)
                    and _torch.backends.mps.is_available())
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    _HAS_CUDA = False
    _HAS_MPS = False


# --- pick_device ------------------------------------------------------------

def test_pick_device_auto_resolves() -> None:
    resolved = ci_gpu.pick_device("auto")
    assert resolved in ("cpu", "cuda", "mps")


def test_pick_device_none_resolves_like_auto() -> None:
    assert ci_gpu.pick_device(None) == ci_gpu.pick_device("auto")


def test_pick_device_cpu_explicit_returns_cpu() -> None:
    assert ci_gpu.pick_device("cpu") == "cpu"


def test_pick_device_invalid_raises() -> None:
    with pytest.raises(SystemExit):
        ci_gpu.pick_device("bogus")


@pytest.mark.skipif(_HAS_CUDA, reason="CUDA is available; cannot test the unavailable branch")
def test_pick_device_cuda_unavailable_hard_errors() -> None:
    if not _HAS_TORCH:
        pytest.skip("torch unavailable; pick_device cannot probe CUDA")
    with pytest.raises(SystemExit):
        ci_gpu.pick_device("cuda")


@pytest.mark.skipif(_HAS_MPS, reason="MPS is available; cannot test the unavailable branch")
def test_pick_device_mps_unavailable_hard_errors() -> None:
    if not _HAS_TORCH:
        pytest.skip("torch unavailable; pick_device cannot probe MPS")
    with pytest.raises(SystemExit):
        ci_gpu.pick_device("mps")


# --- bootstrap_mean_ci CPU equivalence with ci.py ---------------------------

def _binary_vec(seed: int, n: int, p_correct: float = 0.7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) < p_correct).astype(np.float64)


def test_bootstrap_mean_cpu_matches_ci_py_simple_mean() -> None:
    values = _binary_vec(42, n=200, p_correct=0.65)
    n_boot = 500
    seed = 0
    alpha = 0.05

    expected = ci.bootstrap_ci(
        list(values),
        lambda xs: float(np.mean(xs)),
        n_boot=n_boot, alpha=alpha, random_state=seed,
    )
    got = ci_gpu.bootstrap_mean_ci(
        values, n_boot=n_boot, alpha=alpha,
        random_state=seed, device="cpu",
    )

    # Point estimates must be exact.
    assert got.point == pytest.approx(expected.point, abs=1e-15)
    # CI endpoints should match to within float roundoff because:
    #   - Same RNG seed -> same draw stream
    #   - rng.integers(0,n,size=(B,n)) == B reshapes of rng.integers(0,n,size=n)
    #   - mean over axis=1 == mean over each gathered row
    #   - closed-form jackknife matches the python-loop jackknife identically
    assert got.lo == pytest.approx(expected.lo, abs=1e-12)
    assert got.hi == pytest.approx(expected.hi, abs=1e-12)
    # Method label preserved.
    assert got.method == expected.method


def test_bootstrap_mean_cpu_matches_ci_py_percentile() -> None:
    values = _binary_vec(7, n=120, p_correct=0.5)
    n_boot = 400
    seed = 11

    expected = ci.bootstrap_ci(
        list(values),
        lambda xs: float(np.mean(xs)),
        n_boot=n_boot, alpha=0.1, method="percentile", random_state=seed,
    )
    got = ci_gpu.bootstrap_mean_ci(
        values, n_boot=n_boot, alpha=0.1,
        method="percentile", random_state=seed, device="cpu",
    )
    assert got.point == pytest.approx(expected.point, abs=1e-15)
    assert got.lo == pytest.approx(expected.lo, abs=1e-12)
    assert got.hi == pytest.approx(expected.hi, abs=1e-12)
    assert got.method == "percentile"


def test_bootstrap_weighted_mean_cpu_matches_lambda() -> None:
    """The weighted-mean path must reproduce metrics_non_nested._per_field_row's
    `sum(c * a) / max(1, sum(a))` lambda statistic exactly.
    """
    rng = np.random.default_rng(123)
    n = 150
    correct = (rng.random(n) < 0.6).astype(np.float64)
    attempted = (rng.random(n) < 0.85).astype(np.float64)

    n_boot = 300
    seed = 5
    alpha = 0.05

    records = list(zip(range(n), correct.astype(int), attempted.astype(bool)))
    expected = ci.bootstrap_ci(
        records,
        lambda xs: (
            float(np.sum([c for _, c, a in xs if a])) /
            max(1, sum(1 for _, _, a in xs if a))
        ),
        n_boot=n_boot, alpha=alpha, random_state=seed,
    )
    got = ci_gpu.bootstrap_mean_ci(
        correct, weights=attempted,
        n_boot=n_boot, alpha=alpha,
        random_state=seed, device="cpu",
    )
    assert got.point == pytest.approx(expected.point, abs=1e-15)
    assert got.lo == pytest.approx(expected.lo, abs=1e-12)
    assert got.hi == pytest.approx(expected.hi, abs=1e-12)


def test_bootstrap_mean_empty_returns_nan() -> None:
    got = ci_gpu.bootstrap_mean_ci(
        np.array([], dtype=np.float64), n_boot=10, random_state=0, device="cpu",
    )
    assert np.isnan(got.point)
    assert np.isnan(got.lo)
    assert np.isnan(got.hi)


def test_bootstrap_mean_weights_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        ci_gpu.bootstrap_mean_ci(
            np.array([1.0, 0.0, 1.0]),
            weights=np.array([1.0, 1.0]),
            n_boot=10, device="cpu",
        )


# --- paired_bootstrap_diff equivalence --------------------------------------

def test_paired_bootstrap_diff_cpu_matches_ci_py() -> None:
    rng = np.random.default_rng(99)
    n = 80
    a = (rng.random(n) < 0.7).astype(np.float64)
    b = (rng.random(n) < 0.55).astype(np.float64)

    n_boot = 500
    seed = 3
    alpha = 0.05

    expected = ci.paired_bootstrap_diff(
        a, b, n_boot=n_boot, alpha=alpha, random_state=seed,
    )
    got = ci_gpu.paired_bootstrap_diff(
        a, b, n_boot=n_boot, alpha=alpha,
        random_state=seed, device="cpu",
    )
    assert got.point == pytest.approx(expected.point, abs=1e-15)
    assert got.lo == pytest.approx(expected.lo, abs=1e-12)
    assert got.hi == pytest.approx(expected.hi, abs=1e-12)


def test_paired_bootstrap_diff_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        ci_gpu.paired_bootstrap_diff(
            np.array([1.0, 0.0]), np.array([1.0, 0.0, 1.0]),
            n_boot=10, device="cpu",
        )


# --- independent_bootstrap_diff equivalence ---------------------------------

def test_independent_bootstrap_diff_matches_handrolled_loop() -> None:
    """Match the existing scripts/eval/cross_dataset/_per_field_delta loop."""
    rng_seed_data = np.random.default_rng(0)
    l = (rng_seed_data.random(140) < 0.7).astype(np.float64)
    r = (rng_seed_data.random(110) < 0.5).astype(np.float64)
    n_boot = 300
    seed = 17

    # Reference: the original cross_dataset loop.
    rng = np.random.default_rng(seed)
    min_n = min(l.size, r.size)
    ref_boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx_l = rng.integers(0, l.size, size=min_n)
        idx_r = rng.integers(0, r.size, size=min_n)
        ref_boot[i] = float(l[idx_l].mean() - r[idx_r].mean())
    ref_lo = float(np.quantile(ref_boot, 0.025))
    ref_hi = float(np.quantile(ref_boot, 0.975))

    got = ci_gpu.independent_bootstrap_diff(
        l, r, n_boot=n_boot, alpha=0.05,
        random_state=seed, device="cpu",
    )
    assert got.point == pytest.approx(float(l.mean() - r.mean()), abs=1e-15)
    assert got.lo == pytest.approx(ref_lo, abs=1e-12)
    assert got.hi == pytest.approx(ref_hi, abs=1e-12)


# --- bootstrap_kappa_ci -----------------------------------------------------

def _kappa_reference(a: list, b: list) -> float:
    """Reference Cohen's κ matching iaa.cohen_kappa unweighted formula."""
    if not a:
        return float("nan")
    n = len(a)
    cats = set(a) | set(b)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    p_e = 0.0
    from collections import Counter
    fa, fb = Counter(a), Counter(b)
    for c in cats:
        p_e += (fa[c] / n) * (fb[c] / n)
    if p_e >= 1.0:
        return float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def test_bootstrap_kappa_point_matches_iaa_cohen_kappa() -> None:
    """The point estimate must match iaa.cohen_kappa for the same labels."""
    rng = np.random.default_rng(0)
    a = [int(x) for x in rng.integers(0, 4, size=120)]
    b = [int(x) for x in rng.integers(0, 4, size=120)]
    got = ci_gpu.bootstrap_kappa_ci(a, b, n_boot=100, random_state=1, device="cpu")
    expected = _kappa_reference(a, b)
    assert got.point == pytest.approx(expected, abs=1e-12)


def test_bootstrap_kappa_handles_none_labels() -> None:
    """None values must encode as a distinct category (matching the
    safety-net behavior under metrics.normalize)."""
    a = [1, None, 1, None, 0, 0, 1, None]
    b = [1, None, 0, 1, 0, 0, 1, None]
    got = ci_gpu.bootstrap_kappa_ci(a, b, n_boot=50, random_state=0, device="cpu")
    expected = _kappa_reference(a, b)
    assert got.point == pytest.approx(expected, abs=1e-12)


def test_paired_kappa_delta_point_matches_difference() -> None:
    rng = np.random.default_rng(2)
    n = 80
    gold = [int(x) for x in rng.integers(0, 3, size=n)]
    with_v = [int(x) for x in rng.integers(0, 3, size=n)]
    without_v = [int(x) for x in rng.integers(0, 3, size=n)]
    got = ci_gpu.paired_kappa_delta_ci(
        gold, with_v, gold, without_v,
        n_boot=200, random_state=11, device="cpu",
    )
    k1 = _kappa_reference(gold, with_v)
    k2 = _kappa_reference(gold, without_v)
    assert got.point == pytest.approx(k1 - k2, abs=1e-12)
    # CI endpoints should bracket the point under a finite n_boot.
    assert got.lo <= got.point <= got.hi


def test_paired_kappa_delta_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        ci_gpu.paired_kappa_delta_ci(
            [1, 0], [1, 0], [1], [0],
            n_boot=10, device="cpu",
        )


def test_fleiss_kappa_batch_matches_singletons() -> None:
    """Batched output agrees with multirun.fleiss_kappa called per matrix."""
    from digital_registrar_research.benchmarks.eval.multirun import fleiss_kappa as ref

    rng = np.random.default_rng(7)
    matrices = [
        rng.integers(0, 3, size=(20, 4)).astype(float),
        rng.integers(0, 2, size=(50, 3)).astype(float),
        rng.integers(0, 5, size=(30, 5)).astype(float),
    ]
    expected = [ref(m) for m in matrices]
    got = ci_gpu.fleiss_kappa_batch(matrices)
    for i, e in enumerate(expected):
        if math.isnan(e):
            assert math.isnan(got[i])
        else:
            assert got[i] == pytest.approx(e, abs=1e-12)


# --- mcnemar_batch ----------------------------------------------------------

def test_mcnemar_batch_matches_singletons() -> None:
    """For an array of (b, c) tuples covering all three branches, batched
    output must agree with looped singleton ``ci.mcnemar_test`` calls.
    """
    cases = [
        (0, 0),     # trivial
        (1, 0),     # exact, k=0
        (3, 5),     # exact, k=3, n=8
        (12, 11),   # exact, k=11, n=23 (still <25)
        (15, 12),   # chi-square boundary (n=27)
        (50, 30),   # chi-square
        (100, 100), # chi-square equal
        (1000, 5),  # chi-square extreme
    ]
    b_arr = [c[0] for c in cases]
    c_arr = [c[1] for c in cases]

    expected = [ci.mcnemar_test(bb, cc) for bb, cc in cases]
    got = ci_gpu.mcnemar_batch(b_arr, c_arr, continuity=True, device="cpu")

    for i, exp in enumerate(expected):
        assert got["statistic"][i] == pytest.approx(exp["statistic"], abs=1e-12, rel=1e-12)
        assert got["p_value"][i] == pytest.approx(exp["p_value"], abs=1e-12, rel=1e-12)
        assert str(got["method"][i]) == exp["method"]
        assert int(got["b"][i]) == exp["b"]
        assert int(got["c"][i]) == exp["c"]


def test_mcnemar_batch_no_continuity() -> None:
    cases = [(50, 30), (100, 100), (1000, 5)]  # all chi-square
    b_arr = [c[0] for c in cases]
    c_arr = [c[1] for c in cases]
    expected = [ci.mcnemar_test(bb, cc, continuity=False) for bb, cc in cases]
    got = ci_gpu.mcnemar_batch(b_arr, c_arr, continuity=False)
    for i, exp in enumerate(expected):
        assert got["statistic"][i] == pytest.approx(exp["statistic"], rel=1e-12)
        assert got["p_value"][i] == pytest.approx(exp["p_value"], rel=1e-12)
        assert str(got["method"][i]) == "chi2"


def test_mcnemar_batch_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        ci_gpu.mcnemar_batch([1, 2], [1])


# --- GPU vs CPU agreement (skipped when no GPU) -----------------------------

_GPU_DEVICE = "mps" if _HAS_MPS else ("cuda" if _HAS_CUDA else None)


@pytest.mark.skipif(_GPU_DEVICE is None, reason="No GPU backend available")
def test_bootstrap_mean_gpu_matches_cpu_within_mc() -> None:
    values = _binary_vec(0, n=300, p_correct=0.62)
    # Use n_boot large enough that the CI endpoints are deterministic to
    # within ~2/sqrt(n_boot) of the cpu reference.
    n_boot = 4000
    seed = 1234

    cpu = ci_gpu.bootstrap_mean_ci(
        values, n_boot=n_boot, alpha=0.05,
        random_state=seed, device="cpu",
    )
    gpu = ci_gpu.bootstrap_mean_ci(
        values, n_boot=n_boot, alpha=0.05,
        random_state=seed, device=_GPU_DEVICE,
    )
    # Indices are CPU-drawn → distributions match exactly. GPU only does
    # the gather + reduction; expect float64 equality.
    assert gpu.point == pytest.approx(cpu.point, abs=1e-12)
    assert gpu.lo == pytest.approx(cpu.lo, abs=1e-9)
    assert gpu.hi == pytest.approx(cpu.hi, abs=1e-9)


@pytest.mark.skipif(_GPU_DEVICE is None, reason="No GPU backend available")
def test_paired_bootstrap_gpu_matches_cpu() -> None:
    rng = np.random.default_rng(555)
    n = 200
    a = (rng.random(n) < 0.65).astype(np.float64)
    b = (rng.random(n) < 0.55).astype(np.float64)

    cpu = ci_gpu.paired_bootstrap_diff(
        a, b, n_boot=2000, alpha=0.05,
        random_state=42, device="cpu",
    )
    gpu = ci_gpu.paired_bootstrap_diff(
        a, b, n_boot=2000, alpha=0.05,
        random_state=42, device=_GPU_DEVICE,
    )
    assert gpu.point == pytest.approx(cpu.point, abs=1e-12)
    assert gpu.lo == pytest.approx(cpu.lo, abs=1e-9)
    assert gpu.hi == pytest.approx(cpu.hi, abs=1e-9)
