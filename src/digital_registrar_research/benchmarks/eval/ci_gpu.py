"""GPU-accelerated statistics primitives for the eval pipeline.

This module is the opt-in companion to :mod:`ci`. The original
``ci.py`` is preserved untouched as the canonical safety-net
implementation for arbitrary-statistic bootstraps; ``ci_gpu`` provides
specialized, vectorized primitives for the small set of statistics that
actually appear in the eval-pipeline hot loops:

    * mean of a 1-D vector (optionally weighted)
    * paired mean difference between two equal-length vectors
    * Cohen's-kappa bootstrap (Phase 3)
    * batched McNemar test (Phase 2)
    * batched Fleiss-kappa (Phase 3)

Design principles:
    * **Same return types as ci.py** -- :class:`ci.BootstrapResult` is
      reused, so call sites can swap implementations without touching
      downstream code.
    * **Lazy torch import** -- the module is cheap to import on a
      CPU-only machine. ``torch`` is only imported when ``device != 'cpu'``.
    * **Vectorized CPU path** -- ``device="cpu"`` (the default) uses
      fully vectorized numpy. No Python loops over ``n_boot``, no
      Python jackknife loop. With matched RNG seed it produces the
      same draws as ``ci.bootstrap_ci``'s Python loop, so for the mean
      statistic the bootstrap distribution is identical to within
      floating-point roundoff.
    * **Hard error on unavailable backend** -- requesting an
      unavailable backend (e.g. ``--device cuda`` on a Mac) raises
      :class:`SystemExit` rather than silently falling back to CPU.
      Silent fallback would mislead reported wall-times in this
      benchmarking codebase. ``device="auto"`` is the explicit
      "do whatever works" knob.

Public API:
    * :func:`pick_device`
    * :func:`bootstrap_mean_ci`
    * :func:`paired_bootstrap_diff`
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats as sstats

from .ci import BootstrapResult

__all__ = [
    "pick_device",
    "bootstrap_mean_ci",
    "paired_bootstrap_diff",
    "independent_bootstrap_diff",
    "mcnemar_batch",
    "bootstrap_kappa_ci",
    "paired_kappa_delta_ci",
    "fleiss_kappa_batch",
]


# --- Device resolution -------------------------------------------------------

_VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


def pick_device(requested: str | None = None) -> str:
    """Resolve ``'auto' | None | 'cpu' | 'cuda' | 'mps'`` to a concrete device.

    'auto' / None picks ``mps`` if available, else ``cuda``, else ``cpu``.
    Explicit ``cuda`` or ``mps`` raises :class:`SystemExit` if the
    backend is unavailable. Explicit ``cpu`` is always honored.

    Callers running on Apple Silicon should set
    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` near their entry point so the rare
    op MPS lacks falls back to CPU instead of erroring.
    """
    req = (requested or "auto").lower()
    if req not in _VALID_DEVICES:
        raise SystemExit(
            f"Invalid --device {requested!r}. Choose one of {_VALID_DEVICES}."
        )
    if req == "cpu":
        return "cpu"
    # Need torch for any non-CPU resolution.
    try:
        import torch
    except Exception as exc:
        if req == "auto":
            return "cpu"
        raise SystemExit(
            f"--device {req} requested but torch is not importable: {exc}"
        ) from exc

    has_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    has_mps = bool(getattr(torch.backends, "mps", None)
                   and torch.backends.mps.is_available())

    if req == "auto":
        if has_mps:
            return "mps"
        if has_cuda:
            return "cuda"
        return "cpu"
    if req == "cuda":
        if not has_cuda:
            raise SystemExit(
                "--device cuda requested but CUDA is not available on this "
                "build. Pick 'auto', 'mps', or 'cpu'."
            )
        return "cuda"
    if req == "mps":
        if not has_mps:
            raise SystemExit(
                "--device mps requested but MPS (Apple Silicon) is not "
                "available on this build. Pick 'auto', 'cuda', or 'cpu'."
            )
        return "mps"
    return "cpu"  # unreachable


# --- Internal helpers --------------------------------------------------------

def _quantile(a: np.ndarray, q: float) -> float:
    """Match ci.py's quantile semantics (linear method)."""
    return float(np.quantile(a, q, method="linear")) if a.size else float("nan")


def _draw_indices_numpy(
    rng: np.random.Generator,
    n: int,
    n_boot: int,
    strata: np.ndarray | None,
) -> np.ndarray:
    """Draw a (n_boot, n) integer index matrix.

    With ``strata=None``, draws are equivalent to ``n_boot`` consecutive
    calls to ``rng.integers(0, n, size=n)`` reshaped to ``(n_boot, n)``.
    This matches ci.bootstrap_ci's draw stream exactly.

    With strata, draws are stratified: each row of the output contains
    a per-stratum bootstrap of the indices belonging to that stratum.
    """
    if strata is None:
        return rng.integers(0, n, size=(n_boot, n))

    strata = np.asarray(strata)
    if strata.shape[0] != n:
        raise ValueError("strata length must match values length")
    unique = np.unique(strata)
    out = np.empty((n_boot, n), dtype=np.int64)
    write_col = 0
    for s in unique:
        idx = np.where(strata == s)[0]
        m = idx.size
        # Per-stratum draws, vectorized in one call to match the loop's stream.
        picks = rng.integers(0, m, size=(n_boot, m))
        out[:, write_col:write_col + m] = idx[picks]
        write_col += m
    return out


def _bca_endpoints(
    boot: np.ndarray,
    theta_hat: float,
    diff: np.ndarray,
    alpha: float,
) -> tuple[float, float, str]:
    """Compute BCa CI endpoints from bootstrap distribution + jackknife diffs.

    ``diff`` is ``jack_mean - jack[i]`` (one entry per input observation).
    Returns ``(lo, hi, method)`` where method is "bca" on success or
    "percentile_fallback" if the acceleration term blows up.
    """
    if boot.size < 2:
        return (theta_hat, theta_hat, "bca")

    frac_below = float(np.mean(boot < theta_hat))
    frac_below = min(
        max(frac_below, 1.0 / (boot.size + 1)),
        1.0 - 1.0 / (boot.size + 1),
    )
    z0 = float(sstats.norm.ppf(frac_below))

    num = float(np.nansum(diff ** 3))
    den = 6.0 * (float(np.nansum(diff ** 2)) ** 1.5)
    a_hat = num / den if den > 0 else 0.0

    z_lo = sstats.norm.ppf(alpha / 2)
    z_hi = sstats.norm.ppf(1 - alpha / 2)
    denom_lo = 1 - a_hat * (z0 + z_lo)
    denom_hi = 1 - a_hat * (z0 + z_hi)
    if denom_lo <= 0 or denom_hi <= 0:
        lo = _quantile(boot, alpha / 2)
        hi = _quantile(boot, 1 - alpha / 2)
        return (lo, hi, "percentile_fallback")

    alpha_lo = float(sstats.norm.cdf(z0 + (z0 + z_lo) / denom_lo))
    alpha_hi = float(sstats.norm.cdf(z0 + (z0 + z_hi) / denom_hi))
    lo = _quantile(boot, alpha_lo)
    hi = _quantile(boot, alpha_hi)
    return (lo, hi, "bca")


# --- Bootstrap of the (weighted) mean ----------------------------------------

def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: str = "bca",
    strata: Sequence | np.ndarray | None = None,
    random_state: int | None = 0,
    device: str = "cpu",
) -> BootstrapResult:
    """BCa / percentile bootstrap CI for the (weighted) mean of a 1-D vector.

    Parameters
    ----------
    values : 1-D numeric array. NaN entries are dropped before resampling
        (mirrors ci.py's exception-on-NaN-and-skip behavior in spirit).
    weights : optional 1-D non-negative array, same length as ``values``.
        When provided, the statistic is ``sum(values * weights) / sum(weights)``.
        Resamples with all-zero weight return ``0.0`` (matches the
        ``max(1, denom)`` guard at metrics_non_nested.py:268).
        When ``None``, the statistic is the simple mean.
    method : ``"bca"`` (default, accelerated & bias-corrected) or
        ``"percentile"``.
    strata : optional same-length array of stratum labels for stratified
        resampling (samples are drawn with replacement *within* each
        stratum).
    random_state : seeds ``np.random.default_rng``.
    device : ``"cpu" | "cuda" | "mps" | "auto"``. Default ``"cpu"`` uses
        vectorized numpy (no Python loops, no torch import). Other values
        use torch on the resolved device.

    Returns
    -------
    BootstrapResult
        Drop-in compatible with :class:`ci.BootstrapResult`.
    """
    values_arr = np.asarray(list(values) if not isinstance(values, np.ndarray)
                             else values, dtype=np.float64)
    if values_arr.ndim != 1:
        raise ValueError("values must be 1-D")
    n = values_arr.size

    if weights is None:
        weights_arr = None
    else:
        weights_arr = np.asarray(
            list(weights) if not isinstance(weights, np.ndarray) else weights,
            dtype=np.float64,
        )
        if weights_arr.shape != values_arr.shape:
            raise ValueError("weights must have same shape as values")

    # Drop entries with NaN values (and matching weights).
    mask = ~np.isnan(values_arr)
    if weights_arr is not None:
        mask &= ~np.isnan(weights_arr)
    if not mask.all():
        values_arr = values_arr[mask]
        if weights_arr is not None:
            weights_arr = weights_arr[mask]
        if strata is not None:
            strata = np.asarray(strata)[mask]
        n = values_arr.size

    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               np.array([]), method)

    # Point estimate.
    if weights_arr is None:
        theta_hat = float(values_arr.mean())
    else:
        sum_w = float(weights_arr.sum())
        theta_hat = float((values_arr * weights_arr).sum() / max(1.0, sum_w))

    resolved_device = pick_device(device) if device != "cpu" else "cpu"

    # Resample indices on CPU (RNG match with ci.py is the goal). The
    # GPU path only accelerates the per-row reduction over (n_boot, n).
    rng = np.random.default_rng(random_state)
    strata_arr = np.asarray(strata) if strata is not None else None
    indices = _draw_indices_numpy(rng, n, n_boot, strata_arr)

    if resolved_device == "cpu":
        boot = _resample_mean_cpu(values_arr, weights_arr, indices)
    else:
        boot = _resample_mean_torch(values_arr, weights_arr, indices,
                                    resolved_device)

    boot = boot[~np.isnan(boot)]
    if boot.size < 2:
        return BootstrapResult(theta_hat, theta_hat, theta_hat, boot, method)

    if method == "percentile":
        lo = _quantile(boot, alpha / 2)
        hi = _quantile(boot, 1 - alpha / 2)
        return BootstrapResult(theta_hat, lo, hi, boot, "percentile")

    # BCa: closed-form leave-one-out for (weighted) mean.
    if weights_arr is None:
        # jack[i] = (S - x_i) / (n - 1)
        S = float(values_arr.sum())
        if n < 2:
            return BootstrapResult(theta_hat, theta_hat, theta_hat, boot, method)
        jack = (S - values_arr) / (n - 1)
    else:
        S_vw = float((values_arr * weights_arr).sum())
        S_w = float(weights_arr.sum())
        denom = S_w - weights_arr
        denom = np.where(denom > 0, denom, 1.0)
        jack = (S_vw - values_arr * weights_arr) / denom
    jack_mean = float(np.nanmean(jack))
    diff = jack_mean - jack

    lo, hi, used = _bca_endpoints(boot, theta_hat, diff, alpha)
    return BootstrapResult(theta_hat, lo, hi, boot, used)


def _resample_mean_cpu(
    values: np.ndarray,
    weights: np.ndarray | None,
    indices: np.ndarray,
) -> np.ndarray:
    """Vectorized numpy: gather values (and weights) by ``indices`` and
    reduce per row. ``indices`` has shape ``(n_boot, n)``."""
    if weights is None:
        return values[indices].mean(axis=1)
    v = values[indices]            # (n_boot, n)
    w = weights[indices]           # (n_boot, n)
    sum_vw = (v * w).sum(axis=1)
    sum_w = w.sum(axis=1)
    sum_w_safe = np.where(sum_w > 0, sum_w, 1.0)
    return sum_vw / sum_w_safe


def _torch_float_dtype(device: str):
    """Choose tensor float dtype per device.

    GPU paths (cuda + mps) use **float32** by default because consumer /
    workstation GPUs (RTX 6000 Ada, RTX 30/40-series, A6000, Apple
    Silicon, …) have heavily throttled float64 throughput — typically
    1:32 to 1:64 of float32. Sticking with float64 there would erase
    the speedup we came here for. Data-center cards (A100, H100) do
    full-rate float64 but they're not the target environment for this
    codebase.

    Float32 precision (~7 decimal digits) is well above the natural
    Monte-Carlo noise floor of bootstrap CIs at typical ``n_boot=2000``
    (~2/sqrt(B) ≈ 4%); the bootstrap distribution is cast back to
    float64 on CPU before quantile / BCa so endpoint arithmetic
    remains full-precision.

    CPU stays at float64 for maximum precision and bit-equivalence
    with the safety-net :mod:`ci` path.
    """
    import torch
    return torch.float64 if device == "cpu" else torch.float32


def _resample_mean_torch(
    values: np.ndarray,
    weights: np.ndarray | None,
    indices: np.ndarray,
    device: str,
) -> np.ndarray:
    """Torch path. Same shape semantics as ``_resample_mean_cpu``.

    Indices are constructed on CPU (via numpy) so the bootstrap stream
    matches the safety-net implementation exactly; only the gather +
    reduction runs on-device. On MPS the per-row reduction runs in
    float32 (Metal lacks float64); the resulting bootstrap distribution
    is cast back to float64 on CPU before quantile / BCa, so CI
    endpoints retain full-precision arithmetic.
    """
    import torch

    dev = torch.device(device)
    fdtype = _torch_float_dtype(device)
    v_t = torch.as_tensor(values, dtype=fdtype, device=dev)
    idx_t = torch.as_tensor(indices, dtype=torch.long, device=dev)
    if weights is None:
        gathered = v_t[idx_t]                     # (n_boot, n)
        boot_t = gathered.mean(dim=1)
    else:
        w_t = torch.as_tensor(weights, dtype=fdtype, device=dev)
        v_g = v_t[idx_t]
        w_g = w_t[idx_t]
        sum_vw = (v_g * w_g).sum(dim=1)
        sum_w = w_g.sum(dim=1)
        sum_w_safe = torch.where(sum_w > 0, sum_w, torch.ones_like(sum_w))
        boot_t = sum_vw / sum_w_safe
    return boot_t.detach().cpu().numpy().astype(np.float64, copy=False)


# --- Paired bootstrap of mean(a) - mean(b) -----------------------------------

def paired_bootstrap_diff(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = 0,
    device: str = "cpu",
) -> BootstrapResult:
    """Paired bootstrap CI on ``mean(a) - mean(b)``.

    ``a`` and ``b`` must be the same length and represent paired
    observations (e.g. two methods scored on the same cases). NaN entries
    in either array drop the pair before resampling.

    Returns a :class:`ci.BootstrapResult` with method ``"percentile"`` —
    paired Δ is near-symmetric in practice, matching ci.paired_bootstrap_diff.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError("a and b must have identical shape")
    mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a = arr_a[mask]
    arr_b = arr_b[mask]
    n = arr_a.size
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               np.array([]), "percentile")

    delta_hat = float(arr_a.mean() - arr_b.mean())
    resolved_device = pick_device(device) if device != "cpu" else "cpu"

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, n, size=(n_boot, n))

    if resolved_device == "cpu":
        boot = arr_a[indices].mean(axis=1) - arr_b[indices].mean(axis=1)
    else:
        import torch
        dev = torch.device(resolved_device)
        fdtype = _torch_float_dtype(resolved_device)
        a_t = torch.as_tensor(arr_a, dtype=fdtype, device=dev)
        b_t = torch.as_tensor(arr_b, dtype=fdtype, device=dev)
        idx_t = torch.as_tensor(indices, dtype=torch.long, device=dev)
        boot_t = a_t[idx_t].mean(dim=1) - b_t[idx_t].mean(dim=1)
        boot = boot_t.detach().cpu().numpy().astype(np.float64, copy=False)

    lo = _quantile(boot, alpha / 2)
    hi = _quantile(boot, 1 - alpha / 2)
    return BootstrapResult(delta_hat, lo, hi, boot, "percentile")


# --- Batched McNemar test ----------------------------------------------------

def mcnemar_batch(
    b: Sequence[int] | np.ndarray,
    c: Sequence[int] | np.ndarray,
    *,
    continuity: bool = True,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Vectorized McNemar test on a batch of 2×2 paired tables.

    Replaces the per-row Python loop in
    :func:`...benchmarks.eval.completeness.method_pair_deltas` with a
    single vectorized call. ``b`` and ``c`` are equal-length 1-D arrays
    of paired-disagreement counts (b = "A=1, B=0" cell;
    c = "A=0, B=1" cell). Output keys mirror :func:`ci.mcnemar_test`:

        statistic : float array (k for exact branch, χ² for chi-square)
        p_value   : float array
        method    : object array of {"trivial", "exact_binomial",
                                     "chi2_cc", "chi2"}
        b, c      : int arrays (echoes of the input)

    Per-row behavior matches :func:`ci.mcnemar_test` exactly:
        n = b + c
        n == 0      -> trivial (statistic 0, p 1.0)
        0 < n < 25  -> two-sided exact binomial p = 2 * P(K <= min(b, c))
                       under K ~ Binomial(n, 0.5)
        n >= 25     -> chi-square approximation, with optional continuity

    The ``device`` argument is accepted for API symmetry but the math is
    closed-form element-wise arithmetic; the vectorized numpy + scipy
    path is already C-level fast and routing to torch yields negligible
    additional speedup for the realistic batch sizes (~10⁴ tests).
    The argument is preserved so callers can pass through ``args.device``
    uniformly without conditionals.
    """
    del device  # API symmetry; see docstring.
    b_arr = np.asarray(b, dtype=np.int64).ravel()
    c_arr = np.asarray(c, dtype=np.int64).ravel()
    if b_arr.shape != c_arr.shape:
        raise ValueError("b and c must have the same shape")

    n_arr = b_arr + c_arr
    size = b_arr.size

    statistic = np.zeros(size, dtype=np.float64)
    p_value = np.ones(size, dtype=np.float64)
    method_arr = np.empty(size, dtype=object)
    method_arr[:] = "trivial"

    # Exact two-sided binomial for small n.
    exact_mask = (n_arr > 0) & (n_arr < 25)
    if exact_mask.any():
        n_e = n_arr[exact_mask]
        k_e = np.minimum(b_arr[exact_mask], c_arr[exact_mask])
        # 2 * P(K <= k) under Binomial(n, 0.5), clipped at 1.
        p_e = 2.0 * sstats.binom.cdf(k_e, n_e, 0.5)
        p_e = np.clip(p_e, 0.0, 1.0)
        statistic[exact_mask] = k_e.astype(np.float64)
        p_value[exact_mask] = p_e
        method_arr[exact_mask] = "exact_binomial"

    # Chi-square approximation for n >= 25.
    chi_mask = n_arr >= 25
    if chi_mask.any():
        n_c = n_arr[chi_mask].astype(np.float64)
        b_c = b_arr[chi_mask].astype(np.float64)
        c_c = c_arr[chi_mask].astype(np.float64)
        if continuity:
            stat_c = (np.abs(b_c - c_c) - 1.0) ** 2 / n_c
            label = "chi2_cc"
        else:
            stat_c = (b_c - c_c) ** 2 / n_c
            label = "chi2"
        p_c = sstats.chi2.sf(stat_c, df=1)
        statistic[chi_mask] = stat_c
        p_value[chi_mask] = p_c
        method_arr[chi_mask] = label

    return {
        "statistic": statistic,
        "p_value": p_value,
        "method": method_arr,
        "b": b_arr,
        "c": c_arr,
    }


# --- Kappa primitives --------------------------------------------------------

def _encode_labels(*label_arrs: Sequence) -> tuple[list[np.ndarray], int]:
    """Encode an arbitrary set of label sequences into a shared 0..K-1 code
    space. ``None``/``NaN`` values get a dedicated code (last index) so they
    behave like a distinct category — matching the safety-net behavior where
    Python equality treats ``None == None`` as True.

    Returns ``([codes_a, codes_b, ...], n_categories)``.
    """
    cleaned: list[list[object]] = []
    universe: list[object] = []
    universe_idx: dict[object, int] = {}
    for arr in label_arrs:
        cur: list[object] = []
        for v in arr:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                key: object = "__NULL__"
            else:
                key = v
            if key not in universe_idx:
                universe_idx[key] = len(universe)
                universe.append(key)
            cur.append(key)
        cleaned.append(cur)
    K = len(universe) if universe else 1
    out = [np.array([universe_idx[v] for v in arr], dtype=np.int64)
           for arr in cleaned]
    return out, K


def _kappa_from_confusion(
    a_codes: np.ndarray, b_codes: np.ndarray, K: int,
) -> float:
    """Closed-form Cohen's κ on encoded label vectors. NaN if degenerate."""
    n = a_codes.size
    if n == 0:
        return float("nan")
    a_counts = np.bincount(a_codes, minlength=K).astype(np.float64)
    b_counts = np.bincount(b_codes, minlength=K).astype(np.float64)
    p_o = float(np.mean(a_codes == b_codes))
    p_e = float(np.sum(a_counts * b_counts)) / (n * n)
    if p_e >= 1.0:
        return float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def _vectorized_kappa(
    a_codes: np.ndarray,  # (n,)
    b_codes: np.ndarray,  # (n,)
    indices: np.ndarray,  # (n_boot, n)
    K: int,
    device: str,
) -> np.ndarray:
    """Compute Cohen's κ for each row of ``indices`` (resample). Returns
    a 1-D array of length n_boot."""
    n_boot, n = indices.shape

    if device == "cpu":
        a_g = a_codes[indices]   # (n_boot, n)
        b_g = b_codes[indices]   # (n_boot, n)
        # p_o per row
        p_o = (a_g == b_g).mean(axis=1)
        # Per-row marginals via flat bincount + reshape.
        # bincount-per-row trick: offset each row's labels into a unique
        # codespace, bincount once, reshape, slice.
        row_offset = (np.arange(n_boot) * K).reshape(-1, 1)
        a_offset = (a_g + row_offset).ravel()
        b_offset = (b_g + row_offset).ravel()
        a_counts = np.bincount(a_offset, minlength=n_boot * K).reshape(n_boot, K)
        b_counts = np.bincount(b_offset, minlength=n_boot * K).reshape(n_boot, K)
        a_counts = a_counts.astype(np.float64)
        b_counts = b_counts.astype(np.float64)
        p_e = (a_counts * b_counts).sum(axis=1) / (n * n)
        # κ; handle p_e == 1 (degenerate) -> NaN
        denom = 1.0 - p_e
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = np.where(denom > 0, (p_o - p_e) / denom, np.nan)
        return kappa

    import torch
    dev = torch.device(device)
    fdtype = _torch_float_dtype(device)
    a_t = torch.as_tensor(a_codes, dtype=torch.long, device=dev)
    b_t = torch.as_tensor(b_codes, dtype=torch.long, device=dev)
    idx_t = torch.as_tensor(indices, dtype=torch.long, device=dev)
    a_g = a_t[idx_t]   # (n_boot, n)
    b_g = b_t[idx_t]
    p_o = (a_g == b_g).to(fdtype).mean(dim=1)
    # Per-row category counts via scatter_add.
    one = torch.ones_like(a_g, dtype=fdtype)
    a_counts = torch.zeros((n_boot, K), dtype=fdtype, device=dev)
    a_counts.scatter_add_(1, a_g, one)
    b_counts = torch.zeros((n_boot, K), dtype=fdtype, device=dev)
    b_counts.scatter_add_(1, b_g, one)
    p_e = (a_counts * b_counts).sum(dim=1) / float(n * n)
    denom = 1.0 - p_e
    kappa = torch.where(denom > 0, (p_o - p_e) / denom,
                        torch.full_like(p_o, float("nan")))
    return kappa.detach().cpu().numpy().astype(np.float64, copy=False)


def bootstrap_kappa_ci(
    labels_a: Sequence | np.ndarray,
    labels_b: Sequence | np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    method: str = "bca",
    strata: Sequence | np.ndarray | None = None,
    random_state: int | None = 0,
    device: str = "cpu",
) -> BootstrapResult:
    """BCa / percentile bootstrap CI on UNWEIGHTED Cohen's κ for two
    same-length label vectors.

    Replaces the per-resample sklearn-cohen_kappa_score Python loop in
    :func:`...benchmarks.eval.iaa.pairwise_iaa` and
    :func:`...benchmarks.eval.preann.paired_delta_kappa` with one
    vectorized confusion-matrix accumulation across resamples.

    Notes
    -----
    * Quadratic-weighted κ is **not** handled here — those callers should
      keep using :func:`ci.bootstrap_ci` with the existing
      ``cohen_kappa(weights="quadratic")`` callable. (The IAA quadratic
      branch represents a small minority of κ call sites; the bulk are
      unweighted.)
    * NaN/None labels are encoded as a dedicated category — matching
      iaa.cohen_kappa's behavior under :func:`metrics.normalize`.
    """
    labels_a_list = list(labels_a)
    labels_b_list = list(labels_b)
    if len(labels_a_list) != len(labels_b_list):
        raise ValueError("labels_a and labels_b must have the same length")
    n = len(labels_a_list)
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               np.array([]), method)

    (a_codes, b_codes), K = _encode_labels(labels_a_list, labels_b_list)
    theta_hat = _kappa_from_confusion(a_codes, b_codes, K)
    resolved_device = pick_device(device) if device != "cpu" else "cpu"

    rng = np.random.default_rng(random_state)
    strata_arr = np.asarray(strata) if strata is not None else None
    indices = _draw_indices_numpy(rng, n, n_boot, strata_arr)

    boot = _vectorized_kappa(a_codes, b_codes, indices, K, resolved_device)
    boot = boot[~np.isnan(boot)]

    if boot.size < 2 or math.isnan(theta_hat):
        return BootstrapResult(theta_hat, theta_hat, theta_hat, boot, method)

    if method == "percentile":
        lo = _quantile(boot, alpha / 2)
        hi = _quantile(boot, 1 - alpha / 2)
        return BootstrapResult(theta_hat, lo, hi, boot, "percentile")

    # BCa needs jackknife. Closed-form leave-one-out for κ:
    # rebuild p_o, p_e from running totals minus the i-th observation.
    a_counts_full = np.bincount(a_codes, minlength=K).astype(np.float64)
    b_counts_full = np.bincount(b_codes, minlength=K).astype(np.float64)
    matches_full = float(np.sum(a_codes == b_codes))
    n_minus = n - 1
    if n_minus <= 0:
        return BootstrapResult(theta_hat, theta_hat, theta_hat, boot, method)
    # Per-i: subtract that obs from the running totals.
    a_counts_i = a_counts_full[a_codes]   # marginal count at this obs's a-cat
    b_counts_i = b_counts_full[b_codes]
    matches_i = (a_codes == b_codes).astype(np.float64)
    a_counts_jack = a_counts_full[None, :].repeat(n, axis=0)
    a_counts_jack[np.arange(n), a_codes] -= 1.0
    b_counts_jack = b_counts_full[None, :].repeat(n, axis=0)
    b_counts_jack[np.arange(n), b_codes] -= 1.0
    p_o_jack = (matches_full - matches_i) / n_minus
    p_e_jack = (a_counts_jack * b_counts_jack).sum(axis=1) / (n_minus * n_minus)
    denom_jack = 1.0 - p_e_jack
    with np.errstate(divide="ignore", invalid="ignore"):
        jack = np.where(denom_jack > 0,
                        (p_o_jack - p_e_jack) / denom_jack, np.nan)
    jack_mean = float(np.nanmean(jack))
    diff = jack_mean - jack

    lo, hi, used = _bca_endpoints(boot, theta_hat, diff, alpha)
    return BootstrapResult(theta_hat, lo, hi, boot, used)


# --- Paired bootstrap on (κ(a1, b1) - κ(a2, b2)) -----------------------------

def paired_kappa_delta_ci(
    labels_a1: Sequence,
    labels_b1: Sequence,
    labels_a2: Sequence,
    labels_b2: Sequence,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = 0,
    device: str = "cpu",
) -> BootstrapResult:
    """Bootstrap CI on κ(a1, b1) − κ(a2, b2) with paired case-level
    resampling. All four label sequences must be the same length and
    aligned on cases.

    Used by:
        * preann.paired_delta_kappa  → κ(gold, with) − κ(gold, without)
        * preann.disagreement_reduction → κ(a_with, b_with) − κ(a_without, b_without)
          (note: ``delta_disagreement = -delta_kappa``; caller flips sign.)

    Returns a percentile-CI :class:`BootstrapResult`.
    """
    a1 = list(labels_a1)
    b1 = list(labels_b1)
    a2 = list(labels_a2)
    b2 = list(labels_b2)
    if not (len(a1) == len(b1) == len(a2) == len(b2)):
        raise ValueError("all four label sequences must be the same length")
    n = len(a1)
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               np.array([]), "percentile")

    (a1_c, b1_c, a2_c, b2_c), K = _encode_labels(a1, b1, a2, b2)
    k1 = _kappa_from_confusion(a1_c, b1_c, K)
    k2 = _kappa_from_confusion(a2_c, b2_c, K)
    delta_hat = (k1 - k2) if (not math.isnan(k1) and not math.isnan(k2)) \
        else float("nan")

    resolved_device = pick_device(device) if device != "cpu" else "cpu"
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, n, size=(n_boot, n))

    boot1 = _vectorized_kappa(a1_c, b1_c, indices, K, resolved_device)
    boot2 = _vectorized_kappa(a2_c, b2_c, indices, K, resolved_device)
    boot = boot1 - boot2
    boot = boot[~np.isnan(boot)]

    if boot.size < 2 or math.isnan(delta_hat):
        return BootstrapResult(delta_hat, delta_hat, delta_hat, boot, "percentile")
    lo = _quantile(boot, alpha / 2)
    hi = _quantile(boot, 1 - alpha / 2)
    return BootstrapResult(delta_hat, lo, hi, boot, "percentile")


# --- Batched Fleiss kappa ----------------------------------------------------

def fleiss_kappa_batch(
    matrices: Sequence[np.ndarray],
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Vectorized Fleiss κ over a list of (n_subjects, n_raters) integer
    label matrices. NaN rows in any matrix are dropped per-matrix.
    Returns a 1-D float array of κ values aligned with ``matrices``.

    Each matrix is processed independently — sizes and category sets can
    differ. Per-matrix arithmetic is closed-form (no Python loop over
    n_boot, no scipy call), so the speedup over a Python ``for field:
    fleiss_kappa(...)`` loop comes primarily from amortizing Python
    overhead. ``device`` is accepted for API symmetry; the math is small
    enough that GPU dispatch yields no measurable benefit at the
    realistic batch size (~50-100 matrices).
    """
    del device  # closed-form arithmetic; vectorized numpy is plenty.
    out = np.empty(len(matrices), dtype=np.float64)
    for i, m in enumerate(matrices):
        out[i] = _fleiss_kappa_single(m)
    return out


def _fleiss_kappa_single(ratings: np.ndarray) -> float:
    """Single Fleiss κ, matching multirun.fleiss_kappa's semantics."""
    arr = np.asarray(ratings)
    if arr.ndim != 2:
        raise ValueError("ratings must be 2-D")
    # Drop rows with any NaN (matches multirun.fleiss_kappa).
    try:
        nan_mask = np.any(np.isnan(arr.astype(float, copy=False)), axis=1)
    except (TypeError, ValueError):
        nan_mask = np.zeros(arr.shape[0], dtype=bool)
    arr = arr[~nan_mask]
    N, k = arr.shape
    if N == 0 or k < 2:
        return float("nan")
    categories = np.unique(arr)
    n_ij = np.zeros((N, len(categories)), dtype=np.int64)
    for j, cat in enumerate(categories):
        n_ij[:, j] = (arr == cat).sum(axis=1)
    P_i = (np.sum(n_ij * n_ij, axis=1) - k) / (k * (k - 1))
    P_bar = float(P_i.mean())
    p_j = n_ij.sum(axis=0) / (N * k)
    P_e = float(np.sum(p_j * p_j))
    if P_e >= 1.0:
        return float("nan")
    return (P_bar - P_e) / (1.0 - P_e)


# --- Independent (unpaired) bootstrap of mean(a) - mean(b) -------------------

def independent_bootstrap_diff(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    size: int | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = 0,
    device: str = "cpu",
) -> BootstrapResult:
    """Bootstrap CI on ``mean(a) - mean(b)`` with INDEPENDENT resampling.

    Used for cross-dataset comparisons where ``a`` and ``b`` come from
    different patient cohorts and are NOT pairable (different lengths
    are allowed). ``size`` controls the resample length on each side
    (default ``min(len(a), len(b))``).

    To match the existing hand-rolled cross_dataset bootstrap stream, l
    and r index draws are interleaved per iteration via two passes — i.e.
    indices_l[b, :] and indices_r[b, :] are drawn for each ``b`` before
    moving on, the same order the Python loop would emit. With matched
    seed this means the bootstrap distribution is bit-equal to the
    pre-change implementation on a CPU device.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.ndim != 1 or arr_b.ndim != 1:
        raise ValueError("a and b must be 1-D")
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if arr_a.size == 0 or arr_b.size == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               np.array([]), "percentile")

    n_a = arr_a.size
    n_b = arr_b.size
    m = int(size) if size is not None else min(n_a, n_b)
    delta_hat = float(arr_a.mean() - arr_b.mean())
    resolved_device = pick_device(device) if device != "cpu" else "cpu"

    rng = np.random.default_rng(random_state)
    # Interleaved per-iteration draws to match the Python-loop stream:
    #     for i: rng.integers(0, n_a, size=m); rng.integers(0, n_b, size=m).
    # Two pre-allocated index matrices, filled row-by-row in one go via
    # a flat draw cycle. This is not vectorized but is O(n_boot) calls
    # vs O(n_boot) in the original; the GPU win still comes from the
    # gather + reduction below.
    idx_l = np.empty((n_boot, m), dtype=np.int64)
    idx_r = np.empty((n_boot, m), dtype=np.int64)
    for i in range(n_boot):
        idx_l[i] = rng.integers(0, n_a, size=m)
        idx_r[i] = rng.integers(0, n_b, size=m)

    if resolved_device == "cpu":
        boot = arr_a[idx_l].mean(axis=1) - arr_b[idx_r].mean(axis=1)
    else:
        import torch
        dev = torch.device(resolved_device)
        fdtype = _torch_float_dtype(resolved_device)
        a_t = torch.as_tensor(arr_a, dtype=fdtype, device=dev)
        b_t = torch.as_tensor(arr_b, dtype=fdtype, device=dev)
        il = torch.as_tensor(idx_l, dtype=torch.long, device=dev)
        ir = torch.as_tensor(idx_r, dtype=torch.long, device=dev)
        boot_t = a_t[il].mean(dim=1) - b_t[ir].mean(dim=1)
        boot = boot_t.detach().cpu().numpy().astype(np.float64, copy=False)

    lo = _quantile(boot, alpha / 2)
    hi = _quantile(boot, 1 - alpha / 2)
    return BootstrapResult(delta_hat, lo, hi, boot, "percentile")
