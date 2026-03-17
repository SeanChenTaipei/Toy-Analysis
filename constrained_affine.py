"""
Vectorized affine blocks with mixed per-feature parameter constraints.

Example:
    >>> import torch
    >>> block = ParallelConstrainedAffineBlock(
    ...     n_features=3,
    ...     a_lower=[0.0, None, -1.0],
    ...     a_upper=[1.0, None, None],
    ...     init_A=[0.25, 0.0, 0.5],
    ... )
    >>> x = torch.tensor([[1.0, 2.0, 3.0]])
    >>> out = block(x, return_contrib=True)
    >>> tuple(out["y"].shape)
    (1, 1)
    >>> tuple(out["contrib"].shape)
    (1, 3)
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Stable inverse softplus for strictly positive targets."""
    eps = 1e-12
    y = torch.clamp(y, min=eps)
    return torch.where(y > 20.0, y, torch.log(torch.expm1(y)))


def _inv_sigmoid(z: torch.Tensor) -> torch.Tensor:
    """Stable inverse sigmoid for values in (0, 1)."""
    eps = 1e-6
    z = z.clamp(eps, 1.0 - eps)
    return torch.log(z / (1.0 - z))


def _build_bound_tensors(
    lower: Sequence[Optional[float]],
    upper: Sequence[Optional[float]],
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    if len(lower) != len(upper):
        raise ValueError("lower and upper must have the same length")

    n = len(lower)
    lower_val = torch.zeros(n, dtype=dtype)
    upper_val = torch.zeros(n, dtype=dtype)
    has_lower = torch.zeros(n, dtype=torch.bool)
    has_upper = torch.zeros(n, dtype=torch.bool)

    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if lo is not None:
            lower_val[i] = float(lo)
            has_lower[i] = True
        if hi is not None:
            upper_val[i] = float(hi)
            has_upper[i] = True
        if lo is not None and hi is not None and hi < lo:
            raise ValueError(f"Invalid bounds at index {i}: upper < lower")

    fixed = has_lower & has_upper & torch.isclose(lower_val, upper_val)
    both = has_lower & has_upper & (~fixed)
    lower_only = has_lower & (~has_upper)
    upper_only = (~has_lower) & has_upper
    unbounded = (~has_lower) & (~has_upper)

    return {
        "lower_val": lower_val,
        "upper_val": upper_val,
        "fixed": fixed,
        "both": both,
        "lower_only": lower_only,
        "upper_only": upper_only,
        "unbounded": unbounded,
    }


class VectorizedBoundedParameter(nn.Module):
    """
    Maps a single raw parameter vector into values that satisfy mixed bounds.

    Supported cases can coexist in the same vector:
    - unbounded
    - lower-only
    - upper-only
    - both-bounded
    - fixed-value
    """

    def __init__(
        self,
        lower: Sequence[Optional[float]],
        upper: Sequence[Optional[float]],
        init_value: Optional[Sequence[float]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if len(lower) != len(upper):
            raise ValueError("lower and upper must have the same length")

        self.n = len(lower)
        info = _build_bound_tensors(lower, upper, dtype=dtype)

        self.register_buffer("lower_val", info["lower_val"])
        self.register_buffer("upper_val", info["upper_val"])
        self.register_buffer("idx_fixed", torch.where(info["fixed"])[0])
        self.register_buffer("idx_both", torch.where(info["both"])[0])
        self.register_buffer("idx_lower_only", torch.where(info["lower_only"])[0])
        self.register_buffer("idx_upper_only", torch.where(info["upper_only"])[0])
        self.register_buffer("idx_unbounded", torch.where(info["unbounded"])[0])

        init = self._build_init(init_value, dtype)
        self.raw = nn.Parameter(self._to_raw(init))

    def _build_init(
        self,
        init_value: Optional[Sequence[float]],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if init_value is not None:
            init = torch.as_tensor(init_value, dtype=dtype)
            if init.ndim != 1 or init.shape[0] != self.n:
                raise ValueError("init_value length mismatch")
            self._validate_init(init)
            return init

        init = torch.zeros(self.n, dtype=dtype)

        if self.idx_fixed.numel() > 0:
            init[self.idx_fixed] = self.lower_val[self.idx_fixed]
        if self.idx_both.numel() > 0:
            lo = self.lower_val[self.idx_both]
            hi = self.upper_val[self.idx_both]
            init[self.idx_both] = 0.5 * (lo + hi)
        if self.idx_lower_only.numel() > 0:
            lo = self.lower_val[self.idx_lower_only]
            init[self.idx_lower_only] = lo + 1.0
        if self.idx_upper_only.numel() > 0:
            hi = self.upper_val[self.idx_upper_only]
            init[self.idx_upper_only] = hi - 1.0
        if self.idx_unbounded.numel() > 0:
            init[self.idx_unbounded] = 0.0

        return init

    def _validate_init(self, init: torch.Tensor) -> None:
        if self.idx_fixed.numel() > 0:
            expected = self.lower_val[self.idx_fixed]
            if not torch.allclose(init[self.idx_fixed], expected):
                raise ValueError("Fixed-value init entries must equal their bound")
        if self.idx_both.numel() > 0:
            idx = self.idx_both
            if not torch.all((init[idx] >= self.lower_val[idx]) & (init[idx] <= self.upper_val[idx])):
                raise ValueError("Both-bounded init entries must lie within [lower, upper]")
        if self.idx_lower_only.numel() > 0:
            idx = self.idx_lower_only
            if not torch.all(init[idx] > self.lower_val[idx]):
                raise ValueError("Lower-only init entries must be strictly above the lower bound")
        if self.idx_upper_only.numel() > 0:
            idx = self.idx_upper_only
            if not torch.all(init[idx] < self.upper_val[idx]):
                raise ValueError("Upper-only init entries must be strictly below the upper bound")

    def _to_raw(self, init: torch.Tensor) -> torch.Tensor:
        raw = torch.zeros_like(init)

        if self.idx_both.numel() > 0:
            idx = self.idx_both
            lo = self.lower_val[idx]
            hi = self.upper_val[idx]
            z = (init[idx] - lo) / (hi - lo)
            raw[idx] = _inv_sigmoid(z)

        if self.idx_lower_only.numel() > 0:
            idx = self.idx_lower_only
            raw[idx] = _inv_softplus(init[idx] - self.lower_val[idx])

        if self.idx_upper_only.numel() > 0:
            idx = self.idx_upper_only
            raw[idx] = _inv_softplus(self.upper_val[idx] - init[idx])

        if self.idx_unbounded.numel() > 0:
            raw[self.idx_unbounded] = init[self.idx_unbounded]

        return raw

    def forward(self) -> torch.Tensor:
        """Return the constrained parameter vector with shape [n_features]."""
        value = torch.empty_like(self.raw)

        if self.idx_fixed.numel() > 0:
            value[self.idx_fixed] = self.lower_val[self.idx_fixed]
        if self.idx_both.numel() > 0:
            idx = self.idx_both
            lo = self.lower_val[idx]
            hi = self.upper_val[idx]
            value[idx] = lo + (hi - lo) * torch.sigmoid(self.raw[idx])
        if self.idx_lower_only.numel() > 0:
            idx = self.idx_lower_only
            value[idx] = self.lower_val[idx] + F.softplus(self.raw[idx])
        if self.idx_upper_only.numel() > 0:
            idx = self.idx_upper_only
            value[idx] = self.upper_val[idx] - F.softplus(self.raw[idx])
        if self.idx_unbounded.numel() > 0:
            value[self.idx_unbounded] = self.raw[self.idx_unbounded]

        return value


class ParallelConstrainedAffineBlock(nn.Module):
    """
    Parallel affine block for x:[batch, F].

    Computes:
        contrib = x * A + B
        y = contrib.sum(dim=1, keepdim=True) + global_bias

    Notes:
        - The forward pass is vectorized across all features.
        - Per-feature Python loops are only used during initialization.
        - A feature bias vector B is optional. In many interpretable models,
          using only A plus a global bias is easier to identify.
    """

    def __init__(
        self,
        n_features: int,
        a_lower: Sequence[Optional[float]],
        a_upper: Sequence[Optional[float]],
        init_A: Optional[Sequence[float]] = None,
        use_feature_bias: bool = False,
        b_lower: Optional[Sequence[Optional[float]]] = None,
        b_upper: Optional[Sequence[Optional[float]]] = None,
        init_B: Optional[Sequence[float]] = None,
        use_global_bias_if_no_feature_bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if len(a_lower) != n_features or len(a_upper) != n_features:
            raise ValueError("A bound lengths must match n_features")

        self.n_features = n_features
        self.use_feature_bias = use_feature_bias

        self.A_param = VectorizedBoundedParameter(
            lower=a_lower,
            upper=a_upper,
            init_value=init_A,
            dtype=dtype,
        )

        if use_feature_bias:
            b_lower = [None] * n_features if b_lower is None else list(b_lower)
            b_upper = [None] * n_features if b_upper is None else list(b_upper)
            if len(b_lower) != n_features or len(b_upper) != n_features:
                raise ValueError("B bound lengths must match n_features")

            self.B_param: Optional[VectorizedBoundedParameter] = VectorizedBoundedParameter(
                lower=b_lower,
                upper=b_upper,
                init_value=init_B,
                dtype=dtype,
            )
            self.global_bias = None
        else:
            self.B_param = None
            self.global_bias = (
                nn.Parameter(torch.zeros(1, dtype=dtype))
                if use_global_bias_if_no_feature_bias
                else None
            )

    def get_A(self) -> torch.Tensor:
        """Return the current constrained slope vector A."""
        return self.A_param()

    def get_B(self) -> Optional[torch.Tensor]:
        """Return the constrained feature-bias vector B, if enabled."""
        return None if self.B_param is None else self.B_param()

    def forward(
        self,
        x: torch.Tensor,
        return_contrib: bool = False,
    ) -> torch.Tensor | dict[str, Optional[torch.Tensor]]:
        """
        Apply the constrained affine block to a rank-2 input tensor.

        Args:
            x: Input tensor with shape [batch, n_features].
            return_contrib: When True, return y, contrib, A, and B.

        Returns:
            Either:
            - y with shape [batch, 1]
            - or a dict containing:
              y: [batch, 1]
              contrib: [batch, n_features]
              A: [n_features]
              B: [n_features] or None
        """
        if x.ndim != 2:
            raise ValueError("x must be rank-2 tensor [batch, n_features]")
        if x.shape[1] != self.n_features:
            raise ValueError(f"x.shape[1] must be {self.n_features}")

        A = self.get_A()
        contrib = x * A

        B = self.get_B()
        if B is not None:
            contrib = contrib + B

        y = contrib.sum(dim=1, keepdim=True)
        if self.global_bias is not None:
            y = y + self.global_bias

        if return_contrib:
            return {"y": y, "contrib": contrib, "A": A, "B": B}

        return y
