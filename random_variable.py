from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
import scipy.stats
import scipy.stats._distn_infrastructure

import convolve


class FiniteDist(scipy.stats._distn_infrastructure.rv_sample):
    if TYPE_CHECKING:
        # make pylance happy
        def __new__(cls, *args, **kwargs) -> FiniteDist:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, pk) -> None:
        try:
            pk = np.asarray(pk)
        except Exception as e:
            raise ValueError("pk must be an array-like object") from e
        if pk.ndim != 1:
            raise ValueError("pk must be a 1-dimensional array")

        xk = np.arange(len(pk))

        super().__init__(values=(xk, pk))

    def __mul__(self, other) -> Self:
        if not isinstance(other, FiniteDist):
            return NotImplemented
        return self.__class__(np.convolve(self.pk, other.pk))

    def __rmul__(self, other) -> Self:
        return self.__mul__(other)

    def __pow__(self, other: int) -> Self:
        return self.__class__(convolve.convolve_fast_pow_no_recur(self.pk, other, "auto"))

    def __repr__(self) -> str:
        return f"{self.__module__}.{self.__class__.__name__}(pk={self.pk})"
