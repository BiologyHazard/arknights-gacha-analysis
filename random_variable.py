from __future__ import annotations

from typing import TYPE_CHECKING, Self, overload

import numpy as np
import scipy.stats
import scipy.stats._distn_infrastructure

import convolve


class FiniteDist(scipy.stats._distn_infrastructure.rv_sample):
    def __new__(cls, *args, **kwargs) -> FiniteDist:
        return super().__new__(cls)

    def __init__(self, pk=None, cdf=None) -> None:
        if cdf is not None and pk is not None:
            raise ValueError("Only one of pk or cdf must be provided")

        if cdf is None and pk is None:
            raise ValueError("Either pk or cdf must be provided")

        if cdf is not None:
            try:
                cdf = np.asarray(cdf)
            except Exception as e:
                raise ValueError("cdf must be an array-like object") from e
            if cdf.ndim != 1:
                raise ValueError("cdf must be a 1-dimensional array")
            # if not np.all(np.diff(cdf) >= np.finfo(float).eps):
            #     raise ValueError("cdf must be non-decreasing")
            if not np.isclose(cdf[-1], 1):
                raise ValueError("The last value of cdf must be close to 1")
            pk = np.clip(np.diff(cdf, prepend=0), 0, 1)

        else:
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

    # if TYPE_CHECKING:
    #     @overload
    #     def expect(self) -> np.float64:  # type: ignore
    #         ...
