"""自卷积相关函数"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.fft
import scipy.signal


def convolve_n_times(arr, n, method):
    """朴素方法，直接卷积 `n-1` 次"""
    result = arr.copy()
    for _ in range(n - 1):
        result = scipy.signal.convolve(result, arr, method=method)
    return result


def convolve_fft(arr, n):
    """使用 FFT 方法进行卷积"""
    length = n * (len(arr) - 1) + 1
    padded_arr = np.zeros(length, dtype=arr.dtype)
    padded_arr[:len(arr)] = arr
    fft_arr = scipy.fft.fft(padded_arr)
    fft_result = fft_arr ** n
    result = scipy.fft.ifft(fft_result)
    return result.real  # type: ignore


def convolve_fast_pow_recursion(arr, n, method):
    """快速幂递归方法"""
    @lru_cache
    def inner(n, method):
        if n == 0:
            return np.ones(1)
        if n == 1:
            return arr

        n1 = n // 2
        n2 = n - n1
        a1 = inner(n1, method)
        a2 = inner(n2, method)
        return scipy.signal.convolve(a1, a2, method=method)

    return inner(n, method)


def convolve_fast_pow_no_recur(arr, n, method):
    """快速幂非递归方法"""
    ans = np.ones(1)
    temp = arr.copy()
    while n > 0:
        if n % 2 == 1:
            ans = scipy.signal.convolve(ans, temp, method=method)
        n = n // 2
        if n > 0:
            temp = scipy.signal.convolve(temp, temp, method=method)
    return ans


if __name__ == "__main__":
    import timeit

    from matplotlib import pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Source Han Sans SC"]

    def 区间最大模(arr, width: int = 8):
        """计算 arr 的滑动窗口 [i-width, i+width] 最大值"""
        x = np.arange(len(arr))
        y = np.arange(-width, width + 1)
        X, Y = np.meshgrid(x, y)
        indices = np.clip(X + Y, 0, len(arr) - 1)
        max_values = np.max(np.abs(arr[indices]), axis=0)
        return max_values

    a = np.random.rand(512)
    a /= np.sum(a)
    n = 32
    r_n_direct = convolve_n_times(a, n, "direct")
    r_n_fft = convolve_n_times(a, n, "fft")
    r_n_auto = convolve_n_times(a, n, "auto")
    r_fft = convolve_fft(a, n)
    r_fast_pow_no_recur_direct = convolve_fast_pow_no_recur(a, n, "direct")
    r_fast_pow_no_recur_fft = convolve_fast_pow_no_recur(a, n, "fft")
    r_fast_pow_no_recur_auto = convolve_fast_pow_no_recur(a, n, "auto")
    r_fast_pow_recursion_direct = convolve_fast_pow_recursion(a, n, "direct")
    r_fast_pow_recursion_fft = convolve_fast_pow_recursion(a, n, "fft")
    r_fast_pow_recursion_auto = convolve_fast_pow_recursion(a, n, "auto")
    plt.plot(区间最大模(r_n_direct), label="卷积 n-1 次 (direct)")
    plt.plot(区间最大模(r_n_fft), label="卷积 n-1 次 (fft)")
    plt.plot(区间最大模(r_n_auto), label="卷积 n-1 次 (auto)")
    plt.plot(区间最大模(r_fft), label="统一 fft")
    plt.plot(区间最大模(r_fast_pow_no_recur_direct), label="快速幂非递归 (direct)")
    plt.plot(区间最大模(r_fast_pow_no_recur_fft), label="快速幂非递归 (fft)")
    plt.plot(区间最大模(r_fast_pow_no_recur_auto), label="快速幂非递归 (auto)")
    plt.plot(区间最大模(r_fast_pow_recursion_direct), label="快速幂递归 (direct)")
    plt.plot(区间最大模(r_fast_pow_recursion_fft), label="快速幂递归 (fft)")
    plt.plot(区间最大模(r_fast_pow_recursion_auto), label="快速幂递归 (auto)")
    plt.yscale("log")
    plt.title(f"len(a) = {len(a)}, n = {n}")
    plt.legend()
    plt.show()

    print(f"len(a) = {len(a)}, n = {n}")
    t_n_direct = timeit.timeit(lambda: convolve_n_times(a, n, "direct"), number=1)
    print(f"卷积 n-1 次 (direct): {t_n_direct:.6f}s")
    t_n_fft = timeit.timeit(lambda: convolve_n_times(a, n, "fft"), number=1)
    print(f"卷积 n-1 次 (fft): {t_n_fft:.6f}s")
    t_n_auto = timeit.timeit(lambda: convolve_n_times(a, n, "auto"), number=1)
    print(f"卷积 n-1 次 (auto): {t_n_auto:.6f}s")
    t_fft = timeit.timeit(lambda: convolve_fft(a, n), number=1)
    print(f"统一 fft: {t_fft:.6f}s")
    t_fast_pow_no_recur_direct = timeit.timeit(lambda: convolve_fast_pow_no_recur(a, n, "direct"), number=1)
    print(f"快速幂非递归 (direct): {t_fast_pow_no_recur_direct:.6f}s")
    t_fast_pow_no_recur_fft = timeit.timeit(lambda: convolve_fast_pow_no_recur(a, n, "fft"), number=1)
    print(f"快速幂非递归 (fft): {t_fast_pow_no_recur_fft:.6f}s")
    t_fast_pow_no_recur_auto = timeit.timeit(lambda: convolve_fast_pow_no_recur(a, n, "auto"), number=1)
    print(f"快速幂非递归 (auto): {t_fast_pow_no_recur_auto:.6f}s")
    t_fast_pow_recursion_direct = timeit.timeit(lambda: convolve_fast_pow_recursion(a, n, "direct"), number=1)
    print(f"快速幂递归 (direct): {t_fast_pow_recursion_direct:.6f}s")
    t_fast_pow_recursion_fft = timeit.timeit(lambda: convolve_fast_pow_recursion(a, n, "fft"), number=1)
    print(f"快速幂递归 (fft): {t_fast_pow_recursion_fft:.6f}s")
    t_fast_pow_recursion_auto = timeit.timeit(lambda: convolve_fast_pow_recursion(a, n, "auto"), number=1)
    print(f"快速幂递归 (auto): {t_fast_pow_recursion_auto:.6f}s")
