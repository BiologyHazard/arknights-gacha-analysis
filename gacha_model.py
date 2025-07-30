from collections.abc import Sequence

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike, NDArray


from random_variable import FiniteDist


def get_prob(pity_6: int, pity_5: int) -> tuple[float, float, float, float]:
    prob_6: float = COND_PROB_6_STAR[pity_6]
    prob_5: float = max(COND_PROB_5_STAR[pity_5], 1 - prob_6)
    prob_4: float = max(PROB_4_STAR, 1 - prob_6 - prob_5)
    prob_3: float = max(PROB_3_STAR, 1 - prob_6 - prob_5 - prob_4)
    return prob_6, prob_5, prob_4, prob_3


# 设置6星概率递增表
COND_PROB_6_STAR = np.zeros(99)
COND_PROB_6_STAR[0:50] = 0.02
COND_PROB_6_STAR[50:99] = np.arange(1, 50) * 0.02 + 0.02

COND_PROB_5_STAR = np.zeros(40)
COND_PROB_5_STAR[:15] = 0.08
COND_PROB_5_STAR[15:20] = np.arange(1, 6) * 0.02 + 0.08
COND_PROB_5_STAR[20:40] = np.arange(1, 21) * 0.04 + 0.18

PROB_4_STAR = 0.5
PROB_3_STAR = 0.4


def cond_prob_to_dist(cond_prob) -> NDArray:
    """
    计算非齐次几何分布的首次成功次数分布列

    Parameters:
        `cond_prob`: 条件概率数组，`cond_prob[i]` 表示连续 `i` 次失败后下一次成功的概率。
                     `cond_prob[-1]` 必须是 1。

    Returns:
        `NDArray`: 分布列，`prob[i]` 表示首次成功发生在第 `i` 次试验的概率。
                   返回的数组比输入的数组长 1，且首项为 0。
    """

    if not np.isclose(cond_prob[-1], 1.0):
        raise ValueError("The last element of cond_prob must be 1.0")

    dist: NDArray = np.zeros(len(cond_prob) + 1)
    fail_prob = 1.0

    for i, prob in enumerate(cond_prob):
        dist[i + 1] = fail_prob * prob
        fail_prob *= (1 - prob)

    return dist


class PityModel:
    def __init__(self, cond_prob):
        self.cond_prob = cond_prob
        self.full_dist = FiniteDist(cond_prob_to_dist(cond_prob))

    def __call__(self, *, item_num: int = 1, item_pity: int = 0) -> list[FiniteDist]:
        first_success_dist = FiniteDist(cond_prob_to_dist(self.cond_prob[item_pity:]))
        dist = FiniteDist([1])
        ans = [dist]
        for i in range(item_num):
            if i == 0:
                dist *= first_success_dist
            else:
                dist *= self.full_dist
            ans.append(dist)
        return ans
