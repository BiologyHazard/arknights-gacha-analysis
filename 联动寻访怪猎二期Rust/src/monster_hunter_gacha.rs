#![allow(nonstandard_style)]

use std::path::Path;

use crate::io;
use crate::matrix::CooArray;

pub const 六星水位上限: u8 = 98;
pub const 五星水位上限: u8 = 39;
pub const UP六星水位上限: u8 = 119;
pub const 已获取UP六星干员数量上限: u8 = 6;
pub const 已获取UP五星干员数量上限: u8 = 6;
pub const 迭代次数: u16 = 900;

/// 状态总数 = 99 * 40 * 120 * 7 * 7 = 23284800
pub const 状态数量: u32 = (六星水位上限 as u32 + 1)
    * (五星水位上限 as u32 + 1)
    * (UP六星水位上限 as u32 + 1)
    * (已获取UP六星干员数量上限 as u32 + 1)
    * (已获取UP五星干员数量上限 as u32 + 1);

pub const COND_PROB_6_STAR: [f64; 六星水位上限 as usize + 1] = {
    let mut arr = [0.0; 六星水位上限 as usize + 1];
    let mut 水位 = 0;
    while 水位 <= 六星水位上限 {
        arr[水位 as usize] = if 水位 < 50 {
            0.02
        } else {
            (水位 - 49) as f64 * 0.02 + 0.02
        };
        水位 += 1;
    }
    arr
};

pub const COND_PROB_5_STAR: [f64; 五星水位上限 as usize + 1] = {
    let mut arr = [0.0; 五星水位上限 as usize + 1];
    let mut 水位 = 0;
    while 水位 <= 五星水位上限 {
        arr[水位 as usize] = if 水位 < 15 {
            0.08
        } else if 水位 < 20 {
            (水位 - 14) as f64 * 0.02 + 0.08
        } else {
            (水位 - 19) as f64 * 0.04 + 0.18
        };
        水位 += 1;
    }
    arr
};

#[inline]
pub fn 获取状态索引(
    六星水位: u8,
    五星水位: u8,
    UP六星水位: u8,
    已获取UP六星干员数量: u8,
    已获取UP五星干员数量: u8,
) -> u32 {
    (((六星水位 as u32 * (五星水位上限 as u32 + 1) + 五星水位 as u32)
        * (UP六星水位上限 as u32 + 1)
        + UP六星水位 as u32)
        * (已获取UP六星干员数量上限 as u32 + 1)
        + (已获取UP六星干员数量 as u32).min(已获取UP六星干员数量上限 as u32))
        * (已获取UP五星干员数量上限 as u32 + 1)
        + (已获取UP五星干员数量 as u32).min(已获取UP五星干员数量上限 as u32)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum 矩阵类型枚举 {
    第10抽,
    前50抽但非第10抽,
    第51抽及以后,
}

pub fn 状态转移(
    起始六星水位: u8,
    起始五星水位: u8,
    起始UP六星水位: u8,
    起始已获取UP六星干员数量: u8,
    起始已获取UP五星干员数量: u8,
    矩阵类型: 矩阵类型枚举,
) -> Vec<(u32, f64)> {
    let mut 转移概率列表: Vec<(u32, f64)> = Vec::with_capacity(5);

    if 起始UP六星水位 == 119 {
        转移概率列表.push((
            获取状态索引(
                0,
                0,
                0,
                起始已获取UP六星干员数量 + 1,
                起始已获取UP五星干员数量,
            ),
            1.0,
        ));
    } else {
        let 六星概率 = COND_PROB_6_STAR[起始六星水位 as usize];
        let 五星概率;
        if 矩阵类型 == 矩阵类型枚举::第10抽 && 起始五星水位 == 9 {
            五星概率 = 1.0 - 六星概率;
        } else {
            五星概率 = COND_PROB_5_STAR[起始五星水位 as usize].clamp(0.0, 1.0 - 六星概率);
        }
        let 四星或三星概率 = 1.0 - 六星概率 - 五星概率;

        let (UP五星概率, 其他五星概率);
        if 矩阵类型 == 矩阵类型枚举::第51抽及以后 && 起始已获取UP五星干员数量 == 0
        {
            UP五星概率 = 五星概率;
            其他五星概率 = 0.0;
        } else {
            UP五星概率 = 五星概率 / 2.0;
            其他五星概率 = 五星概率 / 2.0;
        }

        转移概率列表.push((
            获取状态索引(
                0,
                0,
                0,
                起始已获取UP六星干员数量 + 1,
                起始已获取UP五星干员数量,
            ),
            六星概率 / 2.0,
        ));

        转移概率列表.push((
            获取状态索引(
                0,
                0,
                起始UP六星水位 + 1,
                起始已获取UP六星干员数量,
                起始已获取UP五星干员数量,
            ),
            六星概率 / 2.0,
        ));

        转移概率列表.push((
            获取状态索引(
                起始六星水位 + 1,
                0,
                起始UP六星水位 + 1,
                起始已获取UP六星干员数量,
                起始已获取UP五星干员数量 + 1,
            ),
            UP五星概率,
        ));

        转移概率列表.push((
            获取状态索引(
                起始六星水位 + 1,
                0,
                起始UP六星水位 + 1,
                起始已获取UP六星干员数量,
                起始已获取UP五星干员数量,
            ),
            其他五星概率,
        ));

        转移概率列表.push((
            获取状态索引(
                起始六星水位 + 1,
                起始五星水位 + 1,
                起始UP六星水位 + 1,
                起始已获取UP六星干员数量,
                起始已获取UP五星干员数量,
            ),
            四星或三星概率,
        ));
    }

    转移概率列表.retain(|&(_状态索引, 概率)| 概率 > 0.0);

    debug_assert!(
        f64::abs(
            转移概率列表
                .iter()
                .map(|&(_状态索引, 概率)| 概率)
                .sum::<f64>()
                - 1.0
        ) < 1e-10,
        "概率和不等于 1: ({起始六星水位}, {起始五星水位}, {起始UP六星水位}, {起始已获取UP六星干员数量}, {起始已获取UP五星干员数量})",
    );

    转移概率列表
}

// ─── 构造状态转移矩阵 ─────────────────────────────────────────────────

pub fn 构造状态转移矩阵(矩阵类型: 矩阵类型枚举) -> CooArray<f64, u32, u32> {
    let mut data: Vec<f64> = Vec::new();
    let mut row_ind: Vec<u32> = Vec::new();
    let mut col_ind: Vec<u32> = Vec::new();

    for 起始六星水位 in 0..=六星水位上限 {
        for 起始五星水位 in 0..=五星水位上限 {
            for 起始UP六星水位 in 0..=UP六星水位上限 {
                for 起始已获取UP六星干员数量 in 0..=已获取UP六星干员数量上限 {
                    for 起始已获取UP五星干员数量 in 0..=已获取UP五星干员数量上限
                    {
                        let 当前状态索引 = 获取状态索引(
                            起始六星水位,
                            起始五星水位,
                            起始UP六星水位,
                            起始已获取UP六星干员数量,
                            起始已获取UP五星干员数量,
                        );
                        let 转移列表 = 状态转移(
                            起始六星水位,
                            起始五星水位,
                            起始UP六星水位,
                            起始已获取UP六星干员数量,
                            起始已获取UP五星干员数量,
                            矩阵类型,
                        );
                        for (目标状态索引, 转移概率) in 转移列表 {
                            data.push(转移概率);
                            row_ind.push(当前状态索引 as u32);
                            col_ind.push(目标状态索引 as u32);
                        }
                    }
                }
            }
        }
    }

    CooArray {
        data,
        row_ind,
        col_ind,
        shape: (状态数量 as usize, 状态数量 as usize),
    }
}

// ─── 迭代计算 ─────────────────────────────────────────────────────────

/// 预计算每个状态对应的 (已获取UP六星干员数量, 已获取UP五星干员数量)
pub fn 预计算计数数组() -> (Vec<u8>, Vec<u8>) {
    let mut 状态已获取UP六星干员数量 = Vec::with_capacity(状态数量 as usize);
    let mut 状态已获取UP五星干员数量 = Vec::with_capacity(状态数量 as usize);

    for _六星水位 in 0..=六星水位上限 {
        for _五星水位 in 0..=五星水位上限 {
            for _UP六星水位 in 0..=UP六星水位上限 {
                for 已获取UP六星干员数量 in 0..=已获取UP六星干员数量上限 {
                    for 已获取UP五星干员数量 in 0..=已获取UP五星干员数量上限 {
                        状态已获取UP六星干员数量.push(已获取UP六星干员数量 as u8);
                        状态已获取UP五星干员数量.push(已获取UP五星干员数量 as u8);
                    }
                }
            }
        }
    }

    (状态已获取UP六星干员数量, 状态已获取UP五星干员数量)
}

/// 将分布向量聚合到 (6★, 5★) 联合分布的 7×7 矩阵
pub fn 状态已获取UP六星和UP五星干员数量联合分布(
    状态分布: &[f64],
    状态已获取UP六星干员数量: &[u8],
    状态已获取UP五星干员数量: &[u8],
) -> [[f64; 7]; 7] {
    let mut result = [[0.0f64; 7]; 7];
    for 状态序号 in 0..状态分布.len() {
        let 概率 = 状态分布[状态序号];
        if 概率 != 0.0 {
            result[状态已获取UP六星干员数量[状态序号] as usize]
                [状态已获取UP五星干员数量[状态序号] as usize] += 概率;
        }
    }
    result
}

/// 将历史获取甲乙数量联合分布保存为二进制文件
/// 布局: (901, 7, 7) 的 f64 数组，按行优先（C order）展平
pub fn 保存结果(历史结果: &[[[f64; 7]; 7]], output_dir: &str) {
    let mut flat = Vec::with_capacity(历史结果.len() * 49);
    for &mat in 历史结果 {
        for &row in &mat {
            flat.extend_from_slice(&row);
        }
    }
    let path = Path::new(output_dir).join("历史获取甲乙数量联合分布.bin");
    io::写入切片(path.to_str().unwrap(), &flat);
    println!("  已保存结果到 {}", path.display());
}
