// 允许使用非标准命名风格
#![allow(nonstandard_style)]

use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use flate2::Compression;
use flate2::write::GzEncoder;

const 六星水位上限: usize = 98;
const 五星水位上限: usize = 39;
const UP六星水位上限: usize = 119;
const 已获取UP六星干员数量上限: usize = 6;
const 已获取UP五星干员数量上限: usize = 6;
const 迭代次数: usize = 900;

/// 状态总数 = 99 * 40 * 120 * 7 * 7 = 23284800
const 状态数量: usize = (六星水位上限 + 1)
    * (五星水位上限 + 1)
    * (UP六星水位上限 + 1)
    * (已获取UP六星干员数量上限 + 1)
    * (已获取UP五星干员数量上限 + 1);

const COND_PROB_6_STAR: [f64; 六星水位上限 + 1] = {
    let mut arr = [0.0; 六星水位上限 + 1];
    let mut 水位 = 0;
    while 水位 <= 六星水位上限 {
        arr[水位] = if 水位 < 50 {
            0.02
        } else {
            (水位 - 49) as f64 * 0.02 + 0.02
        };
        水位 += 1;
    }
    arr
};

const COND_PROB_5_STAR: [f64; 五星水位上限 + 1] = {
    let mut arr = [0.0; 五星水位上限 + 1];
    let mut 水位 = 0;
    while 水位 <= 五星水位上限 {
        arr[水位] = if 水位 < 15 {
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
fn 获取状态索引(
    六星水位: usize,
    五星水位: usize,
    UP六星水位: usize,
    已获取UP六星干员数量: usize,
    已获取UP五星干员数量: usize,
) -> usize {
    (((六星水位 * (五星水位上限 + 1) + 五星水位) * (UP六星水位上限 + 1) + UP六星水位)
        * (已获取UP六星干员数量上限 + 1)
        + 已获取UP六星干员数量.min(已获取UP六星干员数量上限))
        * (已获取UP五星干员数量上限 + 1)
        + 已获取UP五星干员数量.min(已获取UP五星干员数量上限)
}

struct csr_array {
    data: Vec<f64>,
    row_ind: Vec<u32>,
    col_ind: Vec<u32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum 矩阵类型枚举 {
    第10抽,
    前50抽但非第10抽,
    第51抽及以后,
}

fn 状态转移(
    起始六星水位: usize,
    起始五星水位: usize,
    起始UP六星水位: usize,
    起始已获取UP六星干员数量: usize,
    起始已获取UP五星干员数量: usize,
    矩阵类型: 矩阵类型枚举,
) -> Vec<(usize, f64)> {
    let mut 转移概率列表: Vec<(usize, f64)> = Vec::with_capacity(5);

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
        let 六星概率 = COND_PROB_6_STAR[起始六星水位];
        let 五星概率;
        if 矩阵类型 == 矩阵类型枚举::第10抽 && 起始五星水位 == 9 {
            五星概率 = 1.0 - 六星概率;
        } else {
            五星概率 = COND_PROB_5_STAR[起始五星水位].clamp(0.0, 1.0 - 六星概率);
        }
        let 四星或三星概率 = 1.0 - 六星概率 - 五星概率;

        let (UP五星概率, 其他五星概率);
        if 矩阵类型 == 矩阵类型枚举::前50抽但非第10抽 && 起始已获取UP五星干员数量 == 0
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

fn 构造状态转移矩阵(矩阵类型: 矩阵类型枚举) -> csr_array {
    let mut data: Vec<f64> = Vec::new();
    let mut row_ind: Vec<u32> = Vec::new();
    let mut col_ind: Vec<u32> = Vec::new();
    // let mut 当前状态索引: u32 = 0;

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
                        ) as u32;
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
                            row_ind.push(当前状态索引);
                            col_ind.push(目标状态索引 as u32);
                        }
                        // 当前状态索引 += 1;
                    }
                }
            }
        }
    }

    csr_array {
        data,
        row_ind,
        col_ind,
    }
}

/// 将 f64 切片写入文件
fn 写入f64切片(path: &str, data: &[f64]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f64>(),
        )
    };
    fs::write(path, bytes).unwrap_or_else(|e| panic!("写入文件失败 {}: {}", path, e));
}

/// 将 u32 切片写入文件
fn 写入u32切片(path: &str, data: &[u32]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<u32>(),
        )
    };
    fs::write(path, bytes).unwrap_or_else(|e| panic!("写入文件失败 {}: {}", path, e));
}

/// 将 f64 切片以 gzip 压缩写入文件
fn 写入f64切片_gz(path: &str, data: &[f64]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f64>(),
        )
    };
    let file = fs::File::create(path).unwrap_or_else(|e| panic!("创建文件失败 {}: {}", path, e));
    let mut encoder = GzEncoder::new(file, Compression::default());
    encoder
        .write_all(bytes)
        .unwrap_or_else(|e| panic!("写入文件失败 {}: {}", path, e));
    encoder
        .finish()
        .unwrap_or_else(|e| panic!("关闭文件失败 {}: {}", path, e));
}

/// 将 u32 切片以 gzip 压缩写入文件
fn 写入u32切片_gz(path: &str, data: &[u32]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<u32>(),
        )
    };
    let file = fs::File::create(path).unwrap_or_else(|e| panic!("创建文件失败 {}: {}", path, e));
    let mut encoder = GzEncoder::new(file, Compression::default());
    encoder
        .write_all(bytes)
        .unwrap_or_else(|e| panic!("写入文件失败 {}: {}", path, e));
    encoder
        .finish()
        .unwrap_or_else(|e| panic!("关闭文件失败 {}: {}", path, e));
}

/// 将 CSR 矩阵写入文件
fn 写入矩阵文件(out_dir: &str, prefix: &str, mat: &csr_array) {
    let dir = Path::new(out_dir);

    let data_path = dir.join(format!("{prefix}.data.bin"));
    let row_ind_path = dir.join(format!("{prefix}.row_ind.bin"));
    let col_ind_path = dir.join(format!("{prefix}.col_ind.bin"));

    println!("  写入 {}...", data_path.display());
    写入f64切片(data_path.to_str().unwrap(), &mat.data);

    println!("  写入 {}...", row_ind_path.display());
    写入u32切片(row_ind_path.to_str().unwrap(), &mat.row_ind);

    println!("  写入 {}...", col_ind_path.display());
    写入u32切片(col_ind_path.to_str().unwrap(), &mat.col_ind);
}

// ─── 迭代计算 ─────────────────────────────────────────────────────────

/// 预计算每个状态对应的 (已获取UP六星干员数量, 已获取UP五星干员数量)
fn 预计算计数数组() -> (Vec<u8>, Vec<u8>) {
    let mut 状态已获取UP六星干员数量 = Vec::with_capacity(状态数量);
    let mut 状态已获取UP五星干员数量 = Vec::with_capacity(状态数量);

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

/// 计算 `v_new = v_old @ matrix`（COO 格式的稀疏矩阵 × 密集向量乘法）
fn coo_matvec(v_old: &[f64], mat: &csr_array, v_new: &mut [f64]) {
    v_new.fill(0.0);
    for k in 0..mat.data.len() {
        let row = mat.row_ind[k] as usize;
        let col = mat.col_ind[k] as usize;
        v_new[col] += mat.data[k] * v_old[row];
    }
}

/// 将分布向量聚合到 (6★, 5★) 联合分布的 7×7 矩阵
fn aggregate(状态分布: &[f64], 六星计数: &[u8], 五星计数: &[u8]) -> [[f64; 7]; 7] {
    let mut result = [[0.0f64; 7]; 7];
    for i in 0..状态分布.len() {
        let p = 状态分布[i];
        if p != 0.0 {
            result[六星计数[i] as usize][五星计数[i] as usize] += p;
        }
    }
    result
}

/// 将历史获取甲乙数量联合分布保存为二进制文件
/// 布局: (901, 7, 7) 的 f64 数组，按行优先（C order）展平
fn 保存结果(历史结果: &[[[f64; 7]; 7]], output_dir: &str) {
    let mut flat = Vec::with_capacity(历史结果.len() * 49);
    for &mat in 历史结果 {
        for &row in &mat {
            flat.extend_from_slice(&row);
        }
    }
    let path = Path::new(output_dir).join("历史获取甲乙数量联合分布.bin");
    写入f64切片(path.to_str().unwrap(), &flat);
    println!("  已保存结果到 {}", path.display());
}

fn main() {
    let output_dir = "../联动寻访怪猎二期中间结果";

    // 确保输出目录存在
    fs::create_dir_all(output_dir)
        .unwrap_or_else(|e| panic!("创建输出目录失败 {}: {}", output_dir, e));

    println!("状态空间大小: {} 个状态", 状态数量);
    println!("输出目录: {}", output_dir);

    // ── 构建全部 3 个状态转移矩阵 ──
    let total_start = Instant::now();

    println!("\n构建状态转移矩阵...");
    let 状态转移矩阵_第10抽 = {
        let t0 = Instant::now();
        let mat = 构造状态转移矩阵(矩阵类型枚举::第10抽);
        println!(
            "  状态转移矩阵_第10抽，耗时 {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        // 写入矩阵文件(output_dir, "状态转移矩阵_第10抽", &mat);
        mat
    };
    let 状态转移矩阵_前50抽但非第10抽 = {
        let t0 = Instant::now();
        let mat = 构造状态转移矩阵(矩阵类型枚举::前50抽但非第10抽);
        println!(
            "  状态转移矩阵_前50抽但非第10抽，耗时 {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        // 写入矩阵文件(output_dir, "状态转移矩阵_前50抽但非第10抽", &mat);
        mat
    };
    let 状态转移矩阵_第51抽及以后 = {
        let t0 = Instant::now();
        let mat = 构造状态转移矩阵(矩阵类型枚举::第51抽及以后);
        println!(
            "  状态转移矩阵_第51抽及以后，耗时 {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        // 写入矩阵文件(output_dir, "状态转移矩阵_第51抽及以后", &mat);
        mat
    };

    // ── 迭代参数 ──
    let mut 历史结果 = vec![[[0.0f64; 7]; 7]; 迭代次数 + 1];
    历史结果[0][0][0] = 1.0; // 0 抽时 (六星=0, 五星=0)

    let mut 旧状态分布 = vec![0.0f64; 状态数量];
    let mut 新状态分布 = vec![0.0f64; 状态数量];
    旧状态分布[获取状态索引(0, 0, 0, 0, 0)] = 1.0;

    // ── 预计算计数数组 ──
    println!("\n预计算计数数组...");
    let (六星计数, 五星计数) = 预计算计数数组();
    println!("  完成");

    // ── 执行 900 次迭代 ──
    println!("\n开始迭代计算...\n");
    let iter_start = Instant::now();

    for i in 1..=迭代次数 {
        let step_start = Instant::now();
        let mat = if i == 10 {
            &状态转移矩阵_第10抽
        } else if i <= 50 {
            &状态转移矩阵_前50抽但非第10抽
        } else {
            &状态转移矩阵_第51抽及以后
        };

        coo_matvec(&旧状态分布, mat, &mut 新状态分布);
        历史结果[i] = aggregate(&新状态分布, &六星计数, &五星计数);
        std::mem::swap(&mut 旧状态分布, &mut 新状态分布);

        let step_secs = step_start.elapsed().as_secs_f64();
        println!("  i={i:>3}, 耗时 {step_secs:.3}s");
    }

    println!("\n迭代总耗时: {:.3}s", iter_start.elapsed().as_secs_f64());
    println!("\n总耗时: {:.1}s", total_start.elapsed().as_secs_f64());

    // ── 保存结果 ──
    println!("\n保存结果...");
    保存结果(&历史结果, output_dir);
    println!("\n✓ 全部完成！");
}
