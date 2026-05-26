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

#[allow(dead_code)]
struct csr_array {
    data: Vec<f64>,
    row_ind: Vec<usize>,
    col_ind: Vec<usize>,
    shape: (usize, usize),
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
    let mut 转移概率列表: Vec<(usize, f64)> = Vec::new();

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

fn 构造状态转移矩阵(矩阵类型: 矩阵类型枚举) -> csr_array {
    let mut data: Vec<f64> = Vec::new();
    let mut row_ind: Vec<usize> = Vec::new();
    let mut col_ind: Vec<usize> = Vec::new();

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
                            row_ind.push(当前状态索引);
                            col_ind.push(目标状态索引);
                        }
                    }
                }
            }
        }
    }

    csr_array {
        data,
        row_ind,
        col_ind,
        shape: (状态数量, 状态数量),
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

/// 将 CSR 矩阵写入 3 个 .bin.gz 文件（gzip 压缩）
fn 写入矩阵文件(out_dir: &str, prefix: &str, mat: &csr_array) {
    let dir = Path::new(out_dir);

    let data_path = dir.join(format!("{prefix}.data.bin"));
    let indices_path = dir.join(format!("{prefix}.row_ind.bin"));
    let indptr_path = dir.join(format!("{prefix}.col_ind.bin"));

    println!("  写入 {}...", data_path.display());
    写入f64切片(data_path.to_str().unwrap(), &mat.data);

    println!("  写入 {}...", indices_path.display());
    写入u32切片(
        indices_path.to_str().unwrap(),
        &mat.row_ind.iter().map(|&x| x as u32).collect::<Vec<u32>>(),
    );

    println!("  写入 {}...", indptr_path.display());
    写入u32切片(
        indptr_path.to_str().unwrap(),
        &mat.col_ind.iter().map(|&x| x as u32).collect::<Vec<u32>>(),
    );
}

fn main() {
    let output_dir = "../联动寻访怪猎二期中间结果";

    // 确保输出目录存在
    fs::create_dir_all(output_dir)
        .unwrap_or_else(|e| panic!("创建输出目录失败 {}: {}", output_dir, e));

    println!("状态空间大小: {} 个状态", 状态数量);
    println!("输出目录: {}", output_dir);
    println!();

    // ── 构建三个矩阵 ──
    let total_start = Instant::now();

    println!("构建 状态转移矩阵_第10抽...");
    let 状态转移矩阵_第10抽 = {
        let t0 = Instant::now();
        let mat = 构造状态转移矩阵(矩阵类型枚举::第10抽);
        println!("  完成，耗时 {:.1}s", t0.elapsed().as_secs_f64());
        mat
    };

    println!("构建 状态转移矩阵_前50抽但非第10抽...");
    let 状态转移矩阵_前50抽但非第10抽 = {
        let t0 = Instant::now();
        let mat = 构造状态转移矩阵(矩阵类型枚举::前50抽但非第10抽);
        println!("  完成，耗时 {:.1}s", t0.elapsed().as_secs_f64());
        mat
    };

    println!("构建 状态转移矩阵_第51抽及以后...");
    let 状态转移矩阵_第51抽及以后 = {
        let t0 = Instant::now();
        let mat = 构造状态转移矩阵(矩阵类型枚举::第51抽及以后);
        println!("  完成，耗时 {:.1}s", t0.elapsed().as_secs_f64());
        mat
    };

    println!("\n总耗时: {:.1}s", total_start.elapsed().as_secs_f64());

    // ── 写入文件 ──
    println!("\n开始写入文件...");
    let write_start = Instant::now();

    写入矩阵文件(output_dir, "状态转移矩阵_第10抽", &状态转移矩阵_第10抽);
    写入矩阵文件(
        output_dir,
        "状态转移矩阵_前50抽但非第10抽",
        &状态转移矩阵_前50抽但非第10抽,
    );
    写入矩阵文件(
        output_dir,
        "状态转移矩阵_第51抽及以后",
        &状态转移矩阵_第51抽及以后,
    );

    println!(
        "\n全部写入完成，耗时 {:.1}s",
        write_start.elapsed().as_secs_f64()
    );
    println!("\n✓ 全部完成！");
}
