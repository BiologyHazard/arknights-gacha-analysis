use std::fs;
use std::path::Path;
use std::time::Instant;

// ─── 常量 ───────────────────────────────────────────────────────────────

const 六星水位上限: usize = 99; // 六星水位 0..98
const 五星水位上限: usize = 40;
const UP六星水位上限: usize = 120;
const 已获取UP六星干员数量上限: usize = 6;
const 已获取UP五星干员数量上限: usize = 6; // 已获取UP五星干员数量上限

// 状态总数 = 99 * 40 * 120 * 7 * 7 = 23,284,800
const 状态数量: usize = 六星水位上限
    * 五星水位上限
    * UP六星水位上限
    * (已获取UP六星干员数量上限 + 1)
    * (已获取UP五星干员数量上限 + 1);

// ─── 概率函数 ─────────────────────────────────────────────────────────

#[inline]
fn COND_PROB_6_STAR(水位: usize) -> f64 {
    if 水位 < 50 {
        0.02
    } else {
        (水位 - 49) as f64 * 0.02 + 0.02
    }
}

#[inline]
fn COND_PROB_5_STAR(水位: usize) -> f64 {
    if 水位 < 15 {
        0.08
    } else if 水位 < 20 {
        (水位 - 14) as f64 * 0.02 + 0.08
    } else {
        (水位 - 19) as f64 * 0.04 + 0.18
    }
}

// ─── 状态索引 ─────────────────────────────────────────────────────────

#[inline]
fn 状态索引(
    六星水位: usize,
    五星水位: usize,
    UP六星水位: usize,
    已获取UP六星干员数量: usize,
    已获取UP五星干员数量: usize,
) -> usize {
    (((六星水位 * 五星水位上限 + 五星水位) * UP六星水位上限 + UP六星水位)
        * (已获取UP六星干员数量上限 + 1)
        + 已获取UP六星干员数量.min(已获取UP六星干员数量上限))
        * (已获取UP五星干员数量上限 + 1)
        + 已获取UP五星干员数量.min(已获取UP五星干员数量上限)
}

// ─── CSR 矩阵 ────────────────────────────────────────────────────────

struct CSR矩阵 {
    data: Vec<f64>,
    indices: Vec<i32>,
    indptr: Vec<i32>,
}

// ─── 转移逻辑（一次计算全部三种模式）────────────────────────────

/// 给定起始状态，计算全部三种模式下的转移概率列表。
/// 返回 3 个 Vec，每个 Vec 包含 (目标状态索引, 概率)
fn 计算全部模式转移(
    六星水位: usize,
    五星水位: usize,
    UP六星水位: usize,
    已获取UP六星干员数量: usize,
    已获取UP五星干员数量: usize,
    结果缓存: &mut [Vec<(usize, f64)>; 3],
) {
    // 清空上次结果
    for r in 结果缓存.iter_mut() {
        r.clear();
    }

    // ---- UP 6★ 硬保底（连续 119 抽未抽到 UP 6★） ----
    if UP六星水位 == 119 {
        let 目标序号 = 状态索引(0, 0, 0, 已获取UP六星干员数量 + 1, 已获取UP五星干员数量);
        for r in 结果缓存.iter_mut() {
            r.push((目标序号, 1.0));
        }
        return;
    }

    // ---- 计算抽到不同星级干员的概率 ----
    let 六星概率 = COND_PROB_6_STAR(六星水位);

    // 第10抽特殊处理（五星保底），其余模式用 clamp
    let 第10抽五星概率 = {
        let raw = if 五星水位 == 9 {
            1.0 - 六星概率
        } else {
            COND_PROB_5_STAR(五星水位)
        };
        raw.clamp(0.0, 1.0 - 六星概率)
    };
    let 普通五星概率 = COND_PROB_5_STAR(五星水位).clamp(0.0, 1.0 - 六星概率);

    let 各模式五星概率 = [第10抽五星概率, 普通五星概率, 普通五星概率];

    let UP六星概率 = 六星概率 / 2.0;
    let 其他六星概率 = 六星概率 / 2.0;

    let 下一个UP六星水位 = UP六星水位 + 1;
    let 下一个六星水位 = 六星水位 + 1;

    for (m, 结果) in 结果缓存.iter_mut().enumerate() {
        let 五星概率 = 各模式五星概率[m];
        let 四星或三星概率 = 1.0 - 六星概率 - 五星概率;

        // UP 5★ 概率拆分: 前50抽但非第10抽 且 尚未获取过UP5★ → 全部五星均为UP
        let (UP五星概率, 其他五星概率) = match m {
            1 if 已获取UP五星干员数量 == 0 => (五星概率, 0.0),
            _ => (五星概率 / 2.0, 五星概率 / 2.0),
        };

        // 抽到UP6星干员
        结果.push((
            状态索引(0, 0, 0, 已获取UP六星干员数量 + 1, 已获取UP五星干员数量),
            UP六星概率,
        ));

        // 抽到其他6星干员
        结果.push((
            状态索引(
                0,
                0,
                下一个UP六星水位,
                已获取UP六星干员数量,
                已获取UP五星干员数量,
            ),
            其他六星概率,
        ));

        // 抽到UP五星干员
        if UP五星概率 > 0.0 {
            结果.push((
                状态索引(
                    下一个六星水位,
                    0,
                    下一个UP六星水位,
                    已获取UP六星干员数量,
                    已获取UP五星干员数量 + 1,
                ),
                UP五星概率,
            ));
        }

        // 抽到其他五星干员
        if 其他五星概率 > 0.0 {
            结果.push((
                状态索引(
                    下一个六星水位,
                    0,
                    下一个UP六星水位,
                    已获取UP六星干员数量,
                    已获取UP五星干员数量,
                ),
                其他五星概率,
            ));
        }

        // 抽到四星及以下干员
        if 四星或三星概率 > 0.0 {
            结果.push((
                状态索引(
                    下一个六星水位,
                    五星水位 + 1,
                    下一个UP六星水位,
                    已获取UP六星干员数量,
                    已获取UP五星干员数量,
                ),
                四星或三星概率,
            ));
        }

        // 调试断言：概率和应为 1.0
        debug_assert!(
            (结果.iter().map(|&(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-10,
            "概率和不等于1: ({},{},{},{},{}) mode={} sum={}",
            六星水位,
            五星水位,
            UP六星水位,
            已获取UP六星干员数量,
            已获取UP五星干员数量,
            m,
            结果.iter().map(|&(_, p)| p).sum::<f64>(),
        );
    }
}

// ─── 矩阵构建 ────────────────────────────────────────────────────────

/// 构建全部 3 个状态转移矩阵（CSR 格式）
fn 构建矩阵() -> [CSR矩阵; 3] {
    let 总开始 = Instant::now();

    // 每行最多 5 个非零元，预分配以节省 reallocation
    let mut 缓存: [Vec<(usize, f64)>; 3] = [
        Vec::with_capacity(5),
        Vec::with_capacity(5),
        Vec::with_capacity(5),
    ];

    // ====== 第1遍：统计每行非零元数量 ======
    println!("开始第1遍：统计每行非零元数量...");
    let 第一遍开始 = Instant::now();

    let mut indptr组 = [
        vec![0i32; 状态数量 + 1],
        vec![0i32; 状态数量 + 1],
        vec![0i32; 状态数量 + 1],
    ];

    for 六星水位 in 0..六星水位上限 {
        if 六星水位 % 5 == 0 {
            println!("  第1遍: 六星水位 {} / {}", 六星水位, 六星水位上限 - 1);
        }
        for 五星水位 in 0..五星水位上限 {
            for UP六星水位 in 0..UP六星水位上限 {
                for 已获取UP六星干员数量 in 0..=已获取UP六星干员数量上限 {
                    for 已获取UP五星干员数量 in 0..=已获取UP五星干员数量上限 {
                        let 序号 = 状态索引(
                            六星水位,
                            五星水位,
                            UP六星水位,
                            已获取UP六星干员数量,
                            已获取UP五星干员数量,
                        );

                        计算全部模式转移(
                            六星水位,
                            五星水位,
                            UP六星水位,
                            已获取UP六星干员数量,
                            已获取UP五星干员数量,
                            &mut 缓存,
                        );

                        for (m, 结果) in 缓存.iter().enumerate() {
                            indptr组[m][序号 + 1] = 结果.len() as i32;
                        }
                    }
                }
            }
        }
    }

    // 累加得到 indptr（CSR 行指针）
    for m in 0..3 {
        for i in 1..=状态数量 {
            indptr组[m][i] += indptr组[m][i - 1];
        }
    }
    let 非零元数量: [usize; 3] = [
        indptr组[0][状态数量] as usize,
        indptr组[1][状态数量] as usize,
        indptr组[2][状态数量] as usize,
    ];

    let 标签 = ["第10抽", "前50抽但非第10抽", "第51抽及以后"];
    println!("第1遍完成，耗时 {:.1}s", 第一遍开始.elapsed().as_secs_f64());
    for m in 0..3 {
        let 每行平均 = 非零元数量[m] as u64 / 状态数量 as u64;
        println!(
            "  - 矩阵{} ({}): 非零元 = {} (平均 {}/行)",
            m, 标签[m], 非零元数量[m], 每行平均
        );
    }

    // ====== 分配 data 和 indices ======
    let mut 矩阵0 = CSR矩阵 {
        data: vec![0.0f64; 非零元数量[0]],
        indices: vec![0i32; 非零元数量[0]],
        indptr: std::mem::take(&mut indptr组[0]),
    };
    let mut 矩阵1 = CSR矩阵 {
        data: vec![0.0f64; 非零元数量[1]],
        indices: vec![0i32; 非零元数量[1]],
        indptr: std::mem::take(&mut indptr组[1]),
    };
    let mut 矩阵2 = CSR矩阵 {
        data: vec![0.0f64; 非零元数量[2]],
        indices: vec![0i32; 非零元数量[2]],
        indptr: std::mem::take(&mut indptr组[2]),
    };

    // 填充指针 = indptr[0..状态数量]（每行起始位置）
    let 填充指针0 = 矩阵0.indptr[..状态数量].to_vec();
    let 填充指针1 = 矩阵1.indptr[..状态数量].to_vec();
    let 填充指针2 = 矩阵2.indptr[..状态数量].to_vec();
    let 所有填充指针 = [填充指针0, 填充指针1, 填充指针2];

    // ====== 第2遍：填充 data 和 indices ======
    println!("\n开始第2遍：填充 data 和 indices...");
    let 第二遍开始 = Instant::now();

    // 收集可变引用以便循环中写入
    let 所有数据 = [&mut 矩阵0.data, &mut 矩阵1.data, &mut 矩阵2.data];
    let 所有列索引 = [&mut 矩阵0.indices, &mut 矩阵1.indices, &mut 矩阵2.indices];

    for 六星水位 in 0..六星水位上限 {
        if 六星水位 % 5 == 0 {
            println!("  第2遍: 六星水位 {} / {}", 六星水位, 六星水位上限 - 1);
        }
        for 五星水位 in 0..五星水位上限 {
            for UP六星水位 in 0..UP六星水位上限 {
                for 已获取UP六星干员数量 in 0..=已获取UP六星干员数量上限 {
                    for 已获取UP五星干员数量 in 0..=已获取UP五星干员数量上限 {
                        let 序号 = 状态索引(
                            六星水位,
                            五星水位,
                            UP六星水位,
                            已获取UP六星干员数量,
                            已获取UP五星干员数量,
                        );

                        计算全部模式转移(
                            六星水位,
                            五星水位,
                            UP六星水位,
                            已获取UP六星干员数量,
                            已获取UP五星干员数量,
                            &mut 缓存,
                        );

                        for (m, 结果) in 缓存.iter().enumerate() {
                            let start = 所有填充指针[m][序号] as usize;
                            for (j, &(目标序号, 概率)) in 结果.iter().enumerate() {
                                所有数据[m][start + j] = 概率;
                                所有列索引[m][start + j] = 目标序号 as i32;
                            }
                        }
                    }
                }
            }
        }
    }

    println!("第2遍完成，耗时 {:.1}s", 第二遍开始.elapsed().as_secs_f64());

    println!("\n总耗时: {:.1}s", 总开始.elapsed().as_secs_f64());

    [矩阵0, 矩阵1, 矩阵2]
}

// ─── 文件写入 ────────────────────────────────────────────────────────

/// 将 f64 切片以原始二进制写入文件
fn 写入f64切片(path: &str, data: &[f64]) {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f64>(),
        )
    };
    fs::write(path, bytes).unwrap_or_else(|e| panic!("写入文件失败 {}: {}", path, e));
}

/// 将 i32 切片以原始二进制写入文件
fn 写入i32切片(path: &str, data: &[i32]) {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<i32>(),
        )
    };
    fs::write(path, bytes).unwrap_or_else(|e| panic!("写入文件失败 {}: {}", path, e));
}

/// 将 CSR 矩阵写入 3 个 .bin 文件
fn 写入矩阵文件(out_dir: &str, prefix: &str, mat: &CSR矩阵) {
    let dir = Path::new(out_dir);

    let data_path = dir.join(format!("{}.data.bin", prefix));
    let indices_path = dir.join(format!("{}.indices.bin", prefix));
    let indptr_path = dir.join(format!("{}.indptr.bin", prefix));

    println!("  写入 {}...", data_path.display());
    写入f64切片(data_path.to_str().unwrap(), &mat.data);

    println!("  写入 {}...", indices_path.display());
    写入i32切片(indices_path.to_str().unwrap(), &mat.indices);

    println!("  写入 {}...", indptr_path.display());
    写入i32切片(indptr_path.to_str().unwrap(), &mat.indptr);
}

// ─── 主函数 ─────────────────────────────────────────────────────────

fn main() {
    let 输出目录 = "联动寻访怪猎二期中间结果";

    // 确保输出目录存在
    fs::create_dir_all(输出目录).unwrap_or_else(|e| panic!("创建输出目录失败 {}: {}", 输出目录, e));

    println!("状态空间大小: {} 个状态", 状态数量);
    println!(
        "预计每个矩阵非零元 ~{}",
        状态数量 * 5 // 最多 5 个非零元/行
    );
    println!("输出目录: {}", 输出目录);
    println!();

    // 构建三个矩阵
    let [矩阵_第10抽, 矩阵_前50抽但非第10抽, 矩阵_第51抽及以后] = 构建矩阵();

    // 写入文件
    println!("\n开始写入文件...");
    let 写入开始 = Instant::now();

    写入矩阵文件(输出目录, "状态转移矩阵_第10抽", &矩阵_第10抽);
    写入矩阵文件(
        输出目录,
        "状态转移矩阵_前50抽但非第10抽",
        &矩阵_前50抽但非第10抽,
    );
    写入矩阵文件(输出目录, "状态转移矩阵_第51抽及以后", &矩阵_第51抽及以后);

    println!(
        "\n全部写入完成，耗时 {:.1}s",
        写入开始.elapsed().as_secs_f64()
    );
    println!("\n✓ 全部完成！");
}
