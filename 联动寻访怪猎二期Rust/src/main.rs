// 允许使用非标准命名风格
#![allow(nonstandard_style)]

use std::fs;
use std::time::Instant;

use monster_hunter_collab_matrix::matrix::{CsrArray, coo_to_csr, vec_mul_csr_array};
use monster_hunter_collab_matrix::monster_hunter_gacha::{
    保存结果, 构造状态转移矩阵, 状态已获取UP六星和UP五星干员数量联合分布, 状态数量, 矩阵类型枚举,
    获取状态索引, 迭代次数, 预计算计数数组,
};

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
        let coo_array = 构造状态转移矩阵(矩阵类型枚举::第10抽);
        println!(
            "  状态转移矩阵_第10抽，耗时 {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        let t0 = Instant::now();
        let csr_array: CsrArray<f64, u32, u32> = coo_to_csr(&coo_array);
        println!("  COO → CSR，耗时 {:.1}s", t0.elapsed().as_secs_f64());
        // 写入coo矩阵文件(output_dir, "状态转移矩阵_第10抽", &coo_array);
        csr_array
    };
    let 状态转移矩阵_前50抽但非第10抽 = {
        let t0 = Instant::now();
        let coo_array = 构造状态转移矩阵(矩阵类型枚举::前50抽但非第10抽);
        println!(
            "  状态转移矩阵_前50抽但非第10抽，耗时 {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        let t0 = Instant::now();
        let csr_array: CsrArray<f64, u32, u32> = coo_to_csr(&coo_array);
        println!("  COO → CSR，耗时 {:.1}s", t0.elapsed().as_secs_f64());
        // 写入coo矩阵文件(output_dir, "状态转移矩阵_前50抽但非第10抽", &coo_array);
        csr_array
    };
    let 状态转移矩阵_第51抽及以后 = {
        let t0 = Instant::now();
        let coo_array = 构造状态转移矩阵(矩阵类型枚举::第51抽及以后);
        println!(
            "  状态转移矩阵_第51抽及以后，耗时 {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        let t0 = Instant::now();
        let csr_array: CsrArray<f64, u32, u32> = coo_to_csr(&coo_array);
        println!("  COO → CSR，耗时 {:.1}s", t0.elapsed().as_secs_f64());
        // 写入coo矩阵文件(output_dir, "状态转移矩阵_第51抽及以后", &coo_array);
        csr_array
    };

    // ── 迭代参数 ──
    let mut 历史结果 = vec![[[0.0f64; 7]; 7]; 迭代次数 as usize + 1];
    历史结果[0][0][0] = 1.0; // 0 抽时 (六星=0, 五星=0)

    let mut 旧状态分布 = vec![0.0f64; 状态数量 as usize];
    let mut 新状态分布 = vec![0.0f64; 状态数量 as usize];
    旧状态分布[获取状态索引(0, 0, 0, 0, 0) as usize] = 1.0;

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

        vec_mul_csr_array(&旧状态分布, mat, &mut 新状态分布);
        历史结果[i as usize] = 状态已获取UP六星和UP五星干员数量联合分布(
            &新状态分布,
            &六星计数,
            &五星计数,
        );
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
