use monster_hunter_collab_matrix::matrix::{
    coo_array, coo_matvec, coo_to_csc, coo_to_csr, csc_array, csc_matvec, csr_array, csr_matvec,
};
use monster_hunter_collab_matrix::monster_hunter_gacha::{
    构造状态转移矩阵, 状态数量, 矩阵类型枚举, 获取状态索引,
};
use std::time::Instant;

fn main() {
    // ── 构建 COO 矩阵 ──
    println!("构建状态转移矩阵...");
    let coo_mat = {
        let t0 = Instant::now();
        let mat = 构造状态转移矩阵(矩阵类型枚举::前50抽但非第10抽);
        println!(
            "  COO 构建，耗时 {:.4}s，非零元数量: {}",
            t0.elapsed().as_secs_f64(),
            mat.data.len()
        );
        mat
    };

    // ── COO → CSR / CSC 转换 ──
    println!("\n转换矩阵格式...");
    let csr_mat = {
        let t0 = Instant::now();
        let mat = coo_to_csr(&coo_mat);
        println!("  COO → CSR，耗时 {:.4}s", t0.elapsed().as_secs_f64());
        mat
    };
    let csc_mat = {
        let t0 = Instant::now();
        let mat = coo_to_csc(&coo_mat);
        println!("  COO → CSC，耗时 {:.4}s", t0.elapsed().as_secs_f64());
        mat
    };

    // ── 准备初始向量 ──
    let v_init = {
        let mut v = vec![0.0f64; 状态数量];
        v[获取状态索引(0, 0, 0, 0, 0)] = 1.0;
        v
    };

    const 测试迭代次数: usize = 10;

    // ── 用枚举统一三种格式，一个循环搞定 ──
    enum 矩阵格式<'a> {
        Coo(&'a coo_array),
        Csr(&'a csr_array),
        Csc(&'a csc_array),
    }

    let 测试集: [(&str, 矩阵格式, Vec<f64>); 3] = [
        ("COO", 矩阵格式::Coo(&coo_mat), v_init.clone()),
        ("CSR", 矩阵格式::Csr(&csr_mat), v_init.clone()),
        ("CSC", 矩阵格式::Csc(&csc_mat), v_init.clone()),
    ];

    let mut 结果集: Vec<(&str, f64, Vec<f64>)> = Vec::new();

    for (名称, 格式, mut v) in 测试集 {
        println!("\n{名称} 格式迭代 (v * A)...");
        let mut v_out = vec![0.0f64; 状态数量];
        let total = Instant::now();
        for i in 1..=测试迭代次数 {
            let t = Instant::now();
            match 格式 {
                矩阵格式::Coo(mat) => coo_matvec(&v, mat, &mut v_out),
                矩阵格式::Csr(mat) => csr_matvec(&v, mat, &mut v_out),
                矩阵格式::Csc(mat) => csc_matvec(&v, mat, &mut v_out),
            }
            std::mem::swap(&mut v, &mut v_out);
            println!("  i={i:>3}, {:.4}s", t.elapsed().as_secs_f64());
        }
        结果集.push((名称, total.elapsed().as_secs_f64(), v));
    }

    // ── 结果对比 ──
    let coo_total = 结果集[0].1;
    println!("\n═══════════════════════════════════════");
    println!("  格式   总耗时     平均耗时     相对 COO");
    println!("─────────────────────────────────────");
    for &(名称, total, _) in &结果集 {
        println!(
            "  {名称:>3}    {total:.4}s   {:.6}s   {:.2}x",
            total / 测试迭代次数 as f64,
            total / coo_total
        );
    }
    println!("═══════════════════════════════════════");

    // ── 正确性验证 ──
    let 误差 = 结果集[0]
        .2
        .iter()
        .zip(结果集[1].2.iter())
        .zip(结果集[2].2.iter())
        .map(|((&a, &b), &c)| (a - b).abs().max((a - c).abs()))
        .fold(0.0f64, f64::max);
    println!("\n三种格式结果最大差异: {:.2e}", 误差);
}
