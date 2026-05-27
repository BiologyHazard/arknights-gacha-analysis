use std::fs;
use std::io::Write;
use std::path::Path;

use flate2::Compression;
use flate2::write::GzEncoder;

use crate::matrix::CooArray;

// ─── 泛型写入函数 ──────────────────────────────────────────────────

/// 将任意类型的切片以原生二进制格式写入文件
pub fn 写入切片<T>(path: &str, data: &[T]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    };
    fs::write(path, bytes).unwrap_or_else(|e| panic!("写入文件失败 {}: {}", path, e));
}

/// 将任意类型的切片以 gzip 压缩写入文件
#[allow(dead_code)]
pub fn 写入切片_gz<T>(path: &str, data: &[T]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
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

// ─── 矩阵文件写入 ──────────────────────────────────────────────────

/// 将 coo 矩阵写入文件
pub fn 写入coo矩阵文件<T, U, V>(out_dir: &str, prefix: &str, coo_array: &CooArray<T, U, V>) {
    let dir = Path::new(out_dir);

    let data_path = dir.join(format!("{prefix}.data.bin"));
    let row_ind_path = dir.join(format!("{prefix}.row_ind.bin"));
    let col_ind_path = dir.join(format!("{prefix}.col_ind.bin"));

    println!("  写入 {}...", data_path.display());
    写入切片(data_path.to_str().unwrap(), &coo_array.data);

    println!("  写入 {}...", row_ind_path.display());
    写入切片(row_ind_path.to_str().unwrap(), &coo_array.row_ind);

    println!("  写入 {}...", col_ind_path.display());
    写入切片(col_ind_path.to_str().unwrap(), &coo_array.col_ind);
}
