use std::fs;
use std::io::Write;
use std::path::Path;

use flate2::Compression;
use flate2::write::GzEncoder;

use crate::matrix::coo_array;

/// 将 f64 切片写入文件
pub fn 写入f64切片(path: &str, data: &[f64]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f64>(),
        )
    };
    fs::write(path, bytes).unwrap_or_else(|e| panic!("写入文件失败 {}: {}", path, e));
}

/// 将 u32 切片写入文件
pub fn 写入u32切片(path: &str, data: &[u32]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<u32>(),
        )
    };
    fs::write(path, bytes).unwrap_or_else(|e| panic!("写入文件失败 {}: {}", path, e));
}

/// 将 f64 切片以 gzip 压缩写入文件
#[allow(dead_code)]
pub fn 写入f64切片_gz(path: &str, data: &[f64]) {
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
#[allow(dead_code)]
pub fn 写入u32切片_gz(path: &str, data: &[u32]) {
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

/// 将 coo 矩阵写入文件
pub fn 写入coo矩阵文件(out_dir: &str, prefix: &str, mat: &coo_array) {
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
