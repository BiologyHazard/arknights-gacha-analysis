use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use flate2::Compression;
use flate2::write::GzEncoder;
use zip::CompressionMethod;
use zip::write::FileOptions;

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

// ═══════════════════════════════════════════════════════════════════
// NPY / NPZ 格式支持
// ═══════════════════════════════════════════════════════════════════
//
// NPY 格式是 NumPy 的二进制数组存储格式，支持单数组存取。
// 格式规范：魔数(6B) + 版本(2B) + 头部长度(2B) + 头部(ASCII) + 数据(原生二进制)
//
// NPZ 格式是 ZIP 归档，内含多个 .npy 文件。
// - npz: 未压缩的 ZIP
// - npz(compressed): Deflate 压缩的 ZIP

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// 生成 NPY v1.0 格式的头部字节（f64, C 顺序），自动填充到 64 字节对齐
fn 生成npy头部(shape: &[usize], fortran_order: bool) -> Vec<u8> {
    let descr = "<f8";
    let shape_str = if shape.is_empty() {
        String::new()
    } else if shape.len() == 1 {
        format!("{},", shape[0])
    } else {
        shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };

    // 匹配 NumPy 的格式：每个键值对后追加 ", "（包括最后一个）
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': {}, 'shape': ({})}}",
        descr,
        if fortran_order { "True" } else { "False" },
        shape_str
    );

    // 填充到 64 字节对齐：
    // 魔数(6) + 版本(2) + 头部长度(2) + 字典 + 空格填充 + 换行符(\n) = 64 的倍数
    let prefix_len = 10; // 6 + 2 + 2
    let header_with_newline_len = prefix_len + header_dict.len() + 1; // +1 为 \n
    let padding = (64 - (header_with_newline_len % 64)) % 64;

    let mut header: Vec<u8> = header_dict.into_bytes();
    // 先填充空格，最后以换行符结尾
    header.resize(header.len() + padding, b' ');
    header.push(b'\n');
    header
}

/// 解析 NPY 文件的头部，返回 (shape, fortran_order, 数据起始偏移量)
fn 解析npy头部(data: &[u8]) -> (Vec<usize>, bool, usize) {
    assert_eq!(&data[..6], NPY_MAGIC, "非法的 NPY 文件：魔数不匹配");
    let major = data[6];
    let (header_len, header_start) = if major == 1 {
        (u16::from_le_bytes([data[8], data[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize,
            12,
        )
    };

    let header_bytes = &data[header_start..header_start + header_len];
    let header_str = std::str::from_utf8(header_bytes)
        .expect("NPY 头部不是有效的 ASCII 字符串")
        .trim_end_matches(|c: char| c == ' ' || c == '\n');

    // 解析 shape：查找 "'shape': (" 到 ")" 之间的内容
    let shape_key = "'shape': (";
    let shape_start = header_str
        .find(shape_key)
        .expect("NPY 头部缺少 'shape' 字段")
        + shape_key.len();
    let shape_end = header_str[shape_start..]
        .find(')')
        .expect("NPY 头部 shape 格式错误")
        + shape_start;
    let shape_str = &header_str[shape_start..shape_end];
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("NPY 头部 shape 解析失败"))
        .collect();

    let fortran_order = header_str.contains("'fortran_order': True");

    (shape, fortran_order, header_start + header_len)
}

/// 将 f64 切片组装成完整的 NPY v1.0 文件字节（C 顺序）
fn 生成npy字节(data: &[f64], shape: &[usize]) -> Vec<u8> {
    let header = 生成npy头部(shape, false);
    let header_len_u16 = header.len() as u16;
    assert!(
        header.len() <= u16::MAX as usize,
        "NPY 头部过大（{} 字节），需要使用 v2.0 格式",
        header.len()
    );

    let data_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f64>(),
        )
    };

    let mut result = Vec::with_capacity(10 + header.len() + data_bytes.len());
    result.extend_from_slice(NPY_MAGIC);
    result.extend_from_slice(&[1, 0]); // version 1.0
    result.extend_from_slice(&header_len_u16.to_le_bytes());
    result.extend_from_slice(&header);
    result.extend_from_slice(data_bytes);
    result
}

/// 从 NPY 格式的字节数据中解析出 f64 数组及其形状
fn 从npy字节解析(data: &[u8]) -> (Vec<f64>, Vec<usize>) {
    let (shape, _fortran_order, data_start) = 解析npy头部(data);
    let data_bytes = &data[data_start..];
    assert_eq!(
        data_bytes.len() % 8,
        0,
        "NPY 数据长度 {} 不是 8 的倍数",
        data_bytes.len()
    );

    let result = unsafe {
        std::slice::from_raw_parts(data_bytes.as_ptr() as *const f64, data_bytes.len() / 8)
    }
    .to_vec();

    (result, shape)
}

/// 将 f64 数组写入 `.npy` 文件（NPY v1.0, C 顺序）
///
/// # 参数
/// - `path`: 输出文件路径（会自动添加 .npy 后缀）
/// - `data`: 展平的 f64 数据（C 顺序）
/// - `shape`: 数组形状，如 `&[901, 7, 7]`
pub fn 写入npy(path: &str, data: &[f64], shape: &[usize]) {
    let npy_path = if path.ends_with(".npy") {
        path.to_string()
    } else {
        format!("{}.npy", path)
    };
    let header = 生成npy头部(shape, false);
    let header_len_u16 = header.len() as u16;
    assert!(
        header.len() <= u16::MAX as usize,
        "NPY 头部过大（{} 字节），需要使用 v2.0 格式",
        header.len()
    );

    let mut file = fs::File::create(&npy_path)
        .unwrap_or_else(|e| panic!("创建 NPY 文件失败 {}: {}", npy_path, e));
    file.write_all(NPY_MAGIC).unwrap();
    file.write_all(&[1, 0]).unwrap(); // version 1.0
    file.write_all(&header_len_u16.to_le_bytes()).unwrap();
    file.write_all(&header).unwrap();

    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f64>(),
        )
    };
    file.write_all(bytes).unwrap();
}

/// 读取 `.npy` 文件，返回 (展平的 f64 数据, 形状)
///
/// # 返回值
/// - `(Vec<f64>, Vec<usize>)`: 展平的 f64 数组（C 顺序）及其形状
pub fn 读取npy(path: &str) -> (Vec<f64>, Vec<usize>) {
    let data = fs::read(path).unwrap_or_else(|e| panic!("读取 NPY 文件失败 {}: {}", path, e));
    从npy字节解析(&data)
}

// ─── NPZ 格式 ─────────────────────────────────────────────────────

/// 将多个命名的 f64 数组保存为未压缩的 `.npz` 文件
///
/// # 参数
/// - `path`: 输出文件路径（会自动添加 .npz 后缀）
/// - `arrays`: 数组列表，每个元素为 (名称, 展平数据, 形状)
///
/// # 示例
/// ```ignore
/// 写入npz("结果.npz", &[("历史分布", &flat_data, &[901, 7, 7])]);
/// ```
pub fn 写入npz(path: &str, arrays: &[(&str, &[f64], &[usize])]) {
    let npz_path = if path.ends_with(".npz") {
        path.to_string()
    } else {
        format!("{}.npz", path)
    };
    let file = fs::File::create(&npz_path)
        .unwrap_or_else(|e| panic!("创建 NPZ 文件失败 {}: {}", npz_path, e));
    let mut zip_writer = zip::ZipWriter::new(file);
    let options: FileOptions<'_, ()> =
        FileOptions::default().compression_method(CompressionMethod::Stored);

    for (name, data, shape) in arrays {
        let npy_data = 生成npy字节(data, shape);
        zip_writer
            .start_file(&format!("{name}.npy"), options)
            .unwrap();
        zip_writer.write_all(&npy_data).unwrap();
    }

    zip_writer
        .finish()
        .unwrap_or_else(|e| panic!("关闭 NPZ 文件失败 {}: {}", npz_path, e));
}

/// 将多个命名的 f64 数组保存为压缩的 `.npz` 文件（使用 Deflate 压缩）
///
/// # 参数
/// - `path`: 输出文件路径（会自动添加 .npz 后缀）
/// - `arrays`: 数组列表，每个元素为 (名称, 展平数据, 形状)
///
/// # 示例
/// ```ignore
/// 写入npz压缩("结果.npz", &[("历史分布", &flat_data, &[901, 7, 7])]);
/// ```
pub fn 写入npz压缩(path: &str, arrays: &[(&str, &[f64], &[usize])]) {
    let npz_path = if path.ends_with(".npz") {
        path.to_string()
    } else {
        format!("{}.npz", path)
    };
    let file = fs::File::create(&npz_path)
        .unwrap_or_else(|e| panic!("创建 NPZ 文件失败 {}: {}", npz_path, e));
    let mut zip_writer = zip::ZipWriter::new(file);
    let options: FileOptions<'_, ()> =
        FileOptions::default().compression_method(CompressionMethod::Deflated);

    for (name, data, shape) in arrays {
        let npy_data = 生成npy字节(data, shape);
        zip_writer
            .start_file(&format!("{name}.npy"), options)
            .unwrap();
        zip_writer.write_all(&npy_data).unwrap();
    }

    zip_writer
        .finish()
        .unwrap_or_else(|e| panic!("关闭 NPZ 文件失败 {}: {}", npz_path, e));
}

/// 读取 `.npz` 文件，返回所有数组的映射
///
/// # 返回值
/// `HashMap<String, (Vec<f64>, Vec<usize>)>` — 键为数组名，值为 (展平的 f64 数据, 形状)
///
/// # 示例
/// ```ignore
/// let map = 读取npz("结果.npz");
/// let (data, shape) = &map["历史分布"];
/// ```
pub fn 读取npz(path: &str) -> HashMap<String, (Vec<f64>, Vec<usize>)> {
    let file = fs::File::open(path).unwrap_or_else(|e| panic!("打开 NPZ 文件失败 {}: {}", path, e));
    let mut archive =
        zip::ZipArchive::new(file).unwrap_or_else(|e| panic!("读取 NPZ 文件失败 {}: {}", path, e));

    let mut result = HashMap::new();

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).unwrap();
        let entry_name = entry.name().to_string();

        if entry_name.ends_with(".npy") {
            let mut buf = Vec::new();
            entry.read_to_end(&mut buf).unwrap();
            let (数据, 形状) = 从npy字节解析(&buf);
            let name = entry_name.trim_end_matches(".npy").to_string();
            result.insert(name, (数据, 形状));
        }
    }

    result
}
