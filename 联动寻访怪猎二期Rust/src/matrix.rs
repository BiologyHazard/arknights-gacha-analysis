#![allow(nonstandard_style)]

// ─── 稀疏矩阵格式定义 ────────────────────────────────────────────────

pub struct coo_array {
    pub data: Vec<f64>,
    pub row_ind: Vec<u32>,
    pub col_ind: Vec<u32>,
    pub shape: (usize, usize),
}

pub struct csr_array {
    pub data: Vec<f64>,
    pub col_ind: Vec<u32>,
    pub row_ptr: Vec<u32>,
    pub shape: (usize, usize),
}

pub struct csc_array {
    pub data: Vec<f64>,
    pub row_ind: Vec<u32>,
    pub col_ptr: Vec<u32>,
    pub shape: (usize, usize),
}

// ─── COO → CSR / CSC 转换 ─────────────────────────────────────────

/// 将 COO 格式转换为 CSR 格式
pub fn coo_to_csr(coo: &coo_array) -> csr_array {
    let (n_rows, _n_cols) = coo.shape;
    let nnz = coo.data.len();

    // 统计每行的非零元素数量
    let mut row_count = vec![0u32; n_rows];
    for &r in &coo.row_ind {
        row_count[r as usize] += 1;
    }

    // 构建 row_ptr: 前缀和
    let mut row_ptr = vec![0u32; n_rows + 1];
    for i in 0..n_rows {
        row_ptr[i + 1] = row_ptr[i] + row_count[i];
    }

    // 按行填入 data 和 col_ind
    let mut data = vec![0.0f64; nnz];
    let mut col_ind = vec![0u32; nnz];
    // 临时记录每行当前写入位置
    let mut cursor = row_ptr[..n_rows].to_vec();

    for k in 0..nnz {
        let r = coo.row_ind[k] as usize;
        let pos = cursor[r] as usize;
        data[pos] = coo.data[k];
        col_ind[pos] = coo.col_ind[k];
        cursor[r] += 1;
    }

    csr_array {
        data,
        col_ind,
        row_ptr,
        shape: coo.shape,
    }
}

/// 将 COO 格式转换为 CSC 格式
pub fn coo_to_csc(coo: &coo_array) -> csc_array {
    let (_n_rows, n_cols) = coo.shape;
    let nnz = coo.data.len();

    // 统计每列的非零元素数量
    let mut col_count = vec![0u32; n_cols];
    for &c in &coo.col_ind {
        col_count[c as usize] += 1;
    }

    // 构建 col_ptr: 前缀和
    let mut col_ptr = vec![0u32; n_cols + 1];
    for j in 0..n_cols {
        col_ptr[j + 1] = col_ptr[j] + col_count[j];
    }

    // 按列填入 data 和 row_ind
    let mut data = vec![0.0f64; nnz];
    let mut row_ind = vec![0u32; nnz];
    let mut cursor = col_ptr[..n_cols].to_vec();

    for k in 0..nnz {
        let c = coo.col_ind[k] as usize;
        let pos = cursor[c] as usize;
        data[pos] = coo.data[k];
        row_ind[pos] = coo.row_ind[k];
        cursor[c] += 1;
    }

    csc_array {
        data,
        row_ind,
        col_ptr,
        shape: coo.shape,
    }
}

// ─── 稀疏矩阵 × 向量乘法 ────────────────────────────────────────────

/// 计算 `v_new = v_old @ matrix`（COO 格式的稀疏矩阵 × 密集向量乘法）
pub fn coo_matvec(v_old: &[f64], mat: &coo_array, v_new: &mut [f64]) {
    v_new.fill(0.0);
    for k in 0..mat.data.len() {
        let row = mat.row_ind[k] as usize;
        let col = mat.col_ind[k] as usize;
        v_new[col] += mat.data[k] * v_old[row];
    }
}

/// 计算 `v_new = v_old @ csr`（行向量 × CSR 矩阵）
/// 遍历所有非零元: result[col] += v_old[row] * data[k]
pub fn csr_matvec(v_old: &[f64], mat: &csr_array, v_new: &mut [f64]) {
    v_new.fill(0.0);
    for row in 0..mat.shape.0 {
        let row_start = mat.row_ptr[row] as usize;
        let row_end = mat.row_ptr[row + 1] as usize;
        let v_row = v_old[row];
        if v_row == 0.0 {
            continue;
        }
        for k in row_start..row_end {
            let col = mat.col_ind[k] as usize;
            v_new[col] += mat.data[k] * v_row;
        }
    }
}

/// 计算 `v_new = v_old @ csc`（行向量 × CSC 矩阵）
/// 逐列计算: result[j] = sum_{k∈col_ptr[j]..col_ptr[j+1]} data[k] * v_old[row_ind[k]]
pub fn csc_matvec(v_old: &[f64], mat: &csc_array, v_new: &mut [f64]) {
    v_new.fill(0.0);
    for col in 0..mat.shape.1 {
        let col_start = mat.col_ptr[col] as usize;
        let col_end = mat.col_ptr[col + 1] as usize;
        let mut sum = 0.0;
        for k in col_start..col_end {
            let row = mat.row_ind[k] as usize;
            sum += mat.data[k] * v_old[row];
        }
        v_new[col] = sum;
    }
}

/// 计算 `v_new = csr @ v_old`（CSR 矩阵 × 列向量）
/// 逐行计算: result[row] = sum_{k∈row_ptr[row]..row_ptr[row+1]} data[k] * v_old[col_ind[k]]
pub fn csr_matvec_col(v_old: &[f64], mat: &csr_array, v_new: &mut [f64]) {
    v_new.fill(0.0);
    for row in 0..mat.shape.0 {
        let row_start = mat.row_ptr[row] as usize;
        let row_end = mat.row_ptr[row + 1] as usize;
        let mut sum = 0.0;
        for k in row_start..row_end {
            let col = mat.col_ind[k] as usize;
            sum += mat.data[k] * v_old[col];
        }
        v_new[row] = sum;
    }
}

/// 计算 `v_new = csc @ v_old`（CSC 矩阵 × 列向量）
/// 遍历所有非零元: result[row] += data[k] * v_old[col]
pub fn csc_matvec_col(v_old: &[f64], mat: &csc_array, v_new: &mut [f64]) {
    v_new.fill(0.0);
    for col in 0..mat.shape.1 {
        let col_start = mat.col_ptr[col] as usize;
        let col_end = mat.col_ptr[col + 1] as usize;
        let v_col = v_old[col];
        if v_col == 0.0 {
            continue;
        }
        for k in col_start..col_end {
            let row = mat.row_ind[k] as usize;
            v_new[row] += mat.data[k] * v_col;
        }
    }
}
