#![allow(nonstandard_style)]

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul};

pub struct coo_array<T, U = usize, V = usize> {
    pub data: Vec<T>,
    pub row_ind: Vec<U>,
    pub col_ind: Vec<V>,
    pub shape: (usize, usize),
}

pub struct csr_array<T, U = usize, V = usize> {
    pub data: Vec<T>,
    pub col_ind: Vec<U>,
    pub row_ptr: Vec<V>,
    pub shape: (usize, usize),
}

pub struct csc_array<T, U = usize, V = usize> {
    pub data: Vec<T>,
    pub row_ind: Vec<U>,
    pub col_ptr: Vec<V>,
    pub shape: (usize, usize),
}

/// 将 COO 格式转换为 CSR 格式
pub fn coo_to_csr<T, U0, V0, U1, V1>(coo: &coo_array<T, U0, V0>) -> csr_array<T, U1, V1>
where
    T: Default + Copy,
    U0: Copy + TryInto<usize>,
    V0: Copy + TryInto<U1>,
    U1: Copy + Default,
    V1: Copy + Add<Output = V1> + AddAssign + Default + From<u8> + TryInto<usize>,
    <U0 as TryInto<usize>>::Error: Debug,
    <V0 as TryInto<U1>>::Error: Debug,
    <V1 as TryInto<usize>>::Error: Debug,
{
    let (n_rows, _n_cols) = coo.shape;
    let nnz = coo.data.len();

    // 统计每行的非零元素数量
    let mut row_count = vec![V1::default(); n_rows];
    for &r in &coo.row_ind {
        row_count[r.try_into().unwrap()] += 1.into();
    }

    // 构建 row_ptr: 前缀和
    let mut row_ptr = vec![V1::default(); n_rows + 1];
    for i in 0..n_rows {
        row_ptr[i + 1] = row_ptr[i] + row_count[i];
    }

    // 按行填入 data 和 col_ind
    let mut data = vec![T::default(); nnz];
    let mut col_ind = vec![U1::default(); nnz];
    // 临时记录每行当前写入位置
    let mut cursor = row_ptr[..n_rows].to_vec();

    for k in 0..nnz {
        let r = coo.row_ind[k].try_into().unwrap();
        let pos = cursor[r].try_into().unwrap();
        data[pos] = coo.data[k];
        col_ind[pos] = coo.col_ind[k].try_into().unwrap();
        cursor[r] += 1.into();
    }

    csr_array {
        data,
        col_ind,
        row_ptr,
        shape: coo.shape,
    }
}

/// 将 COO 格式转换为 CSC 格式
pub fn coo_to_csc<T, U0, V0, U1, V1>(coo: &coo_array<T, U0, V0>) -> csc_array<T, U1, V1>
where
    T: Default + Copy,
    U0: Copy + TryInto<U1>,
    V0: Copy + TryInto<usize>,
    U1: Copy + Default,
    V1: Copy + Add<Output = V1> + AddAssign + Default + From<u8> + TryInto<usize>,
    <U0 as TryInto<U1>>::Error: Debug,
    <V0 as TryInto<usize>>::Error: Debug,
    <V1 as TryInto<usize>>::Error: Debug,
{
    let (_n_rows, n_cols) = coo.shape;
    let nnz = coo.data.len();

    // 统计每列的非零元素数量
    let mut col_count = vec![V1::default(); n_cols];
    for &c in &coo.col_ind {
        col_count[c.try_into().unwrap()] += 1.into();
    }

    // 构建 col_ptr: 前缀和
    let mut col_ptr = vec![V1::default(); n_cols + 1];
    for j in 0..n_cols {
        col_ptr[j + 1] = col_ptr[j] + col_count[j];
    }

    // 按列填入 data 和 row_ind
    let mut data = vec![T::default(); nnz];
    let mut row_ind = vec![U1::default(); nnz];
    let mut cursor = col_ptr[..n_cols].to_vec();

    for k in 0..nnz {
        let c = coo.col_ind[k].try_into().unwrap();
        let pos = cursor[c].try_into().unwrap();
        data[pos] = coo.data[k];
        row_ind[pos] = coo.row_ind[k].try_into().unwrap();
        cursor[c] += 1.into();
    }

    csc_array {
        data,
        row_ind,
        col_ptr,
        shape: coo.shape,
    }
}

/// 计算 `v_new = v_old @ matrix`（COO 格式的稀疏矩阵 × 密集向量乘法）
pub fn coo_matvec<TV, TA, TR, U, V>(v_old: &[TV], mat: &coo_array<TA, U, V>, v_new: &mut [TR])
where
    TV: Mul<TA, Output = TR> + Default + Copy,
    TA: Copy,
    TR: Default + Copy + AddAssign,
    U: Copy + TryInto<usize>,
    V: Copy + TryInto<usize>,
    <U as TryInto<usize>>::Error: Debug,
    <V as TryInto<usize>>::Error: Debug,
{
    v_new.fill(TR::default());
    for k in 0..mat.data.len() {
        let row = mat.row_ind[k].try_into().unwrap();
        let col = mat.col_ind[k].try_into().unwrap();
        v_new[col] += v_old[row] * mat.data[k];
    }
}

/// 计算 `v_new = v_old @ csr`（行向量 × CSR 矩阵）
pub fn csr_matvec<TV, TA, TR, U, V>(v_old: &[TV], mat: &csr_array<TA, U, V>, v_new: &mut [TR])
where
    TV: Mul<TA, Output = TR> + Default + Copy + PartialEq<TR>,
    TA: Copy,
    TR: Default + Copy + AddAssign,
    U: Copy + TryInto<usize>,
    V: Copy + TryInto<usize>,
    <U as TryInto<usize>>::Error: Debug,
    <V as TryInto<usize>>::Error: Debug,
{
    v_new.fill(TR::default());
    for row in 0..mat.shape.0 {
        let row_start = mat.row_ptr[row].try_into().unwrap();
        let row_end = mat.row_ptr[row + 1].try_into().unwrap();
        let v_row = v_old[row];
        if v_row == TR::default() {
            continue;
        }
        for k in row_start..row_end {
            let col = mat.col_ind[k].try_into().unwrap();
            v_new[col] += v_row * mat.data[k];
        }
    }
}

/// 计算 `v_new = v_old @ csc`（行向量 × CSC 矩阵）
pub fn csc_matvec<TV, TA, TR, U, V>(v_old: &[TV], mat: &csc_array<TA, U, V>, v_new: &mut [TR])
where
    TV: Mul<TA, Output = TR> + Default + Copy,
    TA: Copy,
    TR: Default + Copy + AddAssign,
    U: Copy + TryInto<usize>,
    V: Copy + TryInto<usize>,
    <U as TryInto<usize>>::Error: Debug,
    <V as TryInto<usize>>::Error: Debug,
{
    v_new.fill(TR::default());
    for col in 0..mat.shape.1 {
        let col_start = mat.col_ptr[col].try_into().unwrap();
        let col_end = mat.col_ptr[col + 1].try_into().unwrap();
        let mut sum = TR::default();
        for k in col_start..col_end {
            let row = mat.row_ind[k].try_into().unwrap();
            sum += v_old[row] * mat.data[k];
        }
        v_new[col] = sum;
    }
}

/// 计算 `v_new = csr @ v_old`（CSR 矩阵 × 列向量）
pub fn csr_matvec_col<TV, TA, TR, U, V>(v_old: &[TV], mat: &csr_array<TA, U, V>, v_new: &mut [TR])
where
    TV: Default + Copy,
    TA: Copy + Mul<TV, Output = TR>,
    TR: Default + Copy + AddAssign,
    U: Copy + TryInto<usize>,
    V: Copy + TryInto<usize>,
    <U as TryInto<usize>>::Error: Debug,
    <V as TryInto<usize>>::Error: Debug,
{
    v_new.fill(TR::default());
    for row in 0..mat.shape.0 {
        let row_start = mat.row_ptr[row].try_into().unwrap();
        let row_end = mat.row_ptr[row + 1].try_into().unwrap();
        let mut sum = TR::default();
        for k in row_start..row_end {
            let col = mat.col_ind[k].try_into().unwrap();
            sum += mat.data[k] * v_old[col];
        }
        v_new[row] = sum;
    }
}

/// 计算 `v_new = csc @ v_old`（CSC 矩阵 × 列向量）
pub fn csc_matvec_col<TV, TA, TR, U, V>(v_old: &[TV], mat: &csc_array<TA, U, V>, v_new: &mut [TR])
where
    TV: Default + Copy + PartialEq<TR>,
    TA: Copy + Mul<TV, Output = TR>,
    TR: Default + Copy + AddAssign,
    U: Copy + TryInto<usize>,
    V: Copy + TryInto<usize>,
    <U as TryInto<usize>>::Error: Debug,
    <V as TryInto<usize>>::Error: Debug,
{
    v_new.fill(TR::default());
    for col in 0..mat.shape.1 {
        let col_start = mat.col_ptr[col].try_into().unwrap();
        let col_end = mat.col_ptr[col + 1].try_into().unwrap();
        let v_col = v_old[col];
        if v_col == TR::default() {
            continue;
        }
        for k in col_start..col_end {
            let row = mat.row_ind[k].try_into().unwrap();
            v_new[row] += mat.data[k] * v_col;
        }
    }
}
