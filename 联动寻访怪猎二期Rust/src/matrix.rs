#![allow(nonstandard_style)]

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul};

pub struct CooArray<T, U = usize, V = usize> {
    pub data: Vec<T>,
    pub row_ind: Vec<U>,
    pub col_ind: Vec<V>,
    pub shape: (usize, usize),
}

pub struct CsrArray<T, U = usize, V = usize> {
    pub data: Vec<T>,
    pub col_ind: Vec<U>,
    pub row_ptr: Vec<V>,
    pub shape: (usize, usize),
}

pub struct CscArray<T, U = usize, V = usize> {
    pub data: Vec<T>,
    pub row_ind: Vec<U>,
    pub col_ptr: Vec<V>,
    pub shape: (usize, usize),
}

/// 将 COO 格式转换为 CSR 格式
pub fn coo_to_csr<T, U0, V0, U1, V1>(coo: &CooArray<T, U0, V0>) -> CsrArray<T, U1, V1>
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
    for &row in &coo.row_ind {
        row_count[row.try_into().unwrap()] += 1.into();
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
        let row = coo.row_ind[k].try_into().unwrap();
        let pos = cursor[row].try_into().unwrap();
        data[pos] = coo.data[k];
        col_ind[pos] = coo.col_ind[k].try_into().unwrap();
        cursor[row] += 1.into();
    }

    CsrArray {
        data,
        col_ind,
        row_ptr,
        shape: coo.shape,
    }
}

/// 将 COO 格式转换为 CSC 格式
pub fn coo_to_csc<T, U0, V0, U1, V1>(coo: &CooArray<T, U0, V0>) -> CscArray<T, U1, V1>
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
        let col = coo.col_ind[k].try_into().unwrap();
        let pos = cursor[col].try_into().unwrap();
        data[pos] = coo.data[k];
        row_ind[pos] = coo.row_ind[k].try_into().unwrap();
        cursor[col] += 1.into();
    }

    CscArray {
        data,
        row_ind,
        col_ptr,
        shape: coo.shape,
    }
}

/// 计算 `v_new = v_old @ coo_array`
pub fn vec_mul_coo_array<TV, TA, TR, U, V>(
    vec_old: &[TV],
    coo_array: &CooArray<TA, U, V>,
    vec_new: &mut [TR],
) where
    TV: Mul<TA, Output = TR> + Default + Copy,
    TA: Copy,
    TR: Default + Copy + AddAssign,
    U: Copy + TryInto<usize>,
    V: Copy + TryInto<usize>,
    <U as TryInto<usize>>::Error: Debug,
    <V as TryInto<usize>>::Error: Debug,
{
    vec_new.fill(TR::default());
    for k in 0..coo_array.data.len() {
        let row = coo_array.row_ind[k].try_into().unwrap();
        let col = coo_array.col_ind[k].try_into().unwrap();
        vec_new[col] += vec_old[row] * coo_array.data[k];
    }
}

/// 计算 `v_new = v_old @ csr_array`
pub fn vec_mul_csr_array<TV, TA, TR, U, V>(
    vec_old: &[TV],
    csr_array: &CsrArray<TA, U, V>,
    vec_new: &mut [TR],
) where
    TV: Mul<TA, Output = TR> + Default + Copy + PartialEq<TR>,
    TA: Copy,
    TR: Default + Copy + AddAssign,
    U: Copy + TryInto<usize>,
    V: Copy + TryInto<usize>,
    <U as TryInto<usize>>::Error: Debug,
    <V as TryInto<usize>>::Error: Debug,
{
    vec_new.fill(TR::default());
    for row in 0..csr_array.shape.0 {
        let row_start = csr_array.row_ptr[row].try_into().unwrap();
        let row_end = csr_array.row_ptr[row + 1].try_into().unwrap();
        let v_row = vec_old[row];
        if v_row == TR::default() {
            continue;
        }
        for k in row_start..row_end {
            let col = csr_array.col_ind[k].try_into().unwrap();
            vec_new[col] += v_row * csr_array.data[k];
        }
    }
}

/// 计算 `v_new = v_old @ csc_array`
pub fn vec_mul_csc_array<TV, TA, TR, U, V>(
    vec_old: &[TV],
    csc_array: &CscArray<TA, U, V>,
    vec_new: &mut [TR],
) where
    TV: Mul<TA, Output = TR> + Default + Copy,
    TA: Copy,
    TR: Default + Copy + AddAssign,
    U: Copy + TryInto<usize>,
    V: Copy + TryInto<usize>,
    <U as TryInto<usize>>::Error: Debug,
    <V as TryInto<usize>>::Error: Debug,
{
    vec_new.fill(TR::default());
    for col in 0..csc_array.shape.1 {
        let col_start = csc_array.col_ptr[col].try_into().unwrap();
        let col_end = csc_array.col_ptr[col + 1].try_into().unwrap();
        let mut sum = TR::default();
        for k in col_start..col_end {
            let row = csc_array.row_ind[k].try_into().unwrap();
            sum += vec_old[row] * csc_array.data[k];
        }
        vec_new[col] = sum;
    }
}

/// 计算 `v_new = csr_array @ v_old`
pub fn csr_array_mul_vec<TV, TA, TR, U, V>(
    vec_old: &[TV],
    csr_array: &CsrArray<TA, U, V>,
    vec_new: &mut [TR],
) where
    TV: Default + Copy,
    TA: Copy + Mul<TV, Output = TR>,
    TR: Default + Copy + AddAssign,
    U: Copy + TryInto<usize>,
    V: Copy + TryInto<usize>,
    <U as TryInto<usize>>::Error: Debug,
    <V as TryInto<usize>>::Error: Debug,
{
    vec_new.fill(TR::default());
    for row in 0..csr_array.shape.0 {
        let row_start = csr_array.row_ptr[row].try_into().unwrap();
        let row_end = csr_array.row_ptr[row + 1].try_into().unwrap();
        let mut sum = TR::default();
        for k in row_start..row_end {
            let col = csr_array.col_ind[k].try_into().unwrap();
            sum += csr_array.data[k] * vec_old[col];
        }
        vec_new[row] = sum;
    }
}

/// 计算 `v_new = csc_array @ v_old`
pub fn csc_array_mul_vec<TV, TA, TR, U, V>(
    vec_old: &[TV],
    csc_array: &CscArray<TA, U, V>,
    vec_new: &mut [TR],
) where
    TV: Default + Copy + PartialEq<TR>,
    TA: Copy + Mul<TV, Output = TR>,
    TR: Default + Copy + AddAssign,
    U: Copy + TryInto<usize>,
    V: Copy + TryInto<usize>,
    <U as TryInto<usize>>::Error: Debug,
    <V as TryInto<usize>>::Error: Debug,
{
    vec_new.fill(TR::default());
    for col in 0..csc_array.shape.1 {
        let col_start = csc_array.col_ptr[col].try_into().unwrap();
        let col_end = csc_array.col_ptr[col + 1].try_into().unwrap();
        let v_col = vec_old[col];
        if v_col == TR::default() {
            continue;
        }
        for k in col_start..col_end {
            let row = csc_array.row_ind[k].try_into().unwrap();
            vec_new[row] += csc_array.data[k] * v_col;
        }
    }
}
