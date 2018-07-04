struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>
}

trait IColumnVector {
    fn len(&self) -> usize;
    fn get_view(&self, j: usize) -> &[f32];
    fn get_view_mut(&mut self, j: usize) -> &mut [f32];
}

impl IColumnVector for Matrix {
    fn len(&self) -> usize {
        self.rows
    }

    fn get_view(&self, j: usize) -> &[f32] {
        &self.data[j*self.rows..(j+1)*self.rows]
    }

    fn get_view_mut(&mut self, j: usize) -> &mut [f32] {
        &mut self.data[j*self.rows..(j+1)*self.rows]
    }
}

impl IColumnVector for Vec<f32> {
    fn len(&self) -> usize {
        self.len()
    }

    fn get_view(&self, _j: usize) -> &[f32] {
        &self[..]
    }

    fn get_view_mut(&mut self, _j: usize) -> &mut [f32] {
        &mut self[..]
    }
}

impl<'a> IColumnVector for &'a [f32] {
    fn len(&self) -> usize {
        self[..].len()
    }

    fn get_view(&self, _j: usize) -> &[f32] {
        self
    }

    fn get_view_mut(&mut self, _j: usize) -> &mut [f32] {
        unimplemented!();
    }
}

impl<'a, 'b> std::ops::Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &'b Matrix) -> Matrix {
        let mut ret = Matrix { rows: self.rows, cols: rhs.cols, data: vec![0.0; self.rows*rhs.cols] };
        ret.matmul(self, rhs);
        return ret;
    }
}

impl Matrix {
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn s_product_sum(out: &mut IColumnVector, a: &Matrix, b: &IColumnVector, j: usize) {
        let n = out.len();
        let s_view = b.get_view(j);
        let out_view = out.get_view_mut(j);
        // reset output space to prepare for accumulation
        for p in 0..n {
            out_view[p] = 0.0;
        }

        for k in 0..a.cols {
            let scalar = s_view[k];
            let in_view = &a.data[k*a.rows..(k+1)*a.rows];
            for p in 0..n {
                // the following can be accelerated using FMA
                out_view[p] = scalar.mul_add(in_view[p], out_view[p]);
            }
        }
    }

    fn matmul(&mut self, a: &Matrix, b: &Matrix) {
        for j in 0..b.cols {
            unsafe{
                Matrix::s_product_sum(self, a, b, j);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn row_oplu(&mut self, j: usize){
        // ref: http://www.personal.psu.edu/jhm/f90/lectures/lu.html
        let pivot = self.data[j*self.cols+j];
        for i in j+1..self.rows {
            // cast ref to ptr then back to ref so we can get mutable and immutable views at once
            let in_view = & *(self.data.get_unchecked(j*self.cols..(j+1)*self.cols) as *const [f32]);
            let out_view = &mut *(self.data.get_unchecked_mut(i*self.cols..(i+1)*self.cols) as *mut [f32]);
            let s = -(out_view[j] / pivot);
            for k in j..self.cols {
                out_view[k] = s.mul_add(in_view[k], out_view[k]);
            }
            out_view[j] = -s;
        }
    }

    #[target_feature(enable = "fma")]
    unsafe fn dot_unsafe(a: &[f32], b: &[f32], len: usize) -> f32 {
        let mut s = 0.0f32;
        for i in 0..len {
            s = a[i].mul_add(b[i], s);
        }
        return s;
    }

    fn dot_vecs(a: &IColumnVector, b: &IColumnVector) -> f32 {
        unsafe { Matrix::dot_unsafe(a.get_view(0), b.get_view(0), std::cmp::min(a.len(), b.len())) }
    }

    fn solve(&self, b: &Vec<f32>, x: &mut Vec<f32>){
        // create an augmented matrix in row major order
        let mut aug = Matrix { rows: self.rows, cols: self.cols+1, data: Vec::with_capacity(self.rows * (self.cols+1)) };
        for i in 0..aug.rows {
            let mut p = i;
            for _j in 0..self.cols {
                aug.data.push(self.data[p]);
                p += self.rows;
            }
            aug.data.push(b[i]);
        }
        // LU decomposition
        for j in 0..aug.rows {
            unsafe{ aug.row_oplu(j); }
        }
        // back substitution
        for j in (0..aug.rows).rev() {
            x[j] = (aug.data[(j+1)*aug.cols-1] - Matrix::dot_vecs(&&aug.data[j*aug.cols+j+1..], &&x[j+1..])) / aug.data[j*aug.cols+j];
        }
    }

    fn transpose(&self) -> Matrix {
        let mut transpose = Matrix { rows: self.cols, cols: self.rows, data: Vec::with_capacity(self.data.len()) };
        for j in 0..transpose.cols {
            let mut p = j;
            for _i in 0..transpose.rows {
                transpose.data.push(self.data[p]);
                p += self.cols;
            }
        }
        return transpose;
    }

    fn linear_regression(&self, y: &Vec<f32>, theta: &mut Vec<f32>){
        let transpose = self.transpose();
        let a = &transpose * self;
        let mut b = vec![0.0; self.cols];
        unsafe { Matrix::s_product_sum(&mut b, &transpose, y, 0); }
        a.solve(&b, theta);
    }
}

fn main() {
    let a = Matrix { rows: 2, cols: 2, data: vec![2.0, 0.1, 0.3, 3.0] };
    let b = Matrix { rows: 2, cols: 2, data: vec![1.0, 0.5, 0.5, 1.0] };
    let c = &a * &b;
    println!("{:?}", c.data);

    let a = Matrix { rows: 3, cols: 3, data: vec![1.0, 4.0, 2.0, 2.0, 3.0, 2.0, -1.0, 1.0, 3.0] };
    let b = vec![2.0, 3.0, 5.0];
    let mut x = vec![0.0; 3];
    a.solve(&b, &mut x);
    println!("{:?}", x);
}
