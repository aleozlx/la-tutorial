struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>
}

impl Matrix {
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn s_product_sum(&mut self, a: &Matrix, b: &Matrix, j: usize) {
        let s_view = &b.data[j*b.rows..(j+1)*b.rows];
        // grab the resulting column
        let out_view = &mut self.data[j*self.rows..(j+1)*self.rows];
        // reset output space to prepare for accumulation
        for p in 0..self.rows {
            out_view[p] = 0.0;
        }

        for k in 0..a.cols {
            let scalar = s_view[k];
            let in_view = &a.data[k*a.rows..(k+1)*a.rows];
            for p in 0..self.rows {
                // the following can be accelerated using FMA
                out_view[p] = scalar.mul_add(in_view[p], out_view[p]);
            }
        }
    }

    fn matmul(&mut self, a: &Matrix, b: &Matrix) {
        for j in 0..b.cols {
            unsafe{ self.s_product_sum(a, b, j); }
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
            unsafe{ x[j] = (aug.data[(j+1)*aug.cols-1] - Matrix::dot_unsafe(&aug.data[j*aug.cols+j+1..], &x[j+1..], aug.rows-1-j)) / aug.data[j*aug.cols+j]; }
        }
    }
}

fn main() {
    let a = Matrix { rows: 2, cols: 2, data: vec![2.0, 0.1, 0.3, 3.0] };
    let b = Matrix { rows: 2, cols: 2, data: vec![1.0, 0.5, 0.5, 1.0] };
    let mut c = Matrix { rows: 2, cols: 2, data: vec![0.0; 4] };
    c.matmul(&a, &b);
    println!("{:?}", c.data);

    let a = Matrix { rows: 3, cols: 3, data: vec![1.0, 4.0, 2.0, 2.0, 3.0, 2.0, -1.0, 1.0, 3.0] };
    let b = vec![2.0, 3.0, 5.0];
    let mut x = vec![0.0; 3];
    a.solve(&b, &mut x);
    println!("{:?}", x);
}
