struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>
}

impl Matrix {
    // #[no_mangle]
    // #[inline(never)]
    #[target_feature(enable = "avx")]
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
            // the following can be accelerated using AVX
            for p in 0..self.rows {
                out_view[p] += scalar * in_view[p];
            }
        }
    }

    fn matmul(&mut self, a: &Matrix, b: &Matrix) {
        for j in 0..b.cols {
            unsafe{ self.s_product_sum(a, b, j); }
        }
    }
}

fn main() {
    let a = Matrix { rows: 2, cols: 2, data: vec![2.0, 0.1, 0.3, 3.0] };
    let b = Matrix { rows: 2, cols: 2, data: vec![1.0, 0.5, 0.5, 1.0] };
    let mut c = Matrix { rows: 2, cols: 2, data: vec![0.0; 4] };
    c.matmul(&a, &b);
    println!("{:?}", c.data);
}
