use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

/// Python module implemented in Rust.
#[pymodule]
fn proxy_crm(_py: Python, m: &PyModule) -> PyResult<()> {
    fn q_prim(prod: ArrayView2<'_, f64>, time: ArrayView1<'_, f64>, lambda_prod: ArrayView1<'_, f64>, tau_prim: ArrayView1<'_, f64>) -> Array1<f64> {
        let n_prod: usize = prod.raw_dim()[1];
        let mut result: Array1<f64> = Array1::zeros([n_prod]);

        for j in 0..n_prod {
            let time_decay = (-&time / tau_prim[j]).mapv(f64::exp);
            result[j] = time_decay[j] * prod[[0,j]] * lambda_prod[j]
        }
        result
    }

    fn q_crm(inj: ArrayView2<'_, f64>, time: ArrayView1<'_, f64>, lambda_ip: ArrayView3<'_, f64>, tau: ArrayView1<'_, f64>, mask: ArrayView2<'_, f64> ) -> Array3<f64> {
        let n_t: usize = time.raw_dim()[0];
        let n_inj: usize = lambda_ip.raw_dim()[2];
        let n_prod: usize = mask.raw_dim()[1];
        let mut convolve: Array3<f64> = Array3::zeros([n_t, n_prod, n_inj]);
        
        let lambda_ip_mod = calc_sh_mask(lambda_ip, mask);

        for i in 0..n_inj {
            for j in 0..n_prod {
                convolve[[0,j,i]] = (1.0 - ((time[0] - time[1]) / tau[j]).exp()) * lambda_ip_mod[[0,j,i]] * ((inj[[1,i]] - inj[[0,i]]) / (time[1]-time[0])) * (1.0 - mask[[0,j]]);
                for m in 1..n_t {
                    for n in 1..m+1 {
                        let time_decay = (1.0 - ((time[m-1] - time[m]) / tau[j]).exp()) * ((time[n] - time[m]) / tau[j]).exp();

                        convolve[[m,j,i]] += time_decay * lambda_ip_mod[[m,j,i]] * ((inj[[m-1,i]] - inj[[m,i]]) / (time[m]-time[m-1]));
                    }
                }
            }
        }
        convolve
    }

    fn q_crm_fixed(inj: ArrayView2<'_, f64>, time: ArrayView1<'_, f64>, lambda_ip: ArrayView3<'_, f64>, tau: ArrayView1<'_, f64>, mask: ArrayView2<'_, f64> ) -> Array3<f64> {
        let n_t: usize = time.raw_dim()[0];
        let n_inj: usize = lambda_ip.raw_dim()[2];
        let n_prod: usize = mask.raw_dim()[1];
        let mut convolve: Array3<f64> = Array3::zeros([n_t, n_prod, n_inj]);
        
        let lambda_ip_mod = calc_sh_mask(lambda_ip, mask);

        for i in 0..n_inj {
            for j in 0..n_prod {
                convolve[[0,j,i]] = (1.0 - ((time[0] - time[1]) / tau[j]).exp()) * lambda_ip_mod[[0,j,i]] * inj[[0,i]] * (1.0 - mask[[0,j]]);
                for m in 1..n_t {
                    for n in 1..m+1 {
                        let time_decay = (1.0 - ((time[m-1] - time[m]) / tau[j]).exp()) * ((time[n] - time[m]) / tau[j]).exp();

                        convolve[[m,j,i]] += time_decay * lambda_ip_mod[[m,j,i]] * inj[[m,i]];
                    }
                }
            }
        }
        convolve
    }

    fn q_bhp(time: ArrayView1<'_, f64>, tau: ArrayView1<'_, f64>, press: ArrayView2<'_, f64>, prod_index: ArrayView1<'_, f64>, mask:ArrayView2<'_, f64>) -> Array2<f64> {
        let n_t = time.raw_dim()[0];
        let n_prod = press.raw_dim()[1];
        let mut convolve: Array2<f64> = Array2::zeros([n_t,n_prod]);
        
        for j in 0..n_prod {
            convolve[[0,j]] = (1.0 - ((time[0] - time[1]) / tau[j]).exp()) * prod_index[j] * tau[j] * (press[[1,j]]-press[[0,j]]) / (time[1]-time[0]) * (1.0-mask[[0,j]]);
            for m in 1..n_t {
                for n in 1..m+1 {
                    let time_decay = (1.0 - ((time[m-1] - time[m]) / tau[j]).exp()) * ((time[n] - time[m]) / tau[j]).exp();
                    let delta_bhp = press[[m-1,j]] - press[[m,j]];

                    convolve[[m,j]] += time_decay * prod_index[j] * tau[j] * (delta_bhp / (time[m]-time[m-1])) * (1.0 - mask[[m,j]]);
                }
            }
        }
        convolve
    }

    fn sh_mask(prod: ArrayView2<'_, f64>) -> Array2<f64> {
        let n_prod: usize = prod.raw_dim()[1];
        let n_t: usize = prod.raw_dim()[0];
        let mut mask = Array2::zeros([n_t, n_prod]);

        for t in 0..n_t {
            for j in 0..n_prod {
                if prod[[t,j]] == 0.0 {
                    mask[[t,j]] = 1.0;
                }
            }
        }
        mask
    }

    fn calc_sh_mask(lambda_ip: ArrayView3<'_, f64>, mask: ArrayView2<'_, f64>) -> Array3<f64> {
        let n_prod: usize = mask.raw_dim()[1];
        let n_inj: usize = lambda_ip.raw_dim()[2];
        let n_t: usize = mask.raw_dim()[0];
        
        let mut lambda_ip_result: Array3<f64> = Array3::zeros([n_t,n_prod,n_inj]);
        let temp = sum_lambda_x(lambda_ip, mask);

        let sum_lambda_sh = temp.0;
        let t_sh = temp.1;

        for t in 0..n_t {
            for j in 0..n_prod {
                for i in 0..n_inj {
                    lambda_ip_result[[t,j,i]] = lambda_ip[[t,j,i]] * (1.0 - mask[[t,j]]);
                }
            }
        }
        for t in 0..n_t {
            if t_sh.contains(&t) {
                for j in 0..n_prod {
                    for i in 0..n_inj {
                        if mask[[t,j]] == 0.0 {
                            lambda_ip_result[[t, j, i]] = f64::abs(lambda_ip_result[[t,j,i]]) * (1.0 + sum_lambda_sh[[t,i]]);
                        }
                    }
                }
            }
        }
        lambda_ip_result
    }

    fn sum_lambda_x (lambda_ip: ArrayView3<'_, f64>, mask: ArrayView2<'_, f64>) -> (Array2<f64>, Vec<usize>) {
        let n_prod: usize = mask.raw_dim()[1];
        let n_inj: usize = lambda_ip.raw_dim()[2];
        let n_t: usize = mask.raw_dim()[0];

        let mut t_sh: Vec<usize> = Vec::new();

        let mut sum_lambda_sh: Array3<f64> = Array3::zeros([n_t, n_prod,n_inj]);

        for t in 0..n_t {
            for j in 0..n_prod {
                for i in 0..n_inj {
                    if mask[[t,j]] == 1.0 {
                        sum_lambda_sh[[t,j,i]] = lambda_ip[[t,j,i]];
                        t_sh.push(t);
                    }
                }
            }
        }
        (sum_lambda_sh.sum_axis(Axis(1)), t_sh)
    }

    fn objective_add (rate_prim: ArrayView1<'_, f64>, rate_crm: ArrayView2<'_, f64>, rate_bhp: ArrayView2<'_, f64>) -> Array2<f64> {
        let n_prod = rate_crm.raw_dim()[1];
        let n_t = rate_crm.raw_dim()[0];
        let mut q_hat: Array2<f64> = Array2::zeros([n_t,n_prod]);

        for t in 0..n_t {
            for j in 0..n_prod {
                q_hat[[0,j]] += rate_prim[j];
                q_hat[[t,j]] += rate_crm[[t,j]];
                q_hat[[t,j]] += rate_bhp[[t,j]];
            }
        }
        q_hat
    }

    fn eq_loop(alpha: ArrayView1<'_, f64>, beta: ArrayView1<'_, f64>, lambda_ip: ArrayView2<'_, f64>, inj: ArrayView2<'_, f64>) -> Array2<f64> {
        let n_prod: usize = alpha.raw_dim()[0];
        let n_inj: usize = inj.raw_dim()[1];
        let n_t: usize = inj.raw_dim()[0];

        let mut f_oil: Array2<f64> = Array2::zeros([n_t, n_prod]);

        for t in 0..n_t {
            for j in 0..n_prod {
                for i in 0..n_inj {
                    f_oil[[t,j]] += 1.0 / (1.0 + alpha[j] * ((lambda_ip[[j,i]] * inj[[t,i]]).powf(beta[j])));
                }
            }
        }
        f_oil
    }

    // wrapper
    #[pyfn(m)]
    #[pyo3(name = "q_prim")]
    fn q_prim_py<'py>(py: Python<'py>, prod: PyReadonlyArray2<'_, f64>, time: PyReadonlyArray1<'_, f64>, lambda_prod: PyReadonlyArray1<'_, f64>, tau_prim: PyReadonlyArray1<'_, f64>) -> &'py PyArray1<f64> {
        let prod = prod.as_array();
        let time = time.as_array();
        let lambda_prod = lambda_prod.as_array();
        let tau_prim = tau_prim.as_array();

        let q = q_prim(prod, time, lambda_prod, tau_prim);
        
        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "q_crm")]
    fn q_crm_py<'py>(py: Python<'py>, inj: PyReadonlyArray2<'_, f64>, time: PyReadonlyArray1<'_, f64>, lambda_ip: PyReadonlyArray3<'_, f64>, tau: PyReadonlyArray1<'_, f64>, mask: PyReadonlyArray2<'_, f64>) -> &'py PyArray3<f64> {
        let inj = inj.as_array();
        let time = time.as_array();
        let lambda_ip = lambda_ip.as_array();
        let tau = tau.as_array();
        let mask = mask.as_array();

        let q = q_crm(inj, time, lambda_ip, tau, mask);

        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "q_crm_fixed")]
    fn q_crm_fixed_py<'py>(py: Python<'py>, inj: PyReadonlyArray2<'_, f64>, time: PyReadonlyArray1<'_, f64>, lambda_ip: PyReadonlyArray3<'_, f64>, tau: PyReadonlyArray1<'_, f64>, mask: PyReadonlyArray2<'_, f64>) -> &'py PyArray3<f64> {
        let inj = inj.as_array();
        let time = time.as_array();
        let lambda_ip = lambda_ip.as_array();
        let tau = tau.as_array();
        let mask = mask.as_array();

        let q = q_crm_fixed(inj, time, lambda_ip, tau, mask);

        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "q_bhp")]
    fn q_bhp_py<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, tau: PyReadonlyArray1<'_, f64>, press: PyReadonlyArray2<'_, f64>, prod_index: PyReadonlyArray1<'_, f64>, mask: PyReadonlyArray2<'_, f64>) -> &'py PyArray2<f64> {
        let time = time.as_array();
        let tau = tau.as_array();
        let press = press.as_array();
        let prod_index = prod_index.as_array();
        let mask = mask.as_array();

        let q = q_bhp(time, tau, press, prod_index, mask);

        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "sh_mask")]
    fn sh_mask_py<'py>(py: Python<'py>, prod: PyReadonlyArray2<f64>) -> &'py PyArray2<f64> {
        let prod = prod.as_array();

        let shut_in = sh_mask(prod);

        shut_in.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "calc_sh_mask")]
    fn calc_sh_mask_py<'py>(py: Python<'py>, lambda_ip: PyReadonlyArray3<'_, f64>, mask: PyReadonlyArray2<'_, f64>) -> &'py PyArray3<f64> {
        let lambda_ip = lambda_ip.as_array();
        let mask = mask.as_array();

        let calc_shut_in = calc_sh_mask(lambda_ip, mask);

        calc_shut_in.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "objective_add")]
    fn objective_add_py<'py>(py: Python<'py>, rate_prim: PyReadonlyArray1<'_, f64>, rate_crm: PyReadonlyArray2<'_, f64>, rate_bhp: PyReadonlyArray2<'_, f64>) -> &'py PyArray2<f64> {
        let rate_prim = rate_prim.as_array();
        let rate_crm = rate_crm.as_array();
        let rate_bhp = rate_bhp.as_array();

        let result = objective_add(rate_prim, rate_crm, rate_bhp);

        result.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "eq_loop")]
    fn eq_loop_py<'py>(py: Python<'py>, alpha: PyReadonlyArray1<'_, f64>, beta: PyReadonlyArray1<'_, f64>, lambda_ip: PyReadonlyArray2<'_, f64>, inj: PyReadonlyArray2<'_, f64>) -> &'py PyArray2<f64> {
        let alpha = alpha.as_array();
        let beta = beta.as_array();
        let lambda_ip = lambda_ip.as_array();
        let inj = inj.as_array();

        let f_o = eq_loop(alpha, beta, lambda_ip, inj);

        f_o.into_pyarray(py)
    }

    Ok(())
}