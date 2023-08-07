use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

/// Python module implemented in Rust.
#[pymodule]
fn proxy_crm(_py: Python, m: &PyModule) -> PyResult<()> {
    fn q_prim(prod: ArrayView1<'_, f64>, time: ArrayView1<'_, f64>, lambda_prod: f64, tau_prim: f64, mask: ArrayView1<'_, f64>) -> Array1<f64> {
        let time_decay = (-&time / tau_prim).mapv(f64::exp);
        let prod_masked = prod.to_owned() * mask.mapv(|x| 1.0 - x);

        time_decay * prod_masked[[0]] * lambda_prod
    }

    fn q_crm(inj: ArrayView2<'_, f64>, time: ArrayView1<'_, f64>, lambda_ip: ArrayView1<'_, f64>, sum_lambda_ip: ArrayView1<'_, f64>,  tau: ArrayView1<'_, f64>, mask: ArrayView1<'_, f64> ) -> Array2<f64> {
        let n_t = time.raw_dim()[0];
        let n_inj = lambda_ip.raw_dim()[0];
        let mut convolve: Array2<f64> = Array2::zeros([n_t, n_inj]);
        
        let lambda_ip_mod = calc_sh_mask(lambda_ip, sum_lambda_ip, mask);

        for j in 0..n_inj {
            convolve[[0,j]] = (1.0 - ((time[0] - time[1]) / tau[j]).exp()) * lambda_ip_mod[[0,j]] * inj[[0,j]];
            for k in 1..n_t {
                for m in 1..k+1 {
                    let time_decay = (1.0 - ((time[m-1] - time[m]) / tau[j]).exp()) * ((time[m] - time[k]) / tau[j]).exp();

                    convolve[[k,j]] = convolve[[k,j]] + (time_decay * lambda_ip_mod[[k,j]] * inj[[m,j]]);
                }
            }
        }
        convolve
    }

    fn q_bhp(time: ArrayView1<'_, f64>, tau: ArrayView1<'_, f64>, press: ArrayView1<'_, f64>, prod_index: ArrayView1<'_, f64>) -> Array1<f64> {
        let n_t = time.raw_dim()[0];
        let n_prod = tau.raw_dim()[0];
        let mut convolve: Array1<f64> = Array1::zeros([n_t]);
        
        for j in 0..n_prod {
            convolve[0] = (1.0 - ((time[0] - time[1]) / tau[j]).exp()) * prod_index[0] * tau[j] * (press[0] - press[1]) / ((time[0]-time[1]));
            for k in 1..n_t {
                for m in 1..k+1 {
                    let time_decay = (1.0 - ((time[m-1] - time[m]) / tau[j]).exp()) * ((time[m] - time[k]) / tau[j]).exp();

                    convolve[k] = convolve[k] + (time_decay * prod_index[j] * tau[j] * (press[m-1] - press[m]) / ((time[m-1]-time[m])));
                }
            }
        }
        convolve
    }

    fn sh_mask(prod: ArrayView1<'_, f64>) -> Array1<f64> {
        let n_t: usize = prod.raw_dim()[0];
        let mut mask = Array1::zeros([n_t]);

        for (t, &val) in prod.iter().enumerate() {
            if val == 0.0 {
                mask[t] = 1.0;
            } else {
                mask[t] = 0.0
            }
        }
        mask
    }

    fn calc_sh_mask(lambda_ip:ArrayView1<'_, f64>, sum_lambda_ip: ArrayView1<'_, f64>, mask: ArrayView1<'_, f64>) -> Array2<f64> {
        let n_t: usize = mask.raw_dim()[0];
        let n_inj: usize = lambda_ip.raw_dim()[0];
        
        let tensor: Array2<f64> = Array2::ones([n_t, n_inj]);
        let mut lambda_ip_result: Array2<f64> = Array2::zeros([n_t,n_inj]);

        for t in 0..n_t{
            for i in 0..n_inj{
                lambda_ip_result[[t,i]] = f64::abs((mask[t] - 1.0) * lambda_ip[i] * (tensor[[t,i]] + sum_lambda_ip[i] * tensor[[t,i]] * mask[t]));
            }
        }
        lambda_ip_result
    }

    // wrapper
    #[pyfn(m)]
    #[pyo3(name = "q_prim")]
    fn q_prim_py<'py>(py: Python<'py>, prod: PyReadonlyArray1<f64>, time: PyReadonlyArray1<f64>, lambda_prod: f64, tau_prim: f64, mask: PyReadonlyArray1<f64>) -> &'py PyArray1<f64> {
        let prod = prod.as_array();
        let time = time.as_array();
        let mask = mask.as_array();

        let q = q_prim(prod, time, lambda_prod, tau_prim,mask);
        
        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "q_crm")]
    fn q_crm_py<'py>(py: Python<'py>, inj: PyReadonlyArray2<f64>, time: PyReadonlyArray1<f64>, lambda_ip: PyReadonlyArray1<'_, f64>, sum_lambda_ip: PyReadonlyArray1<'_, f64>, tau: PyReadonlyArray1<'_, f64>, mask: PyReadonlyArray1<'_, f64>) -> &'py PyArray2<f64> {
        let inj = inj.as_array();
        let time = time.as_array();
        let lambda_ip = lambda_ip.as_array();
        let sum_lambda_ip = sum_lambda_ip.as_array();
        let tau = tau.as_array();
        let mask = mask.as_array();

        let q = q_crm(inj, time, lambda_ip, sum_lambda_ip, tau, mask);

        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "q_bhp")]
    fn q_bhp_py<'py>(py: Python<'py>, time: PyReadonlyArray1<f64>, tau: PyReadonlyArray1<'_, f64>, press: PyReadonlyArray1<'_, f64>, prod_index: PyReadonlyArray1<'_, f64>) -> &'py PyArray1<f64> {
        let time = time.as_array();
        let tau = tau.as_array();
        let press = press.as_array();
        let prod_index = prod_index.as_array();

        let q = q_bhp(time, tau, press, prod_index);

        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "sh_mask")]
    fn sh_mask_py<'py>(py: Python<'py>, prod: PyReadonlyArray1<f64>) -> &'py PyArray1<f64> {
        let prod = prod.as_array();

        let shut_in = sh_mask(prod);

        shut_in.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "calc_sh_mask")]
    fn calc_sh_mask_py<'py>(py: Python<'py>, lambda_ip: PyReadonlyArray1<'_, f64>, sum_lambda_ip: PyReadonlyArray1<'_, f64>, mask: PyReadonlyArray1<'_, f64>) -> &'py PyArray2<f64> {
        let lambda_ip = lambda_ip.as_array();
        let sum_lambda_ip = sum_lambda_ip.as_array();
        let mask = mask.as_array();

        let calc_shut_in = calc_sh_mask(lambda_ip, sum_lambda_ip, mask);

        calc_shut_in.into_pyarray(py)
    }

    Ok(())
}