use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

/// Python module implemented in Rust.
#[pymodule]
fn proxy_crm(_py: Python, m: &PyModule) -> PyResult<()> {
    fn q_prim(prod: ArrayView1<'_, f64>, time: ArrayView1<'_, f64>, lambda_prod: f64, tau_prim: f64) -> Array1<f64> {
        let time_decay = (-&time / tau_prim).mapv(f64::exp);
        time_decay * prod[[0]] * lambda_prod
    }

    fn q_crm(inj: ArrayView2<'_, f64>, time: ArrayView1<'_, f64>, lambda_ip: ArrayView1<'_, f64>, tau: ArrayView1<'_, f64>) -> Array1<f64> {
        let n_t = time.raw_dim()[0];
        let n_inj = lambda_ip.raw_dim()[0];
        let mut convolve: Array2<f64> = Array2::zeros([n_t, n_inj]);

        for j in 0..n_inj {
            convolve[[0,j]] = (1.0 - ((time[0] - time[1]) / tau[j]).exp()) * inj[[0,j]];
            for k in 1..n_t {
                for m in 1..k+1 {
                    let time_decay = (1.0 - ((time[m-1] - time[m]) / tau[j]).exp()) * ((time[m] - time[k]) / tau[j]).exp();

                    convolve[[k,j]] += time_decay * inj[[m,j]];
                }
            }
        }
        convolve.dot(&lambda_ip)
    }

    fn q_bhp(press_local: ArrayView1<'_, f64>, press: ArrayView2<'_, f64>, v_matrix: ArrayView1<'_, f64>) -> Array1<f64> {
        let n_t: usize = press.raw_dim()[0];
        let n_prod: usize = press.raw_dim()[1];
        let mut press_diff: Array2<f64> = Array2::zeros([n_t,n_prod]);
        
        for j in 0..n_prod {
            for t in 1..n_t {
                press_diff[[t,j]] = press_local[t-1] - press[[t,j]]
            }
        }
        press_diff.dot(&v_matrix)
    }

    fn sh_mask(prod: ArrayView1<'_, f64>, lambda_ip: ArrayView1<'_, f64>) -> Array1<f64> {
        let n_t: usize = prod.raw_dim()[0];
        let n_inj: usize = lambda_ip.raw_dim()[0];
        
        let tensor: Array1<f64> = Array1::ones([n_inj]);
        let mut mask = Array1::zeros([n_t]);
        let mut lambda_ip_result = lambda_ip.to_owned();

        for t in 0..n_t {
            for i in 0..n_inj {
                //calculate shut-in mask.
                if prod[t] == 0.0 {
                    mask[t] = 1.0;
                }
                let sum_j = (&lambda_ip * &tensor * &mask.slice(s![t])).sum();
                lambda_ip_result[i] = (1.0 - mask[t]) * lambda_ip[i] * (tensor[i] + sum_j);
            }
        }
        lambda_ip_result
    }
    // wrapper
    #[pyfn(m)]
    #[pyo3(name = "q_prim")]
    fn q_prim_py<'py>(py: Python<'py>, prod: PyReadonlyArray1<f64>, time: PyReadonlyArray1<f64>, lambda_prod: f64, tau_prim: f64) -> &'py PyArray1<f64> {
        let prod = prod.as_array();
        let time = time.as_array();
        let q = q_prim(prod, time, lambda_prod, tau_prim);
        
        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "q_crm")]
    fn q_crm_py<'py>(py: Python<'py>, inj: PyReadonlyArray2<f64>, time: PyReadonlyArray1<f64>, lambda_ip: PyReadonlyArray1<'_, f64>, tau: PyReadonlyArray1<'_, f64>) -> &'py PyArray1<f64> {
        let inj = inj.as_array();
        let time = time.as_array();
        let lambda_ip = lambda_ip.as_array();
        let tau = tau.as_array();
        let q = q_crm(inj, time, lambda_ip, tau);

        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "q_bhp")]
    fn q_bhp_py<'py>(py: Python<'py>, press_local: PyReadonlyArray1<'_, f64>, press: PyReadonlyArray2<'_, f64>, v_matrix: PyReadonlyArray1<'_, f64>) -> &'py PyArray1<f64> {
        let press_local = press_local.as_array();
        let press = press.as_array();
        let v_matrix = v_matrix.as_array();
        let q = q_bhp(press_local, press, v_matrix);

        q.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "sh_mask")]
    fn sh_mask_py<'py>(py: Python<'py>, prod: PyReadonlyArray1<f64>, lambda_ip: PyReadonlyArray1<'_, f64>) -> &'py PyArray1<f64> {
        let prod = prod.as_array();
        let lambda_ip = lambda_ip.as_array();

        let shut_in = sh_mask(prod, lambda_ip);

        shut_in.into_pyarray(py)
    }

    Ok(())
}