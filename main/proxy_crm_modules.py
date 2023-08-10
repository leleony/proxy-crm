from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy import optimize
import proxy_crm

def q_prim(prod: NDArray, time: NDArray, lambda_prod: float, tau_prim: float, mask: NDArray) -> NDArray:
  """Calculate primary prod contribution.

  Uses Arps equation with :math:`b=0`

  .. math::
    q_{p}(t) = q_i e^{-bt}

  Args
  ----------
  prod : NDArray
    prod, size: Number of time steps
  time : NDArray
    Producing times to forecast, size: Number of time steps
  lambda_prod : float
    Arps q_i factor
  tau_prim : float
    Arps time constant
  mask : NDArray
    Shut-in detector

  Returns
  ----------
  q_hat : NDArray
    Calculated prod
    size: Number of time steps
  """
  return proxy_crm.q_prim(prod, time, lambda_prod, tau_prim, mask)


def q_crm(inj: NDArray, time: NDArray, lambda_ip: NDArray, sum_lambda_ip: NDArray, tau: float, mask: NDArray) -> NDArray:
    """Calculate per injector-producer pair prod (simplified tank).

    Uses simplified CRMp model that assumes a single tau for each producer. This function is automatically run with 'calc_sh_mask'.

    Args
    ----------
    inj : NDArray
        injected fluid in reservoir volumes
        size: Number of time steps
    time : NDArray
        Producing times to forecast
        size: Number of time steps
    lambda_ip : NDArray
        Connectivities between each injector and the producer
        size: Number of injectors
    sum_lambda_ip: NDArray
        Required to calculate the shut-in mask
        size: Number of injectors
    tau : float
        Time constants all injectors and the producer
    mask: NDArray
        A sensor that will detect when the rate of production equals to zero
        size: Number of time steps

    Returns
    ----------
    q_hat : NDArray
        Calculated prod
        size: Number of time steps x Number of injector wells.
    """
    return proxy_crm.q_crm(inj, time, lambda_ip, sum_lambda_ip, tau, mask)


def q_bhp(time: NDArray, tau: float, press: NDArray, prod_index: NDArray) -> NDArray:
    r"""Calculate the prod effect from bottom-hole pressure variation.

    Args
    ----
    time : NDArray
        Producing times to forecast
        size: Number of time steps
    tau : float
        Time constants all injectors and the producer
        size: Number of inejctors
    pressure : NDArray
        bottomhole pressure
        shape: n_time, n_producers
    prod_index : NDArray
        Productivity Index of the case, size: n_producers

    Returns
    -------
    q : NDArray
        prod from changing BHP
        size: n_time
    """
    return proxy_crm.q_bhp(time, tau, press, prod_index)

def sh_mask(prod: NDArray):
  return proxy_crm.sh_mask(prod)

def calc_sh_mask(lambda_ip: NDArray, sum_lambda_ip: NDArray, sh_mask:NDArray):
  return proxy_crm.calc_sh_mask(lambda_ip, sum_lambda_ip, sh_mask)

def rand_weights(n_prod: int, n_inj: int, axis: int = 0, seed: int | None = None) -> NDArray:
    """Generate random weights for producer-injector lambda_ip.

    Args
    ----
    n_i : int
    n_j : int
    axis : int, default is 0
    seed : int, default is None

    Returns
    -------
    gains_guess: NDArray
    """
    rng = np.random.default_rng(seed)
    limit = 10 * (n_prod if axis == 0 else n_inj)
    vec = rng.integers(0, limit, (n_prod, n_inj))
    axis_sum = vec.sum(axis, keepdims=True)
    return vec / axis_sum


class proxyCRM:
    """Capacitance Resistance Model using proxy method and comparison with PINN-Bayesian method.

    CRM uses a physics-inspired mass balance approach to explain production for waterfloods. It treats each injector-producer well pair as a system with mass input, output, and pressure related to the mass balance.
    In this case, the method of CRM will be CRM-Producer (CRMP), with the appliance of shut-in mask in case of well shut-in.

    Args
    ----------
    pressure : NDArray
      The value of bottomhole pressure from the project. This is optional to the availability of the data.

    References
    ----------
    "Proxy Capacitance-Resistance Modeling for Well Production Forecasts in Case of Well Treatments" - Gubanova et al., 2022.

    * Do note that this code is heavily adapted from 'pywaterflood' by Frank Male (kindly visit his github page).
    """

    def __init__(self, pressure: bool = False):
      if type(pressure)!= bool:
        msg = '(っ °Д °;)っ To initialize pressure, insert True-False (boolean) type. This is False in default.'
        raise TypeError(msg)

    def fit(self, prod: NDArray, inj: NDArray, press: NDArray, time: NDArray, init_guess: NDArray = None, num_cores: int = 1, random: bool = False):
      """Build a CRM model from the prod and inj data.

      Args
      ----------
      prod : NDArray
        prod rates for each time period,
        shape: (n_time, n_producers)
      inj : NDArray
        inj rates for each time period,
        shape: (n_time, n_injectors)
      pressure : NDArray
        average pressure for each producer for each time period,
        shape: (n_time, n_producers)
      time : NDArray
        relative time for each rate measurement, starting from 0,
        shape: (n_time)
      init_guess : NDArray
        initial guesses for lambda_ip, tau, primary prod contribution
        shape: (len(guess), n_producers)
      num_cores (int): number of cores to run fitting procedure on, defaults to 1
      random : bool
        whether to randomly initialize the lambda_ip
      **kwargs:
        keyword arguments to pass to scipy.optimize fitting routine

      Returns
      ----------
      self: trained model
      """
      self.prod = prod
      self.inj = inj
      self.time = time
      self.press = press

      sum_lambda_ip = np.zeros(inj.shape[1])

      if not init_guess:
        init_guess = self._get_init_guess(random=random)
      bounds, constraints = self._get_bounds()
      

      def fit_well(prod, press, x0):
        # residual is an L2 norm
        def residual(x, prod):
          return sum((prod - self._calc_qhat(x, prod, inj, time, press, sum_lambda_ip)) ** 2)

        return optimize.minimize(residual, x0, method='L-BFGS-B', bounds=bounds, constraints=constraints, args=(prod), options={'disp':True})

      if num_cores == 1:
        results = map(fit_well, self.prod.T, press.T, init_guess)
      else:
        results = Parallel(n_jobs=num_cores)(delayed(fit_well)(prod, press, x0) for prod, press, x0 in zip(self.prod.T, press.T, init_guess))

      opts_perwell = [self._split_opts(r["x"]) for r in results]
      lambda_ip_perwell, tau_perwell, lambda_prod, tau_prim, prod_index = map(list, zip(*opts_perwell))

      self.lambda_ip: NDArray = np.vstack(lambda_ip_perwell)
      self.tau: NDArray = np.vstack(tau_perwell)
      self.lambda_prod = np.array(lambda_prod)
      self.tau_prim = np.array(tau_prim)
      self.prod_index: NDArray = np.array(prod_index)
      return self

    def predict(self, inj=None, time=None, connections=None, prod=None, press=None):
      """Predict prod for a trained model.

      If the inj and time are not provided, this will use the training values

      Args
      ----------
      inj : Optional NDArray
        The inj rates to input to the system, shape (n_time, n_inj)
      time : Optional NDArray
        The timesteps to predict
      connections : Optional dict
        if present, the lambda_ip, tau, lambda_prod, tau_prim matrices
      prod : Optional NDArray
        The prod (only takes first row to use for primary prod decline)

      Returns
      ----------
      q_hat :NDArray
        The predicted values, shape (n_time, n_producers)
      """
      if connections is not None:
        lambda_ip = connections.get("lambda_ip", self.lambda_ip)
        tau = connections.get("tau", self.tau)
        lambda_prod = connections.get("lambda_prod", self.lambda_prod)
        tau_prim = connections.get("tau_prim", self.tau_prim)
        prod_index = connections.get("prod_index", self.prod_index)
      else:
        lambda_ip = self.lambda_ip
        tau = self.tau
        lambda_prod = self.lambda_prod
        tau_prim = self.tau_prim
        prod_index = self.prod_index

      if prod is None:
        prod = self.prod

      if press is None:
        press = self.press

      n_prod = prod.shape[1]

      if int(inj is None) + int(time is None) == 1:
        msg = "predict() takes 1 or 3 arguments, 2 given"
        raise TypeError(msg)
      if inj is None:
        inj = self.inj

      n_inj = inj.shape[1]

      if time is None:
        time = self.time
      if time.shape[0] != inj.shape[0]:
        msg = "injection and time need same number of steps"
        raise ValueError(msg)
      
      sum_lambda_ip = np.zeros(inj.shape[1])
      q_hat = np.zeros((len(time), n_prod))

      for i in range(n_prod):
        sum_lambda_ip += lambda_ip[i,:]
        mask = sh_mask(prod[:,i])
        q_hat[:,i] += q_prim(prod[:, i], time, lambda_prod[i], tau_prim[i], mask)
        for j in range(n_inj):
          tau_crm = np.full(n_inj, tau[i])
          q_hat[:,i] += q_crm(inj, time, lambda_ip[i, :], sum_lambda_ip, tau_crm, mask)[:,j]
        q_hat[:,i] += q_bhp(time, tau[i], press[:, i], prod_index[i])
      return q_hat

    def set_rates(self, prod=None, inj=None, time=None):
      """Set prod and inj rates and time array.

      Args
      -----
      prod : NDArray
        prod rates with shape (n_time, n_producers)
      inj : NDArray
        inj rates with shape (n_time, n_injectors)
      time : NDArray
        timesteps with shape n_time
      """
      if prod is not None:
        self.prod = prod
      if inj is not None:
        self.inj = inj
      if time is not None:
        self.time = time

    def set_connections(self, lambda_ip=None, tau=None, lambda_prod=None, tau_prim=None):
      """Set waterflood properties.

      Args
      -----
      lambda_ip : NDArray
        connectivity between injector and producer
        shape: n_gains, n_producers
      tau : NDArray
        time-constant for inj to be felt by prod
        shape: either n_producers or (n_gains, n_producers)
      lambda_prod : NDArray
        gain on primary prod, shape: n_producers
      tau_prim : NDArray
        Arps time constant for primary prod, shape: n_producers
      """
      if lambda_ip is not None:
        self.lambda_ip = lambda_ip
      if tau is not None:
        self.tau = tau
      if lambda_prod is not None:
        self.lambda_prod = lambda_prod
      if tau_prim is not None:
        self.tau_prim = tau_prim

    def residual(self, prod=None, inj=None, time=None):
      """Calculate the prod minus the predicted prod for a trained model.

      If the prod, inj, and time are not provided, this will use the training values

      Args
      ----------
      prod : NDArray
        The prod rates observed, shape: (n_timesteps, n_producers)
      inj : NDArray
        The inj rates to input to the system,
        shape: (n_timesteps, n_injectors)
      time : NDArray
        The timesteps to predict

      Returns
      ----------
      residual : 
        The true prod data minus the predictions, shape (n_time, n_producers)
        """
      q_hat = self.predict(inj, time)
      if prod is None:
        prod = self.prod
      return prod - q_hat

    def to_excel(self, fname: str):
        """Write trained model to an Excel file.

        Args
        ----
        fname : str
            Excel file to write out

        """
        for x in ("lambda_ip", "tau", "lambda_prod", "tau_prim"):
            if x not in self.__dict__.keys():
                msg = "Model has not been trained"
                raise ValueError(msg)
        with pd.ExcelWriter(fname) as f:
            pd.DataFrame(self.lambda_ip).to_excel(f, sheet_name="lambda_ip")
            pd.DataFrame(self.tau).to_excel(f, sheet_name="tau")
            pd.DataFrame(
                {
                    "Producer lambda_ip": self.lambda_prod,
                    "Producer tau": self.tau_prim,
                }
            ).to_excel(f, sheet_name="Primary prod")

    def _get_init_guess(self, random=False) -> [NDArray, NDArray, NDArray, NDArray, NDArray]:
      """Create initial guesses for the CRM model parameters.

      :meta private:

      Args
      ----------
      tau_selection : str, one of 'per-pair' or 'per-producer'
        sets whether to use CRM (per-pair) or CRMp model
      random : bool
        whether initial lambda_ip are randomly (true) or proportionally assigned
        Returns
      ----------
      x0 : NDArray
        Initial primary prod gain, time constant and waterflood lambda_ip and time constants, as one long 1-d array
      """

      n_inj = self.inj.shape[1]
      n_prod = self.prod.shape[1]
      d_t = self.time[1] - self.time[0]

      axis = 0
      if random:
        rng = np.random.default_rng()
        lambda_prod_guess1 = rng.random(n_prod)
        lambda_ip_guess1 = rand_weights(n_prod, n_inj, axis)
      else:
        lambda_ip_unnormed = np.ones((n_prod, n_inj))
        lambda_ip_guess1 = lambda_ip_unnormed / np.sum(lambda_ip_unnormed, axis, keepdims=True)
        lambda_prod_guess1 = np.ones(n_prod)

      tau_prim_guess1 = d_t * np.ones(n_prod)
      tau_guess1 = np.ones((n_prod, 1))
      
      prod_index = np.ones((n_prod,1))/10

      guess = [np.concatenate([lambda_ip_guess1[i, :], tau_guess1[i, :], lambda_prod_guess1[[i]], tau_prim_guess1[[i]]]) for i in range(n_prod)]
      x0 = [np.concatenate([guess[i], prod_index[i,:]])for i in range(len(guess))]
      
      return x0

    def _opt_nums(self) -> tuple[int, int, int, int]:
      """Return the number of lambda_ip, tau, primary production, and production index parameters to fit."""
      n_lambda_ip = self.inj.shape[1]
      n_tau = 1
      n_prim = 2
      
      return n_lambda_ip, n_tau, n_prim, self.prod.shape[1]

    def _get_bounds(self) -> tuple[tuple, tuple | dict]:
      """Create bounds for the model from initialized constraints."""
      bounds = []
      n_inj, _, _, _, = self._opt_nums()

      #lambda_ip bounds
      bounds.extend([(0.0,1.0)] * n_inj)

      #tau_bounds
      bounds.append((0.0001,10))

      #lambda-prod bounds
      bounds.append((0.0,1.0))

      #tau-prim bounds
      bounds.append((0.0001,10))

      #prod-idx bounds
      bounds.append((0.0,10))

      bounds = tuple(bounds)

      constraints_optimizer = ()

      return bounds, constraints_optimizer

    def _calc_qhat(self, x: NDArray, prod: NDArray, inj: NDArray, time: NDArray, press: NDArray, sum_lambda_ip: NDArray):
        lambda_ip, tau, lambda_prod, tau_prim, prod_index = self._split_opts(x)
        _, _, _, n_prod = self._opt_nums()
        mask = sh_mask(prod)
        sum_lambda_ip += lambda_ip 

        tau_crm = np.full(inj.shape[1], tau)
        tau_bhp = np.full(n_prod, tau)
        prod_index = np.full(n_prod, prod_index)

        q_hat = q_prim(prod, time, lambda_prod, tau_prim, mask)
        for i in range(inj.shape[1]):
          q_hat += q_crm(inj, time, lambda_ip, sum_lambda_ip, tau_crm, mask)[:,i]

        q_hat += q_bhp(time, tau_bhp, press, prod_index)

        return q_hat

    def _split_opts(self, x: NDArray):
      n_lambda_ip, n_tau, n_prim = self._opt_nums()[:3]
      n_connectivity = n_lambda_ip + n_tau

      lambda_ip = x[:n_lambda_ip]
      tau = x[n_lambda_ip:n_connectivity]

      lambda_prod = x[n_connectivity:][0]
      tau_prim = x[n_connectivity:][1]

      prod_index = x[n_connectivity + n_prim:]

      if tau < 1e-10:
        tau = 1e-10

      if tau_prim < 1e-10:
        tau_prim = 1e-10

      return lambda_ip, tau, lambda_prod, tau_prim, prod_index
    

"""
--------------------------------------------------------------
Codes are cited from:
@misc{Feinman2021,
  author = {Feinman, Reuben},
  title = {Pytorch-minimize: a library for numerical optimization with autograd},
  publisher = {GitHub},
  year = {2021},
  url = {https://github.com/rfeinman/pytorch-minimize},
}

@misc{Male2023,
  author = {Male, Frank},
  title = {pywaterflood: Waterflood Connectivity Analysis},
  publisher = {GitHub},
  year = {2023},
  url = {https://github.com/frank1010111/pywaterflood},
}
"""