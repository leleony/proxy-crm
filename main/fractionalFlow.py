from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numba import jit
import matplotlib.pyplot as plt
import scienceplots

import proxy_crm

def eq_loop(alpha, beta, lambda_ip, inj):
    return proxy_crm.eq_loop(alpha, beta, lambda_ip, inj)

class frac_flow:
    def __init__(self, wor: NDArray, liquid: NDArray, lambda_ip: NDArray, inj: NDArray):
        """
        This is used to calculate the oil-cut fraction of the total liquid production, by basing on oil-cut model by Liang et al.(2007) and Sayarpour et al. (2009). The chosen WOR or CWI will be plotted in a log-log plot and its correlation of alpha and beta will be used for the fractional flow model calculation, thus the oil production rate can automatically be obtained.

        Args
        ----------
        wor: NDArray
            Water-oil ratio of the whole field/all producer wells/only one producer well,
            shape: (n_time, n_producers)
        inj : NDArray
            injection rates for each time period,
            shape: (n_time, n_injectors)
        pressure : NDArray
            average pressure for each producer for each time period,
            shape: (n_time, n_producers)

        """
        if wor is None:
            msg = 'Please insert water-oil ratio. It should be served in an NDArray'
            raise ValueError(msg)
        if liquid is None:
            self.liquid = proxy_crm.q_hat
        if lambda_ip is None:
            self.lambda_ip = proxy_crm.lambda_ip.reshape((4,5))
        if inj is None:
            self.inj = proxy_crm.inj

        self.wor = wor
        self.liquid = liquid
        self.lambda_ip = lambda_ip.reshape((4,5))
        self.inj = inj

        self.n_prod = lambda_ip.shape[0]
        self.n_inj = inj.shape[1]
        self.n_t = inj.shape[0]
    
    def fit_fractional(self):
        """
        Fitting process before obtaining the alpha and beta required and the final oil production.
        Returns
        ----------
        q_oil: multiplication product of f_oil * given liquid production rate.
        """
        wor = self.wor
        lambda_ip = self.lambda_ip
        inj = self.inj

        @jit(nopython=True)
        def cumulative_water(lambda_ip, inj):
            cwi = np.zeros(inj.shape[0])
            for t in range(inj.shape[0]):
                for j in range(lambda_ip.shape[0]):
                    for i in range(lambda_ip.shape[1]):
                        cwi += lambda_ip[j,i] * inj[t,i]

            return cwi
        
        log_x = np.log10(cumulative_water(lambda_ip, inj))
        log_y = np.log10(wor)

        curve = np.polyfit(log_x, log_y, 1)

        beta = curve[0]
        alpha = curve[1]
        y = beta * log_x - np.log10(alpha)

        self.log_log_plot(cwi=cumulative_water(lambda_ip, inj), wor=wor, log_x=log_x, y=y)
        
        f_oil = eq_loop(alpha=alpha, beta=beta, lambda_ip=lambda_ip, inj=inj)
        
        return f_oil * self.liquid
    
    def log_log_plot(self, cwi, wor, log_x, y):
        fig = plt.figure()
        ax = plt.gca()

        ax.scatter(cwi, wor)
        ax.set_yscale('log')
        ax.set_xscale('log')

        plt.style.use(['science', 'no-latex'])
        plt.title('Log-Log Plot of Fractional Flow Model')
        plt.xlabel('log(CWI)')
        plt.ylabel('log(WOR)')
        plt.plot(log_x, y)