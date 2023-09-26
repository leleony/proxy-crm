from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from numba import jit
import matplotlib.pyplot as plt
import scienceplots

import proxy_crm

def eq_loop(alpha, beta, lambda_ip, inj):
    return proxy_crm.eq_loop(alpha, beta, lambda_ip, inj)

class frac_flow_wor:
    def __init__(self, wor: NDArray, liquid: NDArray, lambda_ip: NDArray, inj: NDArray):
        """
        This is used to calculate the oil-cut fraction of the total liquid production, by basing on oil-cut model by Liang et al.(2007) and Sayarpour et al. (2009). The chosen WOR or CWI will be plotted in a log-log plot and its correlation of alpha and beta will be used for the fractional flow model calculation, thus the oil production rate can automatically be obtained.

        Args
        ----------
        liquid: NDArray
            Liquid production rate for each time period,
            shape: (n_time, n_producers)
        wor: NDArray
            Water-oil ratio of the whole field/all producer wells/only one producer well,
            shape: (n_time, n_producers)
        lambda_ip : NDArray
            Interwell connectivity weight obtained from proxy_crm.py output.
            shape: (n_producers, n_injectors)
        inj : NDArray
            injection rates for each time period,
            shape: (n_time, n_injectors)

        """
        n_prod = liquid.shape[1]
        n_inj = inj.shape[1]
        if wor is None:
            msg = 'Please insert water-oil ratio. It should be served in an NDArray'
            raise ValueError(msg)
        if liquid is None:
            self.liquid = proxy_crm.q_hat
        if lambda_ip is None:
            self.lambda_ip = proxy_crm.lambda_ip.reshape((n_prod,n_inj))
        if inj is None:
            self.inj = proxy_crm.inj

        self.wor = wor
        self.liquid = liquid
        self.lambda_ip = lambda_ip.reshape((n_prod,n_inj))
        self.inj = inj

        self.n_prod = lambda_ip.shape[0]
        self.n_inj = inj.shape[1]
        self.n_t = inj.shape[0]
    
    def fit_fractional(self):
        """
        Fitting process before obtaining the alpha and beta required and the final oil production.
        Returns
        ----------
        q_oil: The predicted oil production rate, multiplication product of f_oil * given liquid production rate.
        """
        wor = self.wor
        lambda_ip = self.lambda_ip
        inj = self.inj

        # Calculating CWI using lambda_ip * inj
        @jit(nopython=True)
        def cumulative_water(lambda_ip, inj):
            cwi = np.zeros(inj.shape[0])
            for t in range(inj.shape[0]):
                for j in range(lambda_ip.shape[0]):
                    for i in range(lambda_ip.shape[1]):
                        cwi += lambda_ip[j,i] * inj[t,i]

            return cwi
        
        # Defining log(CWI) and log(WOR)
        cwi = cumulative_water(lambda_ip, inj)
        mask = (cwi != 0) & (wor != 0)
        log_x = np.log10(cwi[mask])
        log_y = np.log10(wor[mask])

        # Power function for the log-log plot linear regression
        def powFunc(x,a,b):
            return a * np.power(x,b)
        
        popt, _ = curve_fit(powFunc, cwi, wor) # curve fitting method

        # Defining alpha and beta
        beta = popt[0]
        alpha = popt[1]

        # Plot settings
        self.log_log_plot(cwi=cwi[mask],wor=wor[mask])
        plt.plot(cwi[mask], powFunc(cwi[mask], *popt), 'r-', label="({0:.2e}*x**{1:.3f})".format(*popt)) # Adding the power function curve fit
        plt.legend(loc='lower right', frameon=True)
        plt.show()
        
        f_oil = eq_loop(alpha=alpha, beta=beta, lambda_ip=lambda_ip, inj=inj)
        
        return f_oil * self.liquid
    
    def log_log_plot(self, cwi, wor):
        """
        This function is meant to directly plot the log-log plot between CWI and WOR.
        Returns
        ----------
        A linear correlation of WOR vs CWI log-log plot.
        """
        fig = plt.figure()
        ax = plt.gca()

        ax.scatter(cwi, wor)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(visible=True)

        plt.style.use(['science', 'no-latex'])
        plt.title('Log-Log Plot of Fractional Flow Model')
        plt.xlabel('log(CWI)')
        plt.ylabel('log(WOR)')

class frac_flow_fit:
    def __init__(self, liquid: NDArray, lambda_ip: NDArray, inj: NDArray, oil: NDArray):
        """
        This is also used to calculate the oil-cut fraction of the total liquid production, by basing on oil-cut model by Liang et al.(2007) and Sayarpour et al. (2009). The difference is this will be using optimizers to find the optimum alpha and beta values as the parameters.

        Args
        ----------
        liquid: NDArray
            Liquid production rate for each time period,
            shape: (n_time, n_producers)
        lambda_ip: NDArray
            Interwell connectivity weight obtained from proxy_crm.py output.
            shape: (n_producers, n_injectors)
        inj: NDArray
            injection rates for each time period,
            shape: (n_time, n_injectors)
        oil: NDArray
            predicted oil rates from each time period from proxy_crm.py output.
            shape: (n_time, n_producers)
        """
        # Registering observed values
        self.liquid = liquid
        self.inj = inj
        self.oil = oil

        self.n_time = self.liquid.shape[0]
        self.n_prod = self.liquid.shape[1]
        self.n_inj = self.inj.shape[1]

        if len(lambda_ip.shape) < 2: # Check whether lambda_ip is already reshaped
        self.lambda_ip = lambda_ip.reshape((self.n_prod, self.n_ij))
        elif len(lambda_ip.shape) == 2:
        self.lambda_ip = lambda_ip

        # Create zero values for the target of the code (the predicted data and params)
        self.f_oil = self.q_oil = np.zeros((self.n_time, self.n_prod))
        self.conv = np.zeros((self.n_time, self.n_prod))
        self.alpha = self.beta = np.zeros((self.n_prod, self.n_inj))

    def solver(self, method='leastsq'):
        """
        Optimize the alpha and beta parameters using leastsq method (by default).
        
        Args
        ----------
        method: str
            Choosing optimization/solving method. Default to 'leastsq', but can be changed according to selections available in lmfit.
        
        Return
        ----------
        final: NDArray
            Predicted oil rate from the optimization,
            Shape: (n_time, n_producers)
        alpha: NDArray
            Alpha parameter
            Shape: (n_producers)
        beta: NDArray
            Beta parameter
            Shape: (n_producers)
        """
        liquid = self.liquid
        lambda_ip = self.lambda_ip
        inj = self. inj
        oil = self.oil

        # Defining objective function
        def fcn_min(params, oil, liquid, lambda_ip, inj):
        for j in range(self.n_prod):
            self.alpha[j] = params[f'alpha_{j}']
            self.beta[j] = params[f'beta_{j}']
        
        oil_model = self.f_oil(lambda_ip=lambda_ip, inj=inj, alpha=self.alpha, beta= self.beta)
        return oil - (oil_model * liquid)

        # Creating parameters
        params = Parameters()
        for j in range(self.n_prod):
        for i in range(self.n_inj):
            params.add(f'alpha_{j}', value=1e-10, min=0)
            params.add(f'beta_{j}', value=0.01, min=0)

        # Solving with lmfit minimizer, methods can be customized.
        solve = Minimizer(fcn_min, params, fcn_args=(oil, liquid, lambda_ip, inj))
        result = solve.minimize(method=method)

        final = oil + result.residual.reshape(self.n_time, self.n_prod)

        return final, self.alpha, self.beta, report_fit(result.params)

    def f_oil(self, f_ij, inj, alpha, beta):
        # Calculates oil-cut for CRMP method.
        for t in range(self.n_time):
            for j in range(self.n_prod):
                for i in range(self.n_inj):
                self.conv[t,j] = alpha[j] * (f_ij[j,i] * inj[t,i]) ** beta[j]
                self.f_oil[t,j] = 1.0 / (1.0 + self.conv[t,j])

        return self.f_oil