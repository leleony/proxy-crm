# Proxy CRM
Inspired by Frank Male's [pywaterflood](https://github.com/frank1010111/pywaterflood) and Deepthi Sen's [Capacitance Resistance Model](https://github.com/deepthisen/CapacitanceResistanceModel) repository, this repository is made to analyze the interwell connectivity of shut-in well during waterflooding [1]. The base model itself 

## Shut-In Well Algorithm
One new thing that have been added into this repository is the algorithm for shut-in well, which is written as this indicator function (this is in equation form):
![equation](https://latex.codecogs.com/svg.image?\left.S&space;H_{\text{mask}}\right|_t=\left\{\begin{array}{l}0,\forall&space;j:q_j(t)\neq&space;0\\1,\forall&space;j:q_j(t)=0\end{array}\right.)
During the shut-in well, the indicator function will return 1.0 and the shut-in algorithm will begin. This will make the interwell connectivity (λ) of the shut-in producer-x into the value of zero. However, we have to also store the interwell connectivity before it got zeroed to be added to other non-shut-in producer-j at that same timestep.

## Fractional Flow Model
Here, we referenced the (Liang et al., 2007) [2] method to obtain the oil production rate from the liquid production rate. The equation to obtain the fraction of oil is:
![equation](https://latex.codecogs.com/svg.image?\begin{aligned}&f_{o&space;j}(t)=\left[1&plus;\alpha_j\left(\sum_{m=1}^n\left\{\sum_{i=1}^{N&space;I}\lambda_{i&space;j}i_i\left(t_m\right)\right\}\right)^{\beta_j}\right]^{-1}\end{aligned})
Finally, the oil fraction is multiplied with liquid production rate! However, this is still a WIP as there are some problems on the log functions.

## "I Would Love To Contribute!"
Thank you! You can open an issue or just send a pull request.

## License
This repository is licensed to [GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

## Citations
[1] Gubanova, A., Orlov, D., Koroteev, D., & Shmidt, S. (2022). Proxy Capacitance-Resistance Modeling for Well Production Forecasts in Case of Well Treatments. SPE Journal, 27(06), 3474–3488. [https://doi.org/10.2118/209829-PA](https://doi.org/10.2118/209829-PA)
[2] Liang, X., Weber, D. B., Edgar, T. F., Lake, L. W., Sayarpour, M., and A. Al-Yousef. (2007). Optimization of Oil Production Based on A Capacitance Model of Production and Injection Rates. [https://doi.org/10.2118/107713-MS](https://doi.org/10.2118/107713-MS)
[3] Sayarpour, M., Zuluaga, E., Kabir, C. S., & Lake, L. W. (2009). The use of capacitance–resistance models for rapid estimation of waterflood performance and optimization. Journal of Petroleum Science and Engineering, 69(3–4), 227–238. [https://doi.org/10.1016/j.petrol.2009.09.006](https://doi.org/10.1016/j.petrol.2009.09.006)
