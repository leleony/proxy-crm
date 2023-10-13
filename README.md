# Proxy Capacitance Resistance Model (Proxy-CRM)
This repository is highly inspired from [pywaterflood](https://github.com/frank1010111/pywaterflood) and [CapacitanceResistanceModel](https://github.com/deepthisen/CapacitanceResistanceModel). The main source of this code [1] has written about creating Capacitance Resistance Model (CRM), specifically CRMP (Capacitance Resistance Model-Produer)[3], for shut-in well and well treatments that also consider geological uncertainties. However, this repository will only explore the shut-in nature. CRM itself will predict (history match) the observed liquid/oil production rate, providing rapid calculation for history matching in early stage of reservoir performance analysis.

>TLDR: This repository will predict oil/liquid production rate of shut-in period using CRMP.

## Shut-in Well Algorithm
The main star of the repository is this indicator function of shut-in mask.

![equation](https://latex.codecogs.com/svg.image?%5Cleft.S%20H_%7B%5Ctext%7Bmask%7D%7D%5Cright%7C_t=%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bl%7D0,%5Cforall%20j:q_j(t)%5Cneq%200%5C%5C1,%5Cforall%20j:q_j(t)=0%5Cend%7Barray%7D%5Cright.)

During shut-in period, the indicator function will return value of 1. This will cause the interwell connectivity between the injector-producer well (λ) to be converted into the value of 0. Before getting converted to the value of 0, the interwell connectivity of shut-in `producer-x` will be summed with the interwell connectivity of the non-shut-in `producer-j`. This looks like:

![equation](https://latex.codecogs.com/svg.image?%5Clambda_%7Bi%20j%7D%5E%7B(x)%7D=%5Clambda_%7Bi%20j%7D%5Cleft(1&plus;%5Csum_x%5Clambda_%7Bi%20x%7D%5Cright))

There is also another reference that explains the usage of CRMP for shut-in wells [4].

## Fractional Flow Model
To calculate the predicted oil production rate, as we will usually predict the liquid production rate, we use a fractional flow model for CRMP[2]:

![equation](https://latex.codecogs.com/svg.image?%5Cbegin%7Baligned%7D&f_%7Bo%20j%7D(t)=%5Cleft%5B1&plus;%5Calpha_j%5Cleft(%5Csum_%7Bm=1%7D%5En%5Cleft%5C%7B%5Csum_%7Bi=1%7D%5E%7BN%20I%7D%5Clambda_%7Bi%20j%7Di_i%5Cleft(t_m%5Cright)%5Cright%5C%7D%5Cright)%5E%7B%5Cbeta_j%7D%5Cright%5D%5E%7B-1%7D%5Cend%7Baligned%7D)

From the equation, we will obtain the fraction of the oil, which will be multiplied by the predicted liquid production rate.

## Data
There are two types of data, the synthetic case data and [UNISIM-I](https://www.unisim.cepetro.unicamp.br/benchmarks/en/unisim-i/overview) data, specifically [UNISIM-I-M](https://www.unisim.cepetro.unicamp.br/benchmarks/en/unisim-i/unisim-i-m) [5]. The synthetic case data itself is a field with the grid of `20×20×3` and homogeneous permeability. It has 5 injector and 4 producer wells, and is divided into base (no shut-in period), single shut-in well, and two shut-in wells cases.

## Simple Guideline
Proxy CRM can now be installed by using (big disclaimer on the still ongoing dependencies management):
```
pip install proxy-crm
```

To automatically create a conda environment (due to the aforementioned dependecies problem), you can always use:
```
conda env create --file requirements.yml
```

You can first import the module as any names that you like. In this case, we will use `pCRM`.

```
import proxy_crm_modules as pCRM
```

Then, you can address the class `proxy_crm` as, for example:

```
base_pcrm = pCRM.proxyCRM()
```

Then, you can address the data and fitting process as follows. Do note that synthetic case data is divided into 75%-25% train-test.

```
data_src = "D:/crmProject/crmp_code_test/proxy_crm/data/test/"
oil_prod = pd.read_excel(data_src + 'Base_PROD.xlsx', header=None)
prod = pd.read_excel(data_src + "Base_LIQUID.xlsx", header=None)
inj = pd.read_excel(data_src + "Base_INJ.xlsx", header=None)
time = pd.read_excel(data_src + "TIME.xlsx", header= None)
pressure = pd.read_excel(data_src + "Base_BHP.xlsx", header=None)
wor = pd.read_excel(data_src + "Base_WOR.xlsx", header=None)
cwi = pd.read_excel(data_src + "Base_CWI.xlsx", header=None)

... #train-test splitting

base_pcrm.fit(oil_prod_train, inj_train, press_train, time_train[:,0],num_cores=4, ftol=1e-3)
```

## To-Do List
This project is very much WIP (Work In Progress), so future works will be concentrated on:
- [ ] Fixing issues for fractional flow model
- [ ] Creating connectivity parameter visualization of lambda parameter using networkx and plotly

## License
This repository used [GPLv3 license](https://choosealicense.com/licenses/gpl-3.0/).

## Citations
[1] Gubanova, A., Orlov, D., Koroteev, D., & Shmidt, S. (2022). Proxy Capacitance-Resistance Modeling for Well Production Forecasts in Case of Well Treatments. SPE Journal, 27(06), 3474–3488. [https://doi.org/10.2118/209829-PA](https://doi.org/10.2118/209829-PA)

[2] Lake, L.W., Liang, X., Edgar, T.F., Al-yousef, A.A., Sayarpour, M., & Weber, D. (2007). Optimization Of Oil Production Based On A Capacitance Model Of Production And Injection Rates. [https://doi.org/10.2118/107713-MS](https://doi.org/10.2118/107713-MS)

[3] Sayarpour, M., Zuluaga, E., Kabir, C. S., & Lake, L. W. (2009). The use of capacitance–resistance models for rapid estimation of waterflood performance and optimization. Journal of Petroleum Science and Engineering, 69(3–4), 227–238. [https://doi.org/10.1016/j.petrol.2009.09.006](https://doi.org/10.1016/j.petrol.2009.09.006)

[4] Salehian, M., & Çınar, M. (2019). Reservoir characterization using dynamic capacitance–resistance model with application to shut-in and horizontal wells. Journal of Petroleum Exploration and Production Technology, 9(4), 2811–2830. [https://doi.org/10.1007/s13202-019-0655-4] (https://doi.org/10.1007/s13202-019-0655-4)

[5] Gaspar, A. T., Avansi, G. D., Maschio, C., Santos, A. A., & Schiozer, D. J. (2016). UNISIM-I-M: Benchmark Case Proposal for Oil Reservoir Management Decision-Making. SPE-180848-MS. [https://doi.org/10.2118/180848-MS] (https://doi.org/10.2118/180848-MS)