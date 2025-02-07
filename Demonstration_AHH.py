# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17phiwVkb12vObRNQplZIJ0FxoPbQg7JC
"""

!wget https://raw.githubusercontent.com/saffanaha1/MSNBurr_for_PyMC/main/MSNBurr_distribution.py -O MSNBurr_distribution.py
from MSNBurr_distribution import msnburr as msnburr
from scipy import stats
import math
import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
import pytensor.tensor as pt
import matplotlib.pyplot as plt

sampler_kwargs = {
    "chains": 4,
    "cores": 4,
    "return_inferencedata": True,
    "random_seed": 42,
}

ahh = 'https://raw.githubusercontent.com/saffanaha1/MSNBurr_for_PyMC/main/AHH%20Indo%202023.xlsx'
df = pd.read_excel(ahh)
data = df.iloc[:, 1]
plt.hist(data)
plt.xlabel("AHH")
plt.ylabel("Frekuensi")
plt.title("Histogram AHH Tahun 2023")
plt.show()

print("AHH Tahun 2023")
# Normality test (Shapiro-Wilk test)
shapiro_test_result = stats.shapiro(data)
print(f"Shapiro-Wilk test (p-value): {shapiro_test_result[1]}")
# Skewness test
skewness = stats.skew(data)
print(f"Skewness: {skewness}")


with pm.Model() as modelNormal:
    mu = pm.Normal('mu',74,5)
    sigma = pm.HalfCauchy('sigma',5)

    pm.Normal('Regresi Normal', mu=mu, sigma=sigma, observed=data)
    traceNormal = pm.sample(**sampler_kwargs)

az.plot_trace(traceNormal)
az.summary(traceNormal, round_to=3, hdi_prob=0.95)

with pm.Model() as modelAHH:
    mu = pm.Normal('mu',74,5)
    sigma = pm.HalfCauchy('sigma',5)
    alpha = pm.Gamma('alpha',2,0.5)

    msnburr(
        'MSNBurr',
        mu, sigma, alpha,
        data
    )
    traceMSNBurr = pm.sample(**sampler_kwargs)

az.plot_trace(traceMSNBurr)
az.summary(traceMSNBurr, round_to=3, hdi_prob=0.95)


summary1 = az.summary(traceNormal, round_to=3)
summary2 = az.summary(traceMSNBurr, round_to=3)
mean_values_normal = summary1['mean']
mean_values = summary2['mean']
print("Parameter Distribusi Normal")
print(mean_values_normal)
print("Parameter Distribusi MSNBurr")
print(mean_values)

import scipy.special as special
def expected_value(mu, sigma, alpha):
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    expected_value = mu + sigma/omega*(special.polygamma(0, alpha)-special.polygamma(0, 1)-np.log(alpha))
    return expected_value
def variance(sigma, alpha):
    omega = (1+(1/alpha))**(alpha+1)/np.sqrt(2*np.pi)
    variance = (sigma**2)/(omega**2)*(special.polygamma(1, alpha)+special.polygamma(1,1))
    return variance
def skewness(alpha):
    skewness = (special.polygamma(2, alpha)-special.polygamma(2, 1))/(special.polygamma(1, alpha)+special.polygamma(1, 1))**(3/2)
    return skewness

mu = mean_values[0]
sigma = mean_values[1]
alpha = mean_values[2]

expected_value_ = expected_value(mu, sigma, alpha)
variance_ = variance(sigma, alpha)
mode_ = mu
skewness_ = skewness(alpha)

print("Expected value (E(x)):", expected_value_)
print("Variance (Var(x)):", variance_)
print("Mode:", mode_)
print("Skewness:", skewness_)

mu = pt.scalar('mu')
sigma = pt.scalar('sigma')
alpha = pt.scalar('alpha')
value = pt.scalar('value')

rv = msnburr.dist(mu=mu, sigma=sigma, alpha=alpha)
rv_logp = pm.logp(rv, value)
rv_logp_fn = pm.compile_pymc([value, mu, sigma, alpha], rv_logp)

from scipy.stats import norm
x = np.linspace(50, 83, 100)
y = [math.exp(rv_logp_fn(value=xi, mu=mean_values[0], sigma=mean_values[1], alpha=mean_values[2])) for xi in x]
y1 = [norm.pdf(x, mean_values_normal[0], mean_values_normal[1] ) for x in x]

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(data, label='Data Asli', color='skyblue')

ax2 = ax.twinx()
ax2.plot(x, y1, label='Hasil Estimasi Normal', color='green', linewidth=2)
ax2.plot(x, y, label='Hasil Estimasi MSNBurr', color='red', linewidth=2)
ax2.set_ylim([0, max(y) * 1.1])

ax.set_xlabel("x")
ax.set_ylabel("Density")
ax2.set_ylabel("Estimated Density")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=13)

plt.show()