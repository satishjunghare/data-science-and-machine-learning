# Statsmodels Library

## Overview
Statsmodels is a Python library that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests and statistical data exploration. It includes descriptive statistics, statistical tests, plotting functions, and result statistics for different types of data and estimators. Statsmodels is particularly useful for econometrics, time series analysis, and statistical modeling.

## Installation
```bash
# Basic installation
pip install statsmodels

# With additional dependencies
pip install statsmodels[all]

# Latest version
pip install statsmodels==0.14.0

# From conda
conda install -c conda-forge statsmodels
```

## Key Features
- **Statistical Models**: Linear and non-linear regression models
- **Time Series Analysis**: ARIMA, VAR, and other time series models
- **Statistical Tests**: Hypothesis testing and diagnostic tests
- **Descriptive Statistics**: Summary statistics and data exploration
- **Plotting**: Statistical plots and diagnostic plots
- **Formula Interface**: R-like formula interface for model specification
- **Robust Statistics**: Robust estimation methods
- **Econometrics**: Specialized econometric models and tests

## Core Concepts

### Basic Linear Regression
```python
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])

# Add constant (intercept)
X_with_const = sm.add_constant(X_df)

# Fit linear regression model
model = sm.OLS(y, X_with_const)
results = model.fit()

# Print summary
print(results.summary())

# Access results
print(f"R-squared: {results.rsquared:.4f}")
print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")
print(f"AIC: {results.aic:.4f}")
print(f"BIC: {results.bic:.4f}")

# Get coefficients
print("\nCoefficients:")
print(results.params)

# Get p-values
print("\nP-values:")
print(results.pvalues)

# Get confidence intervals
print("\nConfidence Intervals:")
print(results.conf_int())
```

### Formula Interface (R-like)
```python
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'y': np.random.normal(0, 1, n),
    'x1': np.random.normal(0, 1, n),
    'x2': np.random.normal(0, 1, n),
    'category': np.random.choice(['A', 'B', 'C'], n)
})

# Add some relationships
data['y'] = 2 + 0.5 * data['x1'] + 0.3 * data['x2'] + np.random.normal(0, 0.1, n)

# Fit model using formula interface
model = smf.ols('y ~ x1 + x2', data=data)
results = model.fit()

print(results.summary())

# Model with categorical variables
model_cat = smf.ols('y ~ x1 + x2 + C(category)', data=data)
results_cat = model_cat.fit()

print(results_cat.summary())

# Interaction terms
model_interaction = smf.ols('y ~ x1 * x2', data=data)
results_interaction = model_interaction.fit()

print(results_interaction.summary())
```

## Statistical Tests

### Hypothesis Testing
```python
import statsmodels.api as sm
import numpy as np
from scipy import stats

# Generate sample data
np.random.seed(42)
group1 = np.random.normal(100, 15, 50)
group2 = np.random.normal(105, 15, 50)

# T-test
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"T-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

# ANOVA
from statsmodels.stats.anova import anova_lm
import pandas as pd

# Create data for ANOVA
data_anova = pd.DataFrame({
    'values': np.concatenate([group1, group2]),
    'group': ['Group1'] * 50 + ['Group2'] * 50
})

# Fit model for ANOVA
model_anova = smf.ols('values ~ C(group)', data=data_anova).fit()
anova_table = anova_lm(model_anova, typ=2)
print("\nANOVA Table:")
print(anova_table)

# Chi-square test
from statsmodels.stats.contingency_tables import Table

# Create contingency table
observed = np.array([[10, 20, 30], [15, 25, 35]])
table = Table(observed)
chi2, p_value, dof, expected = table.test_nominal_association()
print(f"\nChi-square test: chi2 = {chi2:.4f}, p-value = {p_value:.4f}")
```

### Diagnostic Tests
```python
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Generate data and fit model
np.random.seed(42)
X = np.random.normal(0, 1, (100, 3))
y = 2 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.normal(0, 0.1, 100)

X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const)
results = model.fit()

# Normality test (Jarque-Bera)
from statsmodels.stats.diagnostic import jarque_bera
jb_stat, jb_p_value = jarque_bera(results.resid)
print(f"Jarque-Bera test: statistic = {jb_stat:.4f}, p-value = {jb_p_value:.4f}")

# Heteroscedasticity test (Breusch-Pagan)
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_p_value, bp_f_stat, bp_f_p_value = het_breuschpagan(results.resid, X_with_const)
print(f"Breusch-Pagan test: statistic = {bp_stat:.4f}, p-value = {bp_p_value:.4f}")

# Multicollinearity (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

X_df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
vif_df = calculate_vif(X_df)
print("\nVariance Inflation Factors:")
print(vif_df)

# Durbin-Watson test for autocorrelation
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(results.resid)
print(f"\nDurbin-Watson statistic: {dw_stat:.4f}")
```

## Time Series Analysis

### ARIMA Models
```python
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate time series data
np.random.seed(42)
n = 1000
t = np.arange(n)
trend = 0.01 * t
seasonal = 10 * np.sin(2 * np.pi * t / 50)
noise = np.random.normal(0, 1, n)
ts_data = trend + seasonal + noise

# Test for stationarity
adf_stat, adf_p_value, _, _, _, _ = adfuller(ts_data)
print(f"ADF test: statistic = {adf_stat:.4f}, p-value = {adf_p_value:.4f}")

# Plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(ts_data, ax=ax1, lags=40)
plot_pacf(ts_data, ax=ax2, lags=40)
plt.tight_layout()
plt.show()

# Fit ARIMA model
model_arima = ARIMA(ts_data, order=(1, 1, 1))
results_arima = model_arima.fit()

print(results_arima.summary())

# Forecast
forecast_steps = 50
forecast = results_arima.forecast(steps=forecast_steps)
print(f"\nForecast for next {forecast_steps} periods:")
print(forecast)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(ts_data, label='Original')
plt.plot(range(len(ts_data), len(ts_data) + forecast_steps), forecast, label='Forecast')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()
```

### VAR Models
```python
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.var_model import VAR
import numpy as np
import pandas as pd

# Generate multivariate time series
np.random.seed(42)
n = 500
t = np.arange(n)

# Create two correlated time series
y1 = 0.5 * np.random.normal(0, 1, n)
y2 = 0.3 * y1 + 0.7 * np.random.normal(0, 1, n)

# Add some lagged effects
for i in range(1, n):
    y1[i] += 0.2 * y1[i-1] + 0.1 * y2[i-1]
    y2[i] += 0.1 * y1[i-1] + 0.3 * y2[i-1]

# Create DataFrame
data_var = pd.DataFrame({'y1': y1, 'y2': y2})

# Fit VAR model
model_var = VAR(data_var)
results_var = model_var.fit(maxlags=5, ic='aic')

print(results_var.summary())

# Granger causality test
from statsmodels.tsa.stattools import grangercausalitytests

# Test if y2 Granger-causes y1
gc_result = grangercausalitytests(data_var, maxlag=5, verbose=False)
print("\nGranger Causality Test (y2 -> y1):")
for lag, result in gc_result.items():
    print(f"Lag {lag}: p-value = {result[0]['ssr_chi2test'][1]:.4f}")

# Impulse response analysis
irf = results_var.irf(periods=20)
irf.plot(orth=True)
plt.show()
```

## Generalized Linear Models (GLM)

### Logistic Regression
```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

# Generate sample data for logistic regression
np.random.seed(42)
n = 1000
X = np.random.normal(0, 1, (n, 3))
beta = np.array([0.5, 0.3, -0.2])
logit = X @ beta + np.random.normal(0, 0.1, n)
prob = 1 / (1 + np.exp(-logit))
y = np.random.binomial(1, prob)

# Create DataFrame
data_logit = pd.DataFrame({
    'y': y,
    'x1': X[:, 0],
    'x2': X[:, 1],
    'x3': X[:, 2]
})

# Fit logistic regression
model_logit = smf.glm('y ~ x1 + x2 + x3', data=data_logit, family=sm.families.Binomial())
results_logit = model_logit.fit()

print(results_logit.summary())

# Odds ratios
odds_ratios = np.exp(results_logit.params)
print("\nOdds Ratios:")
print(odds_ratios)

# Confidence intervals for odds ratios
ci = np.exp(results_logit.conf_int())
ci.columns = ['2.5%', '97.5%']
print("\nConfidence Intervals for Odds Ratios:")
print(ci)
```

### Poisson Regression
```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

# Generate count data
np.random.seed(42)
n = 1000
X = np.random.normal(0, 1, (n, 2))
beta = np.array([0.5, 0.3])
log_lambda = X @ beta + np.random.normal(0, 0.1, n)
lambda_param = np.exp(log_lambda)
y = np.random.poisson(lambda_param)

# Create DataFrame
data_poisson = pd.DataFrame({
    'y': y,
    'x1': X[:, 0],
    'x2': X[:, 1]
})

# Fit Poisson regression
model_poisson = smf.glm('y ~ x1 + x2', data=data_poisson, family=sm.families.Poisson())
results_poisson = model_poisson.fit()

print(results_poisson.summary())

# Rate ratios
rate_ratios = np.exp(results_poisson.params)
print("\nRate Ratios:")
print(rate_ratios)
```

## Robust Statistics

### Robust Regression
```python
import statsmodels.api as sm
import numpy as np
import pandas as pd

# Generate data with outliers
np.random.seed(42)
n = 100
X = np.random.normal(0, 1, (n, 2))
y = 2 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.normal(0, 0.1, n)

# Add outliers
outlier_indices = np.random.choice(n, 5, replace=False)
y[outlier_indices] += 10

# Fit OLS (sensitive to outliers)
X_with_const = sm.add_constant(X)
model_ols = sm.OLS(y, X_with_const)
results_ols = model_ols.fit()

# Fit robust regression
from statsmodels.robust.robust_linear_model import RLM
model_robust = RLM(y, X_with_const, M=sm.robust.norms.HuberT())
results_robust = model_robust.fit()

print("OLS Results:")
print(results_ols.summary().tables[1])

print("\nRobust Regression Results:")
print(results_robust.summary().tables[1])

# Compare coefficients
comparison = pd.DataFrame({
    'OLS': results_ols.params,
    'Robust': results_robust.params
})
print("\nCoefficient Comparison:")
print(comparison)
```

## Descriptive Statistics

### Summary Statistics
```python
import statsmodels.api as sm
import numpy as np
import pandas as pd

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'normal': np.random.normal(0, 1, 1000),
    'uniform': np.random.uniform(0, 1, 1000),
    'exponential': np.random.exponential(1, 1000)
})

# Descriptive statistics
print("Descriptive Statistics:")
print(data.describe())

# Additional statistics
from statsmodels.stats.descriptivestats import describe

desc_stats = describe(data)
print("\nDetailed Statistics:")
print(desc_stats)

# Correlation analysis
correlation_matrix = data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Normality tests
from scipy import stats

for column in data.columns:
    stat, p_value = stats.normaltest(data[column])
    print(f"\nNormality test for {column}:")
    print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
```

## Use Cases
- **Econometrics**: Economic modeling and analysis
- **Time Series Analysis**: Forecasting and trend analysis
- **Statistical Modeling**: Regression and hypothesis testing
- **Quality Control**: Process monitoring and control charts
- **Research**: Academic and research applications
- **Business Analytics**: Business intelligence and reporting
- **Risk Management**: Financial risk modeling
- **Epidemiology**: Medical and health statistics

## Best Practices
1. **Data Quality**: Ensure data quality before analysis
2. **Model Diagnostics**: Always check model assumptions
3. **Multiple Tests**: Use multiple diagnostic tests
4. **Interpretation**: Carefully interpret statistical results
5. **Documentation**: Document model specifications and results
6. **Validation**: Validate models on out-of-sample data
7. **Robustness**: Consider robust methods for outliers
8. **Reporting**: Report effect sizes and confidence intervals

## Advantages
- **Comprehensive**: Wide range of statistical models
- **Academic Quality**: Rigorous statistical methods
- **Formula Interface**: R-like formula specification
- **Diagnostics**: Extensive diagnostic tools
- **Documentation**: Excellent documentation and examples
- **Integration**: Works well with pandas and numpy
- **Active Development**: Regular updates and improvements
- **Community**: Strong academic community

## Limitations
- **Learning Curve**: Steep learning curve for complex models
- **Performance**: May be slower for large datasets
- **Memory Usage**: Can be memory-intensive for large models
- **Visualization**: Limited built-in visualization capabilities
- **Deployment**: Not designed for production deployment

## Related Libraries
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing and statistics
- **Scikit-learn**: Machine learning
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualization
- **Prophet**: Time series forecasting
- **ARCH**: Financial time series modeling 