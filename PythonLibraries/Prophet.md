# Prophet Library

## Overview
Prophet is an open-source forecasting tool developed by Facebook (Meta) for time series forecasting. It's designed to be easy to use and interpret, making it accessible to both data scientists and business analysts. Prophet handles many of the common challenges in time series forecasting, such as seasonality, holidays, and trend changes, while being robust to missing data and trend shifts.

## Installation
```bash
# Basic installation
pip install prophet

# With additional dependencies
pip install prophet[plotting]

# Latest version
pip install prophet==1.1.4

# From conda
conda install -c conda-forge prophet

# Install dependencies for plotting
pip install plotly
```

## Key Features
- **Automatic Seasonality**: Handles daily, weekly, and yearly seasonality
- **Holiday Effects**: Incorporates holiday and special event effects
- **Trend Changes**: Detects and adapts to trend changes
- **Missing Data**: Robust to missing data and outliers
- **Easy to Use**: Simple API with minimal parameter tuning
- **Interpretable**: Provides clear trend and seasonality components
- **Scalable**: Can handle multiple time series efficiently
- **Uncertainty Intervals**: Provides prediction intervals

## Core Concepts

### Basic Forecasting
```python
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Create sample time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
trend = np.linspace(100, 200, 1000)
seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
noise = np.random.normal(0, 5, 1000)
y = trend + seasonal + noise

# Create DataFrame (Prophet requires 'ds' and 'y' columns)
df = pd.DataFrame({
    'ds': dates,
    'y': y
})

# Initialize and fit Prophet model
model = Prophet()
model.fit(df)

# Create future dataframe for predictions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Display forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecast
fig = model.plot(forecast)
plt.show()

# Plot components
fig2 = model.plot_components(forecast)
plt.show()
```

### Advanced Forecasting with Seasonality
```python
import pandas as pd
import numpy as np
from prophet import Prophet

# Create more complex time series with multiple seasonalities
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')

# Trend with change point
trend = np.concatenate([
    np.linspace(100, 150, 500),  # First trend
    np.linspace(150, 200, 500)   # Second trend
])

# Multiple seasonalities
weekly_seasonal = 5 * np.sin(2 * np.pi * np.arange(1000) / 7)
yearly_seasonal = 15 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
noise = np.random.normal(0, 3, 1000)

y = trend + weekly_seasonal + yearly_seasonal + noise

# Create DataFrame
df = pd.DataFrame({
    'ds': dates,
    'y': y
})

# Initialize Prophet with custom seasonality
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)

# Add custom seasonality
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Fit model
model.fit(df)

# Make predictions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot results
fig = model.plot(forecast)
plt.title('Prophet Forecast with Multiple Seasonalities')
plt.show()

# Plot components
fig2 = model.plot_components(forecast)
plt.show()

# Access trend and seasonality components
print("Trend component:")
print(forecast[['ds', 'trend']].tail())

print("\nYearly seasonality:")
print(forecast[['ds', 'yearly']].tail())
```

## Holiday Effects

### Custom Holidays
```python
import pandas as pd
import numpy as np
from prophet import Prophet

# Create sample data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
trend = np.linspace(100, 200, 1000)
seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
noise = np.random.normal(0, 5, 1000)
y = trend + seasonal + noise

# Add holiday effects
holiday_dates = ['2020-12-25', '2021-12-25', '2020-07-04', '2021-07-04']
for date in holiday_dates:
    idx = pd.to_datetime(date) - pd.to_datetime('2020-01-01')
    if 0 <= idx.days < 1000:
        y[idx.days] -= 20  # Holiday effect

df = pd.DataFrame({
    'ds': dates,
    'y': y
})

# Define holidays
holidays = pd.DataFrame({
    'holiday': 'Christmas',
    'ds': pd.to_datetime(['2020-12-25', '2021-12-25', '2022-12-25']),
    'lower_window': -1,
    'upper_window': 1,
})

# Add Independence Day
independence_day = pd.DataFrame({
    'holiday': 'Independence Day',
    'ds': pd.to_datetime(['2020-07-04', '2021-07-04', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 0,
})

holidays = pd.concat([holidays, independence_day])

# Initialize Prophet with holidays
model = Prophet(holidays=holidays)
model.fit(df)

# Make predictions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
plt.title('Prophet Forecast with Holiday Effects')
plt.show()

# Plot holiday effects
fig2 = model.plot_components(forecast)
plt.show()

# Access holiday effects
print("Holiday effects:")
holiday_effects = forecast[['ds', 'holidays']]
holiday_effects = holiday_effects[holiday_effects['holidays'] != 0]
print(holiday_effects)
```

### Built-in Holidays
```python
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.hdays import USFederalHolidayCalendar

# Create sample data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
trend = np.linspace(100, 200, 1000)
seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
noise = np.random.normal(0, 5, 1000)
y = trend + seasonal + noise

df = pd.DataFrame({
    'ds': dates,
    'y': y
})

# Initialize Prophet with built-in US holidays
model = Prophet(
    holidays=USFederalHolidayCalendar(),
    yearly_seasonality=True,
    weekly_seasonality=True
)

model.fit(df)

# Make predictions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot results
fig = model.plot(forecast)
plt.title('Prophet Forecast with US Federal Holidays')
plt.show()
```

## Trend Changes and Changepoints

### Automatic Changepoint Detection
```python
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Create data with trend changes
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')

# Create trend with multiple change points
trend = np.concatenate([
    np.linspace(100, 120, 200),   # First trend
    np.linspace(120, 80, 300),    # Second trend (decline)
    np.linspace(80, 150, 300),    # Third trend (recovery)
    np.linspace(150, 180, 200)    # Fourth trend
])

seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
noise = np.random.normal(0, 3, 1000)
y = trend + seasonal + noise

df = pd.DataFrame({
    'ds': dates,
    'y': y
})

# Initialize Prophet with changepoint settings
model = Prophet(
    changepoint_prior_scale=0.05,  # Flexibility of trend
    changepoint_range=0.8,         # Range for changepoints
    yearly_seasonality=True
)

model.fit(df)

# Make predictions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
plt.title('Prophet Forecast with Trend Changes')
plt.show()

# Plot changepoints
fig = model.plot(forecast)
a = fig.add_subplot(111)
a.axvline(x=pd.to_datetime('2020-07-01'), color='red', linestyle='--', alpha=0.5)
a.axvline(x=pd.to_datetime('2021-01-01'), color='red', linestyle='--', alpha=0.5)
a.axvline(x=pd.to_datetime('2021-07-01'), color='red', linestyle='--', alpha=0.5)
plt.title('Forecast with Detected Changepoints')
plt.show()

# Access changepoints
print("Detected changepoints:")
print(model.changepoints)
```

### Custom Changepoints
```python
import pandas as pd
import numpy as np
from prophet import Prophet

# Create sample data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
trend = np.linspace(100, 200, 1000)
seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
noise = np.random.normal(0, 5, 1000)
y = trend + seasonal + noise

df = pd.DataFrame({
    'ds': dates,
    'y': y
})

# Define custom changepoints
custom_changepoints = pd.to_datetime([
    '2020-06-01',
    '2020-12-01',
    '2021-06-01'
])

# Initialize Prophet with custom changepoints
model = Prophet(
    changepoints=custom_changepoints,
    yearly_seasonality=True
)

model.fit(df)

# Make predictions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot results
fig = model.plot(forecast)
plt.title('Prophet Forecast with Custom Changepoints')
plt.show()
```

## Uncertainty Intervals and Cross-Validation

### Uncertainty Quantification
```python
import pandas as pd
import numpy as np
from prophet import Prophet

# Create sample data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
trend = np.linspace(100, 200, 1000)
seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
noise = np.random.normal(0, 5, 1000)
y = trend + seasonal + noise

df = pd.DataFrame({
    'ds': dates,
    'y': y
})

# Initialize Prophet with uncertainty settings
model = Prophet(
    interval_width=0.95,  # 95% prediction intervals
    mcmc_samples=1000     # Number of MCMC samples
)

model.fit(df)

# Make predictions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Access uncertainty intervals
print("Forecast with uncertainty intervals:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot with uncertainty
fig = model.plot(forecast)
plt.title('Prophet Forecast with Uncertainty Intervals')
plt.show()

# Plot components with uncertainty
fig2 = model.plot_components(forecast)
plt.show()
```

### Cross-Validation
```python
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Create sample data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
trend = np.linspace(100, 200, 1000)
seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
noise = np.random.normal(0, 5, 1000)
y = trend + seasonal + noise

df = pd.DataFrame({
    'ds': dates,
    'y': y
})

# Initialize and fit model
model = Prophet(yearly_seasonality=True)
model.fit(df)

# Perform cross-validation
df_cv = cross_validation(
    model,
    initial='730 days',    # Initial training period
    period='180 days',     # Spacing between cutoff dates
    horizon='365 days'     # Forecast horizon
)

# Calculate performance metrics
df_p = performance_metrics(df_cv)

print("Cross-validation performance metrics:")
print(df_p.head())

# Plot cross-validation results
from prophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')
plt.title('Cross-validation MAPE')
plt.show()

# Plot forecast vs actual
fig = plot_cross_validation_metric(df_cv, metric='rmse')
plt.title('Cross-validation RMSE')
plt.show()
```

## Multiple Time Series

### Forecasting Multiple Series
```python
import pandas as pd
import numpy as np
from prophet import Prophet

# Create multiple time series
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')

# Series 1
trend1 = np.linspace(100, 200, 1000)
seasonal1 = 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
noise1 = np.random.normal(0, 5, 1000)
y1 = trend1 + seasonal1 + noise1

# Series 2
trend2 = np.linspace(50, 150, 1000)
seasonal2 = 15 * np.sin(2 * np.pi * np.arange(1000) / 365.25 + np.pi/4)
noise2 = np.random.normal(0, 8, 1000)
y2 = trend2 + seasonal2 + noise2

# Create DataFrame with multiple series
df1 = pd.DataFrame({
    'ds': dates,
    'y': y1,
    'series': 'A'
})

df2 = pd.DataFrame({
    'ds': dates,
    'y': y2,
    'series': 'B'
})

df_combined = pd.concat([df1, df2], ignore_index=True)

# Function to forecast multiple series
def forecast_multiple_series(df, series_column='series'):
    forecasts = {}
    
    for series_name in df[series_column].unique():
        # Filter data for current series
        series_data = df[df[series_column] == series_name].copy()
        series_data = series_data[['ds', 'y']]
        
        # Initialize and fit model
        model = Prophet(yearly_seasonality=True)
        model.fit(series_data)
        
        # Make predictions
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        
        forecasts[series_name] = forecast
    
    return forecasts

# Forecast all series
forecasts = forecast_multiple_series(df_combined)

# Plot results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for i, (series_name, forecast) in enumerate(forecasts.items()):
    axes[i].plot(df_combined[df_combined['series'] == series_name]['ds'], 
                df_combined[df_combined['series'] == series_name]['y'], 
                label='Actual', alpha=0.7)
    axes[i].plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
    axes[i].fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                        alpha=0.3, color='red', label='95% CI')
    axes[i].set_title(f'Series {series_name}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Use Cases
- **Business Forecasting**: Sales, revenue, and demand forecasting
- **Web Analytics**: Website traffic and user engagement
- **Financial Markets**: Stock prices and trading volumes
- **Supply Chain**: Inventory and logistics planning
- **Energy**: Electricity consumption and renewable energy
- **Healthcare**: Patient admissions and disease outbreaks
- **Retail**: Store traffic and product demand
- **Transportation**: Passenger counts and route optimization

## Best Practices
1. **Data Quality**: Ensure clean, consistent time series data
2. **Seasonality**: Identify and model appropriate seasonalities
3. **Holidays**: Include relevant holiday effects
4. **Changepoints**: Monitor for trend changes
5. **Cross-Validation**: Use cross-validation for model evaluation
6. **Uncertainty**: Consider prediction intervals
7. **Interpretability**: Analyze trend and seasonality components
8. **Regular Updates**: Retrain models with new data

## Advantages
- **Easy to Use**: Simple API with minimal parameter tuning
- **Robust**: Handles missing data and outliers well
- **Interpretable**: Clear trend and seasonality decomposition
- **Automatic**: Detects seasonality and changepoints automatically
- **Scalable**: Can handle multiple time series
- **Uncertainty**: Provides prediction intervals
- **Holiday Effects**: Built-in holiday modeling
- **Documentation**: Excellent documentation and examples

## Limitations
- **Assumptions**: Assumes additive or multiplicative seasonality
- **Computational Cost**: Can be slow for very large datasets
- **Flexibility**: Less flexible than custom models
- **Interpretation**: May not capture complex patterns
- **Dependencies**: Requires specific data format
- **Black Box**: Limited control over internal algorithms

## Related Libraries
- **Statsmodels**: Statistical time series models (ARIMA, VAR)
- **Scikit-learn**: Machine learning for time series
- **TensorFlow**: Deep learning for time series
- **PyTorch**: Alternative deep learning framework
- **Pandas**: Data manipulation and time series functionality
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **Plotly**: Interactive visualizations 