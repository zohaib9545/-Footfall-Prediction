import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# STEP 1: Load and clean data
url = "https://data.melbourne.vic.gov.au/api/v2/catalog/datasets/pedestrian-counting-system-monthly-counts-per-hour/exports/csv"
df = pd.read_csv(url, delimiter=';', parse_dates=['sensing_date'])

# Rename columns for easier use
df.columns = ['id', 'location_id', 'sensing_date', 'hour', 'direction_1', 'direction_2',
              'pedestrian_count', 'sensor_name', 'location']

# Convert types
df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
df['pedestrian_count'] = pd.to_numeric(df['pedestrian_count'], errors='coerce')
df.dropna(subset=['sensing_date', 'pedestrian_count'], inplace=True)

# Create time features
df['date'] = df['sensing_date'].dt.date
df['weekday'] = df['sensing_date'].dt.day_name()
df['month'] = df['sensing_date'].dt.month_name()
df['hour'] = df['hour'].astype(int)

# Aggregate daily total footfall
daily_footfall = df.groupby('date')['pedestrian_count'].sum()

# ====================
# 📊 PART 1: Footfall Over Time
# ====================
plt.figure(figsize=(12, 4))
daily_footfall.plot()
plt.title("Footfall Over Time")
plt.xlabel("Date")
plt.ylabel("Total Visitors")
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================
# 📊 PART 2: Footfall by Day of the Week
# ====================
weekday_footfall = df.groupby('weekday')['pedestrian_count'].sum()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_footfall = weekday_footfall.reindex(weekday_order)

plt.figure(figsize=(8, 4))
weekday_footfall.plot(kind='bar', color='skyblue')
plt.title("Footfall by Day of the Week")
plt.ylabel("Visitors")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ====================
# ⏰ PART 3: Footfall by Time of Day
# ====================
hourly_footfall = df.groupby('hour')['pedestrian_count'].mean()

plt.figure(figsize=(10, 4))
hourly_footfall.plot(kind='line', marker='o', color='orange')
plt.title("Average Footfall by Time of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Visitors")
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================
# 📅 PART 4: Monthly Footfall Trend
# ====================
monthly_footfall = df.groupby('month')['pedestrian_count'].sum()
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_footfall = monthly_footfall.reindex(month_order)

plt.figure(figsize=(10, 4))
monthly_footfall.plot(kind='bar', color='purple')
plt.title("Monthly Footfall Trend")
plt.ylabel("Visitors")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ====================
# 🤖 PART 5: AI Forecasting - Actual vs Predicted
# ====================
# Convert daily data to time series
ts = daily_footfall.asfreq('D').fillna(method='ffill')

# Train ARIMA model
model = ARIMA(ts, order=(7, 1, 1))
fitted = model.fit()

# Predict
pred = fitted.predict(start=ts.index[0], end=ts.index[-1], dynamic=False)

plt.figure(figsize=(12, 4))
plt.plot(ts, label="Actual")
plt.plot(pred, label="Predicted", color='red')
plt.title("Actual vs Predicted Footfall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================
# 📈 PART 6: Forecast Future Footfall
# ====================
forecast_days = 30
future = fitted.forecast(steps=forecast_days)

plt.figure(figsize=(12, 4))
plt.plot(ts[-60:], label="Last 60 Days")
plt.plot(pd.date_range(ts.index[-1], periods=forecast_days+1, freq='D')[1:], future, label="Forecast", color='green')
plt.title("Footfall Forecast for Next 30 Days")
plt.xlabel("Date")
plt.ylabel("Visitor Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
