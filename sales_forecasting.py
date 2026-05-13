import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
df = pd.read_csv(
    r"C:\Users\leela\OneDrive\Desktop\Sales_Forecasting_Project\data\sales_forecast.csv"
)

# Show first rows
print(df.head())

# Convert Order Date properly
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

# Sort by date
df = df.sort_values('Order Date')

# Monthly sales calculation
monthly_sales = df.groupby(
    df['Order Date'].dt.to_period('M')
)['Sales'].sum()

# Convert to dataframe
monthly_sales = monthly_sales.reset_index()

# Convert date to string
monthly_sales['Order Date'] = monthly_sales['Order Date'].astype(str)

# Create numeric month index
monthly_sales['Month_Index'] = np.arange(len(monthly_sales))

# Features and target
X = monthly_sales[['Month_Index']]
y = monthly_sales['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Forecast next 6 months
future_index = np.arange(
    len(monthly_sales),
    len(monthly_sales) + 6
).reshape(-1, 1)

forecast = model.predict(future_index)

# Print forecast values
print("\nFuture Sales Forecast:")
print(forecast)

# Plot actual sales
plt.figure(figsize=(10, 5))

plt.plot(
    monthly_sales['Month_Index'],
    y,
    marker='o',
    label='Actual Sales'
)

# Plot forecast
plt.plot(
    future_index,
    forecast,
    marker='o',
    linestyle='dashed',
    label='Forecast Sales'
)

# Labels
plt.xlabel("Month Index")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()

# Save graph
plt.savefig(
    r"C:\Users\leela\OneDrive\Desktop\Sales_Forecasting_Project\outputs\forecast.png"
)

# Show graph
plt.show()