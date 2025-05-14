import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
sales = np.random.normal(loc=100, scale=20, size=100).cumsum() + 500
expenses = sales * 0.6 + np.random.normal(loc=0, scale=50, size=100)
profit = sales - expenses
category = np.random.choice(['A', 'B', 'C', 'D'], size=100)
region = np.random.choice(['North', 'South', 'East', 'West'], size=100)

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'sales': sales,
    'expenses': expenses,
    'profit': profit,
    'category': category,
    'region': region
})

# Save to CSV
df.to_csv('sample_data/sales_data.csv', index=False)