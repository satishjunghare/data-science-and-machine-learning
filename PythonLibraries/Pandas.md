# Pandas Library

## Overview
Pandas is an open-source Python library for data manipulation and analysis. It provides powerful data structures for working with structured data, making it the foundation of data science workflows in Python. Pandas excels at handling tabular data from various sources including CSV files, Excel spreadsheets, SQL databases, and more.

## Installation
```bash
pip install pandas
```

## Key Features
- **Data Structures**: Series (1D) and DataFrame (2D) for efficient data handling
- **Data Import/Export**: Read from and write to multiple file formats
- **Data Cleaning**: Handle missing values, duplicates, and data type conversions
- **Data Analysis**: Powerful aggregation, grouping, and statistical functions
- **Time Series**: Built-in support for time series data and operations
- **Integration**: Works seamlessly with NumPy, Matplotlib, and other scientific libraries

## Core Data Structures

### Series
A one-dimensional labeled array capable of holding any data type.

```python
import pandas as pd
import numpy as np

# Creating a Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

# Series with custom index
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(s)

# Series from dictionary
data = {'a': 1, 'b': 2, 'c': 3}
s = pd.Series(data)
print(s)
```

### DataFrame
A two-dimensional labeled data structure with columns that can be of different types.

```python
# Creating a DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['NYC', 'LA', 'Chicago', 'Boston'],
    'Salary': [50000, 60000, 70000, 55000]
})
print(df)

# DataFrame from list of lists
data = [
    ['Alice', 25, 'NYC', 50000],
    ['Bob', 30, 'LA', 60000],
    ['Charlie', 35, 'Chicago', 70000]
]
df = pd.DataFrame(data, columns=['Name', 'Age', 'City', 'Salary'])
print(df)
```

## Data Import and Export

### Reading Data
```python
# Read CSV file
df = pd.read_csv('data.csv')

# Read Excel file
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Read JSON file
df = pd.read_json('data.json')

# Read SQL database
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table_name", conn)

# Read from URL
df = pd.read_csv('https://raw.githubusercontent.com/dataset.csv')
```

### Writing Data
```python
# Save to CSV
df.to_csv('output.csv', index=False)

# Save to Excel
df.to_excel('output.xlsx', sheet_name='Data', index=False)

# Save to JSON
df.to_json('output.json', orient='records')

# Save to SQL database
df.to_sql('table_name', conn, if_exists='replace', index=False)
```

## Data Manipulation

### Basic Operations
```python
# View data
print(df.head())  # First 5 rows
print(df.tail())  # Last 5 rows
print(df.info())  # Data types and memory usage
print(df.describe())  # Statistical summary

# Selecting data
print(df['Name'])  # Single column
print(df[['Name', 'Age']])  # Multiple columns
print(df.iloc[0:3])  # Rows by position
print(df.loc[df['Age'] > 30])  # Rows by condition

# Adding/removing columns
df['Department'] = 'IT'  # Add new column
df = df.drop('Department', axis=1)  # Remove column
```

### Data Cleaning
```python
# Handle missing values
print(df.isnull().sum())  # Count missing values
df = df.dropna()  # Remove rows with missing values
df = df.fillna(0)  # Fill missing values with 0
df['Age'] = df['Age'].fillna(df['Age'].mean())  # Fill with mean

# Remove duplicates
df = df.drop_duplicates()

# Data type conversion
df['Age'] = df['Age'].astype(int)
df['Salary'] = df['Salary'].astype(float)

# String operations
df['Name'] = df['Name'].str.upper()
df['Name'] = df['Name'].str.strip()
```

### Filtering and Sorting
```python
# Filtering
high_salary = df[df['Salary'] > 60000]
young_employees = df[df['Age'] < 30]
nyc_employees = df[df['City'] == 'NYC']

# Multiple conditions
filtered = df[(df['Age'] > 25) & (df['Salary'] > 50000)]

# Sorting
df_sorted = df.sort_values('Salary', ascending=False)
df_sorted = df.sort_values(['Age', 'Salary'], ascending=[True, False])
```

## Data Analysis

### Aggregation and Grouping
```python
# Basic statistics
print(df['Salary'].mean())
print(df['Salary'].median())
print(df['Salary'].std())
print(df['Salary'].min())
print(df['Salary'].max())

# Group by operations
city_stats = df.groupby('City')['Salary'].agg(['mean', 'count', 'std'])
print(city_stats)

# Multiple aggregations
summary = df.groupby('City').agg({
    'Age': ['mean', 'min', 'max'],
    'Salary': ['mean', 'sum', 'count']
})
print(summary)

# Pivot tables
pivot = df.pivot_table(
    values='Salary',
    index='City',
    columns='Age',
    aggfunc='mean',
    fill_value=0
)
print(pivot)
```

### Time Series Operations
```python
# Create time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts_data = pd.DataFrame({
    'Date': dates,
    'Value': np.random.randn(100).cumsum()
})

# Set date as index
ts_data.set_index('Date', inplace=True)

# Time-based operations
monthly_avg = ts_data.resample('M').mean()
weekly_sum = ts_data.resample('W').sum()

# Rolling statistics
rolling_mean = ts_data['Value'].rolling(window=7).mean()
rolling_std = ts_data['Value'].rolling(window=7).std()
```

## Advanced Operations

### Merging and Joining
```python
# Create sample dataframes
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana']
})

df2 = pd.DataFrame({
    'ID': [1, 2, 3, 5],
    'Department': ['IT', 'HR', 'Finance', 'Marketing']
})

# Inner join
merged = pd.merge(df1, df2, on='ID', how='inner')

# Left join
left_merged = pd.merge(df1, df2, on='ID', how='left')

# Concatenate dataframes
combined = pd.concat([df1, df2], axis=0, ignore_index=True)
```

### Data Transformation
```python
# Apply function to column
df['Salary_K'] = df['Salary'].apply(lambda x: x/1000)

# Apply function to multiple columns
df[['Age', 'Salary']] = df[['Age', 'Salary']].apply(pd.to_numeric)

# Vectorized operations
df['Age_Next_Year'] = df['Age'] + 1
df['Salary_Bonus'] = df['Salary'] * 1.1

# Conditional operations
df['Salary_Category'] = np.where(df['Salary'] > 60000, 'High', 'Low')
```

## Performance Optimization

### Efficient Operations
```python
# Use vectorized operations instead of loops
# Slow (don't do this)
for i in range(len(df)):
    df.loc[i, 'New_Column'] = df.loc[i, 'Salary'] * 1.1

# Fast (do this)
df['New_Column'] = df['Salary'] * 1.1

# Use query for complex filtering
result = df.query('Age > 25 and Salary > 50000')

# Use eval for complex expressions
df.eval('Salary_Plus_Bonus = Salary * 1.1', inplace=True)
```

## Use Cases
- **Data Cleaning**: Handle messy, real-world datasets
- **Exploratory Data Analysis**: Understand data structure and patterns
- **Data Transformation**: Reshape and prepare data for analysis
- **Statistical Analysis**: Perform descriptive and inferential statistics
- **Time Series Analysis**: Analyze temporal data patterns
- **Data Visualization**: Prepare data for plotting with matplotlib/seaborn
- **Machine Learning**: Preprocess data for ML algorithms

## Best Practices
1. **Data Types**: Use appropriate data types to save memory
2. **Missing Values**: Handle missing values explicitly
3. **Vectorization**: Use vectorized operations instead of loops
4. **Memory Usage**: Monitor memory usage with large datasets
5. **Data Validation**: Validate data after import and transformation
6. **Documentation**: Document data transformations and assumptions

## Advantages
- **Efficiency**: Fast operations on large datasets
- **Flexibility**: Handle various data formats and structures
- **Integration**: Works seamlessly with other data science libraries
- **Rich Functionality**: Comprehensive set of data manipulation tools
- **Community Support**: Large, active community and extensive documentation
- **Performance**: Optimized C code under the hood

## Limitations
- **Memory Usage**: Can be memory-intensive for very large datasets
- **Learning Curve**: Steep learning curve for advanced operations
- **Performance**: Slower than specialized libraries for specific tasks
- **Complexity**: Some operations can be complex and hard to debug

## Related Libraries
- **NumPy**: Numerical computing foundation
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms
- **SQLAlchemy**: Database operations
- **Dask**: Parallel computing for large datasets