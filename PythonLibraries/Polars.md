# Polars Library

## Overview
Polars is a fast DataFrame library implemented in Rust with Python bindings, designed to be a high-performance alternative to Pandas. It provides a familiar DataFrame API while offering significant performance improvements through its Rust backend, lazy evaluation, and optimized memory management. Polars excels at handling large datasets and complex data operations with minimal memory usage.

## Installation
```bash
# Basic installation
pip install polars

# With additional features
pip install "polars[all]"

# For specific features
pip install "polars[pyarrow]"  # PyArrow integration
pip install "polars[connectorx]"  # Fast database connectors
pip install "polars[fsspec]"  # File system support
```

## Key Features
- **High Performance**: Rust backend for fast data processing
- **Memory Efficient**: Optimized memory usage and garbage collection
- **Lazy Evaluation**: Deferred computation for better performance
- **Type Safety**: Strong typing with automatic type inference
- **Parallel Processing**: Multi-threaded operations by default
- **Arrow Integration**: Native Apache Arrow support
- **SQL Support**: SQL-like operations on DataFrames
- **Streaming**: Process data in chunks for large datasets

## Core Data Structures

### DataFrame Creation
```python
import polars as pl
import numpy as np

# Create DataFrame from dictionary
df = pl.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'city': ['NYC', 'LA', 'Chicago', 'Boston'],
    'salary': [50000, 60000, 70000, 55000]
})
print(df)

# Create from list of lists
data = [
    ['Alice', 25, 'NYC', 50000],
    ['Bob', 30, 'LA', 60000],
    ['Charlie', 35, 'Chicago', 70000]
]
df = pl.DataFrame(data, schema=['name', 'age', 'city', 'salary'])

# Create from NumPy arrays
names = np.array(['Alice', 'Bob', 'Charlie'])
ages = np.array([25, 30, 35])
df = pl.DataFrame({
    'name': names,
    'age': ages
})
```

### Series Operations
```python
# Create Series
s = pl.Series('numbers', [1, 2, 3, 4, 5])
print(s)

# Series operations
squared = s ** 2
filtered = s.filter(s > 3)
print(f"Squared: {squared}")
print(f"Filtered: {filtered}")

# Type conversion
s_float = s.cast(pl.Float64)
print(f"Float series: {s_float}")
```

## Data Operations

### Basic Operations
```python
# View data
print(df.head())
print(df.tail())
print(df.describe())

# Select columns
selected = df.select(['name', 'age'])
print(selected)

# Filter rows
young_people = df.filter(pl.col('age') < 30)
print(young_people)

# Add/modify columns
df_with_bonus = df.with_columns([
    (pl.col('salary') * 1.1).alias('salary_with_bonus'),
    (pl.col('age') > 30).alias('is_senior')
])
print(df_with_bonus)
```

### Aggregations
```python
# Group by operations
grouped = df.group_by('city').agg([
    pl.col('salary').mean().alias('avg_salary'),
    pl.col('age').count().alias('count'),
    pl.col('salary').sum().alias('total_salary')
])
print(grouped)

# Multiple aggregations
stats = df.group_by('city').agg([
    pl.col('salary').mean().alias('mean_salary'),
    pl.col('salary').std().alias('std_salary'),
    pl.col('salary').min().alias('min_salary'),
    pl.col('salary').max().alias('max_salary')
])
print(stats)
```

### Window Functions
```python
# Window functions
df_with_rank = df.with_columns([
    pl.col('salary').rank().alias('salary_rank'),
    pl.col('salary').rank(method='dense').alias('dense_rank')
])
print(df_with_rank)

# Rolling operations
df_with_rolling = df.with_columns([
    pl.col('salary').rolling_mean(window_size=2).alias('rolling_avg_salary')
])
print(df_with_rolling)
```

## Lazy Evaluation

### LazyFrame Operations
```python
# Create LazyFrame
lazy_df = df.lazy()

# Build query
query = lazy_df.filter(pl.col('age') > 25)\
    .group_by('city')\
    .agg([
        pl.col('salary').mean().alias('avg_salary'),
        pl.col('age').count().alias('count')
    ])\
    .sort('avg_salary', descending=True)

# Execute query
result = query.collect()
print(result)

# Show query plan
print(query.explain())
```

### Optimization
```python
# Multiple operations in one query
optimized_query = lazy_df\
    .filter(pl.col('salary') > 50000)\
    .with_columns([
        (pl.col('salary') * 1.1).alias('new_salary'),
        pl.col('age').cast(pl.Float64).alias('age_float')
    ])\
    .group_by('city')\
    .agg([
        pl.col('new_salary').mean().alias('avg_new_salary'),
        pl.col('age_float').mean().alias('avg_age')
    ])\
    .sort('avg_new_salary')

result = optimized_query.collect()
print(result)
```

## Data I/O

### Reading Data
```python
# Read CSV
df = pl.read_csv('data.csv')
print(df.head())

# Read with options
df = pl.read_csv('data.csv', 
                 separator=',',
                 has_header=True,
                 null_values=['NA', 'null'])

# Read Parquet
df = pl.read_parquet('data.parquet')

# Read JSON
df = pl.read_json('data.json')

# Read Excel
df = pl.read_excel('data.xlsx', sheet_name='Sheet1')

# Read from database
df = pl.read_database(
    query="SELECT * FROM users",
    connection="postgresql://user:pass@localhost/db"
)
```

### Writing Data
```python
# Write to CSV
df.write_csv('output.csv')

# Write to Parquet
df.write_parquet('output.parquet')

# Write to JSON
df.write_json('output.json')

# Write to Excel
df.write_excel('output.xlsx')

# Write to database
df.write_database(
    table_name='users',
    connection="postgresql://user:pass@localhost/db"
)
```

## Advanced Operations

### Joins
```python
# Create another DataFrame
df2 = pl.DataFrame({
    'city': ['NYC', 'LA', 'Chicago', 'Boston'],
    'state': ['NY', 'CA', 'IL', 'MA'],
    'population': [8400000, 4000000, 2700000, 675000]
})

# Inner join
joined = df.join(df2, on='city', how='inner')
print(joined)

# Left join
left_joined = df.join(df2, on='city', how='left')
print(left_joined)

# Multiple join conditions
joined_multi = df.join(df2, on=['city'], how='inner')
```

### Pivot Operations
```python
# Pivot table
pivot = df.pivot(
    values='salary',
    index='city',
    columns='age',
    aggregate_function='mean'
)
print(pivot)

# Melt (unpivot)
melted = df.melt(
    id_vars=['name', 'city'],
    value_vars=['age', 'salary'],
    variable_name='metric',
    value_name='value'
)
print(melted)
```

### String Operations
```python
# String operations
df_strings = df.with_columns([
    pl.col('name').str.to_uppercase().alias('name_upper'),
    pl.col('name').str.lengths().alias('name_length'),
    pl.col('name').str.contains('a').alias('contains_a')
])
print(df_strings)

# String splitting
df_split = df.with_columns([
    pl.col('name').str.split(' ').alias('name_parts')
])
print(df_split)
```

## Performance Features

### Memory Management
```python
# Check memory usage
print(f"Memory usage: {df.estimated_size()} bytes")

# Optimize memory usage
df_optimized = df.with_columns([
    pl.col('age').cast(pl.UInt8),  # Use smaller integer type
    pl.col('salary').cast(pl.UInt32)  # Use appropriate type
])

# Garbage collection
import gc
gc.collect()
```

### Parallel Processing
```python
# Enable parallel processing
pl.Config.set_global_string_cache(True)

# Parallel group by
parallel_result = df.group_by('city', maintain_order=False)\
    .agg(pl.col('salary').mean())\
    .sort('city')
print(parallel_result)
```

## SQL Integration

### SQL Operations
```python
# SQL-like operations
sql_result = df.sql("""
    SELECT city, AVG(salary) as avg_salary, COUNT(*) as count
    FROM self
    WHERE age > 25
    GROUP BY city
    ORDER BY avg_salary DESC
""")
print(sql_result)

# Complex SQL query
complex_sql = df.sql("""
    SELECT 
        city,
        AVG(salary) as avg_salary,
        MAX(salary) as max_salary,
        MIN(salary) as min_salary,
        COUNT(*) as employee_count
    FROM self
    GROUP BY city
    HAVING COUNT(*) > 1
    ORDER BY avg_salary DESC
""")
print(complex_sql)
```

## Streaming Operations

### Process Large Files
```python
# Stream processing for large files
def process_large_file():
    for chunk in pl.read_csv_batched('large_file.csv', batch_size=10000):
        # Process each chunk
        processed = chunk.filter(pl.col('value') > 0)
        yield processed

# Collect results
results = []
for chunk_result in process_large_file():
    results.append(chunk_result)

# Combine results
final_df = pl.concat(results)
```

## Use Cases
- **Big Data Processing**: Handle large datasets efficiently
- **Data Analysis**: Fast exploratory data analysis
- **ETL Pipelines**: Extract, transform, and load data
- **Real-time Analytics**: Process streaming data
- **Data Science**: Preprocessing for machine learning
- **Business Intelligence**: Fast reporting and analytics
- **Data Migration**: Convert between different data formats

## Best Practices
1. **Use Lazy Evaluation**: Leverage lazy evaluation for complex operations
2. **Type Optimization**: Use appropriate data types to save memory
3. **Batch Processing**: Process large files in chunks
4. **Memory Management**: Monitor memory usage and optimize when needed
5. **Query Optimization**: Build efficient queries with lazy evaluation
6. **Parallel Processing**: Enable parallel processing for better performance
7. **Caching**: Cache frequently used data in memory

## Advantages
- **Performance**: Significantly faster than Pandas for most operations
- **Memory Efficiency**: Lower memory usage and better garbage collection
- **Type Safety**: Strong typing with automatic type inference
- **Lazy Evaluation**: Optimized query execution
- **Parallel Processing**: Multi-threaded operations by default
- **Arrow Integration**: Native Apache Arrow support
- **SQL Support**: SQL-like operations on DataFrames
- **Streaming**: Process data in chunks for large datasets

## Limitations
- **Learning Curve**: Different API from Pandas
- **Ecosystem**: Smaller ecosystem compared to Pandas
- **Documentation**: Less extensive documentation and examples
- **Community**: Smaller community compared to Pandas
- **Compatibility**: May not be compatible with all Pandas-based libraries

## Related Libraries
- **Pandas**: Traditional DataFrame library
- **Apache Arrow**: Columnar memory format
- **NumPy**: Numerical computing foundation
- **Dask**: Parallel computing framework
- **Vaex**: Alternative high-performance DataFrame library
- **PyArrow**: Python bindings for Apache Arrow
- **ConnectorX**: Fast database connectors 