# NumPy Library

## Overview
NumPy (Numerical Python) is the fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of high-level mathematical functions to operate on these arrays efficiently. NumPy serves as the foundation for most data science and machine learning libraries in Python.

## Installation
```bash
pip install numpy
```

## Key Features
- **Multi-dimensional Arrays**: Efficient n-dimensional array objects
- **Mathematical Functions**: Comprehensive mathematical operations
- **Linear Algebra**: Built-in linear algebra operations
- **Random Number Generation**: Various probability distributions
- **Broadcasting**: Automatic handling of arrays with different shapes
- **Memory Efficiency**: Optimized C code for fast array operations
- **Integration**: Foundation for pandas, scikit-learn, and other libraries

## Core Data Structure: ndarray

### Creating Arrays
```python
import numpy as np

# From Python lists
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# Using NumPy functions
zeros = np.zeros((3, 4))  # 3x4 array of zeros
ones = np.ones((2, 3))    # 2x3 array of ones
empty = np.empty((2, 2))  # 2x2 uninitialized array

# Range-based arrays
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Random arrays
random_arr = np.random.rand(3, 3)  # 3x3 random array
normal_arr = np.random.normal(0, 1, (2, 4))  # Normal distribution
```

### Array Properties and Information
```python
# Array properties
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Shape: {arr.shape}")        # (2, 3)
print(f"Dimensions: {arr.ndim}")    # 2
print(f"Data type: {arr.dtype}")    # int64
print(f"Size: {arr.size}")          # 6
print(f"Memory usage: {arr.nbytes} bytes")

# Array information
print(f"Array: {arr}")
print(f"Transpose: {arr.T}")
print(f"Flattened: {arr.flatten()}")
```

## Array Operations

### Basic Operations
```python
# Arithmetic operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"Addition: {a + b}")         # [5, 7, 9]
print(f"Subtraction: {a - b}")      # [-3, -3, -3]
print(f"Multiplication: {a * b}")   # [4, 10, 18]
print(f"Division: {a / b}")         # [0.25, 0.4, 0.5]
print(f"Power: {a ** 2}")           # [1, 4, 9]

# Comparison operations
print(f"Greater than: {a > 2}")     # [False, False, True]
print(f"Equal to: {a == 2}")        # [False, True, False]
```

### Broadcasting
```python
# Broadcasting allows operations between arrays of different shapes
arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 2

# Broadcasting scalar to array
result = arr * scalar
print(f"Broadcasted multiplication: {result}")

# Broadcasting arrays
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
result = a + b  # Broadcasting (3,) + (3,1) = (3,3)
print(f"Broadcasted addition: {result}")
```

## Mathematical Functions

### Basic Math Functions
```python
arr = np.array([1, 2, 3, 4, 5])

# Trigonometric functions
print(f"Sine: {np.sin(arr)}")
print(f"Cosine: {np.cos(arr)}")
print(f"Tangent: {np.tan(arr)}")

# Exponential and logarithmic
print(f"Exponential: {np.exp(arr)}")
print(f"Natural log: {np.log(arr)}")
print(f"Base 10 log: {np.log10(arr)}")

# Power and roots
print(f"Square root: {np.sqrt(arr)}")
print(f"Square: {np.square(arr)}")
print(f"Power: {np.power(arr, 3)}")
```

### Statistical Functions
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Basic statistics
print(f"Mean: {np.mean(arr)}")
print(f"Median: {np.median(arr)}")
print(f"Standard deviation: {np.std(arr)}")
print(f"Variance: {np.var(arr)}")
print(f"Minimum: {np.min(arr)}")
print(f"Maximum: {np.max(arr)}")
print(f"Sum: {np.sum(arr)}")
print(f"Product: {np.prod(arr)}")

# Percentiles
print(f"25th percentile: {np.percentile(arr, 25)}")
print(f"75th percentile: {np.percentile(arr, 75)}")
```

## Linear Algebra Operations

### Matrix Operations
```python
# Matrix creation
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)
print(f"Matrix multiplication: {C}")

# Alternative matrix multiplication
C = A @ B
print(f"Matrix multiplication (@): {C}")

# Matrix properties
print(f"Determinant: {np.linalg.det(A)}")
print(f"Eigenvalues: {np.linalg.eigvals(A)}")
print(f"Inverse: {np.linalg.inv(A)}")
print(f"Transpose: {A.T}")
```

### Solving Linear Equations
```python
# Solve Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([4, 5])

x = np.linalg.solve(A, b)
print(f"Solution: {x}")

# Verify solution
print(f"Verification: {np.dot(A, x)}")
```

## Array Manipulation

### Reshaping and Resizing
```python
arr = np.arange(12)

# Reshape
reshaped = arr.reshape(3, 4)
print(f"Reshaped: {reshaped}")

# Resize
resized = np.resize(arr, (2, 6))
print(f"Resized: {resized}")

# Flatten
flattened = arr.flatten()
print(f"Flattened: {flattened}")

# Transpose
transposed = reshaped.T
print(f"Transposed: {transposed}")
```

### Indexing and Slicing
```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Basic indexing
print(f"Element at (1, 2): {arr[1, 2]}")

# Slicing
print(f"First row: {arr[0, :]}")
print(f"Second column: {arr[:, 1]}")
print(f"Subarray: {arr[0:2, 1:3]}")

# Boolean indexing
mask = arr > 5
print(f"Elements > 5: {arr[mask]}")

# Fancy indexing
indices = [0, 2]
print(f"Selected rows: {arr[indices, :]}")
```

## Random Number Generation

### Random Distributions
```python
# Set seed for reproducibility
np.random.seed(42)

# Uniform distribution
uniform = np.random.uniform(0, 1, 10)
print(f"Uniform: {uniform}")

# Normal distribution
normal = np.random.normal(0, 1, 10)
print(f"Normal: {normal}")

# Integer random numbers
integers = np.random.randint(1, 100, 10)
print(f"Integers: {integers}")

# Random choice
choices = np.random.choice([1, 2, 3, 4, 5], size=10, p=[0.1, 0.2, 0.3, 0.2, 0.2])
print(f"Random choice: {choices}")

# Shuffle array
arr = np.arange(10)
np.random.shuffle(arr)
print(f"Shuffled: {arr}")
```

## Performance and Memory

### Memory Views and Copies
```python
# Create array
arr = np.array([1, 2, 3, 4, 5])

# View (no copy)
view = arr.view()
view[0] = 10
print(f"Original: {arr}")  # Changed
print(f"View: {view}")     # Changed

# Copy (new memory)
copy = arr.copy()
copy[1] = 20
print(f"Original: {arr}")  # Unchanged
print(f"Copy: {copy}")     # Changed
```

### Efficient Operations
```python
# Vectorized operations (fast)
arr = np.arange(1000000)
result = arr * 2 + 1

# Avoid Python loops for large arrays
# Slow:
# result = []
# for x in arr:
#     result.append(x * 2 + 1)

# Fast:
result = arr * 2 + 1
```

## Use Cases
- **Scientific Computing**: Numerical simulations and calculations
- **Data Analysis**: Foundation for pandas and other data science tools
- **Machine Learning**: Core arrays for scikit-learn, TensorFlow, PyTorch
- **Image Processing**: Multi-dimensional arrays for image data
- **Signal Processing**: Time series and frequency domain analysis
- **Linear Algebra**: Matrix operations and decompositions

## Best Practices
1. **Use Vectorized Operations**: Avoid Python loops when possible
2. **Choose Appropriate Data Types**: Use smaller dtypes when precision allows
3. **Pre-allocate Arrays**: Create arrays of the right size initially
4. **Use Broadcasting**: Leverage NumPy's broadcasting for efficiency
5. **Memory Management**: Be aware of views vs copies
6. **Random Seeds**: Set seeds for reproducible results

## Advantages
- **Performance**: Optimized C code for fast array operations
- **Memory Efficiency**: Compact memory representation
- **Broadcasting**: Automatic handling of different array shapes
- **Rich Functionality**: Comprehensive mathematical operations
- **Integration**: Foundation for the entire Python data science ecosystem
- **Maturity**: Well-established and extensively tested

## Limitations
- **Learning Curve**: Can be complex for beginners
- **Memory Usage**: Can be memory-intensive for very large arrays
- **Single-threaded**: Core operations are single-threaded
- **Type System**: Less flexible than Python lists

## Related Libraries
- **Pandas**: Built on NumPy for data manipulation
- **Scikit-learn**: Uses NumPy arrays for machine learning
- **Matplotlib**: Uses NumPy for plotting data
- **SciPy**: Scientific computing built on NumPy
- **TensorFlow/PyTorch**: Deep learning frameworks using NumPy-like arrays 