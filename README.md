# 🚀 NumPy: The Ultimate Guide to Mastering Python's Scientific Computing Engine
![NumPy Logo](https://numpy.org/images/logo.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.26.0-blue?logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?logo=python)
![License](https://img.shields.io/badge/License-MIT-red)
![Downloads](https://img.shields.io/pypi/dm/numpy?color=yellow)
---

## 📖 Table of Contents
1. [Theoretical Foundations](#theoretical-foundations)
2. [Installation Guide](#installation-guide)
3. [Core Concepts](#core-concepts)
4. [Array Operations](#array-operations)
5. [Mathematical Functions](#mathematical-functions)
6. [Linear Algebra](#linear-algebra)
7. [Random Numbers](#random-numbers)
8. [File I/O](#file-io)
9. [Performance Optimization](#performance-optimization)
10. [Best Practices](#best-practices)
11. [Advanced Topics](#advanced-topics)
12. [Learning Roadmap](#learning-roadmap)

---

## 🧠 Theoretical Foundations

### What is NumPy?
NumPy (Numerical Python) is the backbone of scientific computing in Python. It provides:
- **N-dimensional arrays** (`ndarray`): Homogeneous, fixed-size, typed arrays for efficient computation
- **Vectorized operations**: Element-wise operations without explicit loops
- **Broadcasting**: Automatic alignment of arrays of different shapes
- **Linear algebra**: Matrix operations, decompositions, and solvers
- **Random number generation**: High-performance PRNGs for simulations

### Why NumPy?
- **Speed**: Written in C, 100-1000x faster than native Python loops
- **Memory efficiency**: Stores data in contiguous blocks
- **Convenience**: Unified API for numerical operations
- **Ecosystem**: Foundation for pandas, scikit-learn, TensorFlow, etc.

### Key Concepts
| Concept          | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| **Array**         | Homogeneous collection of elements of same type                            |
| **Shape**         | Tuple indicating array dimensions (rows, columns, depth, etc.)             |
| **Stride**        | Bytes to step in each dimension to access next element                     |
| **Contiguity**    | Memory layout (C-style row-major or Fortran-style column-major)            |
| **Universal Func**| Functions that operate element-wise on arrays                              |

---

## 🛠️ Installation Guide

```bash
# Using pip
pip install numpy

# Using conda
conda install numpy
```

---

## 🧮 Core Concepts

### Creating Arrays
```python
import numpy as np

# 1D array
arr = np.array([1, 2, 3])

# 2D array
matrix = np.array([[1, 2], [3, 4]])

# Zeros and ones
zeros = np.zeros((3, 4))
ones = np.ones((2, 2), dtype=int)

# Range arrays
arange = np.arange(0, 10, 2)  # 0-8 stepping by 2
linspace = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
```

### Array Attributes
```python
arr.shape      # (3,)
arr.dtype      # dtype('int64')
arr.ndim       # 1
arr.size       # 3
arr.itemsize   # 8 bytes
```

### Reshaping & Slicing
```python
reshaped = arr.reshape(3, 1)  # Column vector
sliced = arr[1:3]             # [2, 3]
strided = arr[::2]            # [1, 3]
```

---

## 📐 Array Operations

### Element-wise Operations
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b         # [5, 7, 9]
a * b         # [4, 10, 18]
a ** 2        # [1, 4, 9]
np.sin(a)     # [sin(1), sin(2), sin(3)]
```

### Broadcasting Rules
1. **Dimension matching**: Smaller array is padded with 1s
2. **Size compatibility**: Dimensions must be equal or one of them 1
3. **Shape expansion**: Smaller array is stretched to match larger shape

```python
# Valid broadcasting
np.array([1, 2, 3]) + np.array([10])       # [11, 12, 13]
np.array([[1], [2], [3]]) + np.array([4, 5])  # 3x2 matrix
```

---

## 📊 Mathematical Functions

### Aggregation
```python
arr.sum()       # Sum all elements
arr.mean()      # Average
arr.max(axis=0) # Max along columns
arr.std()       # Standard deviation
```

### Trigonometric
```python
np.sin(arr)
np.cos(arr)
np.tan(arr)
```

### Exponential
```python
np.exp(arr)     # e^x
np.log(arr)     # Natural log
np.sqrt(arr)    # Square root
```

---

## 🧮 Linear Algebra

```python
# Matrix multiplication
np.dot(a, b)    # Dot product
a @ b           # Same as above

# Matrix decompositions
np.linalg.inv(matrix)    # Inverse
np.linalg.eig(matrix)    # Eigenvalues/vectors
np.linalg.svd(matrix)    # Singular value decomposition

# Solving systems
np.linalg.solve(A, b)    # Ax = b
```

---

## 🎲 Random Numbers

```python
# Basic distributions
np.random.rand(3)         # Uniform [0,1)
np.random.randn(3)        # Standard normal
np.random.randint(0, 10, 5) # Random integers

# Seeding
np.random.seed(42)
```

---

## 💾 File I/O

```python
# Binary files
np.save('array.npy', arr)
loaded = np.load('array.npy')

# Text files
np.savetxt('data.csv', arr, delimiter=',')
loaded = np.loadtxt('data.csv', delimiter=',')
```

---

## ⚡ Performance Optimization

1. **Vectorize everything**: Replace loops with array operations
2. **Preallocate memory**: Use `np.empty()` instead of appending
3. **Use in-place operations**: `a += b` instead of `a = a + b`
4. **Leverage broadcasting**: Avoid explicit reshaping when possible
5. **Profile critical code**: Use `%timeit` in Jupyter

---

## 📜 Best Practices

1. **Use meaningful variable names**: `weights` instead of `arr`
2. **Leverage broadcasting**: Minimize explicit reshaping
3. **Prefer boolean indexing**: `arr[arr > 0]` instead of loops
4. **Use context managers**: For memory-mapped files
5. **Document shapes**: Comment array dimensions in complex code

---

## 🧩 Advanced Topics

### Structured Arrays
```python
dtype = np.dtype([('name', 'U10'), ('age', int)])
people = np.array([('Alice', 30), ('Bob', 25)], dtype=dtype)
```

### Universal Functions
```python
# Create custom ufunc
add = np.frompyfunc(lambda x,y: x+y, 2, 1)
```

### C API Integration
```c
#include <numpy/ndarrayobject.h>
static PyObject *myfunc(PyObject *self, PyObject *args) {
    PyArrayObject *arr;
    if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &arr)) return NULL;
    // ... C code ...
}
```

---

## 🌟 Learning Roadmap

1. **Basics**: Arrays, slicing, reshaping
2. **Operations**: Broadcasting, vectorization
3. **Math**: Linear algebra, statistics
4. **Optimization**: Memory management, profiling
5. **Advanced**: Structured arrays, C extensions
6. **Applications**: Image processing, simulations

---

## 🌐 Community & Resources

- [Official Documentation](https://numpy.org/doc/)
- [NumPy GitHub](https://github.com/numpy/numpy)
- [SciPy Lectures](http://scipy-lectures.org/intro/numpy/index.html)
- [NumPy for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)

---

## 📝 Conclusion

NumPy is more than just a Python library—it's a fundamental paradigm shift in how we think about data manipulation. By mastering its array operations, broadcasting rules, and memory model, you gain superpowers in numerical computing that extend far beyond Python itself. Whether you're processing satellite imagery, simulating quantum systems, or building machine learning models, NumPy provides the foundation that makes complex computations elegant and efficient.


##---

# 🐼 Pandas: The Ultimate Guide to Mastering Data Manipulation in Python
![Pandas Logo](https://pandas.pydata.org/static/img/pandas.svg)
![PyPI version](https://img.shields.io/pypi/v/pandas.svg)
![Python versions](https://img.shields.io/pypi/pyversions/pandas.svg)
![License](https://img.shields.io/badge/License-BSD_3_Clause-blue.svg)
![Downloads](https://img.shields.io/pypi/dm/pandas.svg)

---

## 📖 Table of Contents
1. [Theoretical Foundations](#theoretical-foundations)
2. [Installation Guide](#installation-guide)
3. [Core Concepts](#core-concepts)
4. [Data Structures](#data-structures)
5. [Data I/O](#data-io)
6. [Data Manipulation](#data-manipulation)
7. [Missing Data](#missing-data)
8. [Grouping and Aggregation](#grouping-and-aggregation)
9. [Merging and Joining](#merging-and-joining)
10. [Time Series](#time-series)
11. [Visualization](#visualization)
12. [Performance Optimization](#performance-optimization)
13. [Best Practices](#best-practices)
14. [Advanced Topics](#advanced-topics)
15. [Learning Roadmap](#learning-roadmap)

---

## 🧠 Theoretical Foundations

### What is Pandas?
Pandas is an open-source library providing high-performance, easy-to-use data structures and data analysis tools for Python. It's built on top of NumPy and designed to handle structured data efficiently.

### Key Features
- **Data Structures**: Series (1D) and DataFrame (2D)
- **Data Alignment**: Automatic alignment of data in computations
- **Missing Data Handling**: Sophisticated methods for handling missing values
- **Time Series Functionality**: Robust tools for time series analysis
- **Data I/O**: Read/write capabilities for various formats (CSV, Excel, SQL, etc.)
- **Vectorized Operations**: Fast element-wise operations
- **Grouping and Aggregation**: Powerful group-by functionality

### Why Pandas?
- **Productivity**: High-level abstractions for common data tasks
- **Flexibility**: Works with messy, irregular, and heterogeneous data
- **Performance**: Optimized C/Cython code under the hood
- **Integration**: Seamless with other scientific Python libraries
- **Community**: Large and active community supporting development

---

## 🛠️ Installation Guide

```bash
# Using pip
pip install pandas

# Using conda
conda install pandas
```

---

## 🧮 Core Concepts

### Creating Data Structures
```python
import pandas as pd

# Series (1D)
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# DataFrame (2D)
data = {'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']}
df = pd.DataFrame(data)
```

### Basic Operations
```python
# Viewing data
df.head()      # First 5 rows
df.tail()      # Last 5 rows
df.describe()  # Summary statistics
df.info()      # Data types and memory usage

# Indexing
df['Column1']  # Select column
df.iloc[0]     # Select row by position
df.loc[0]      # Select row by label
```

### Data Types
```python
df.dtypes      # Data types of columns
df.astype()    # Convert data types
df.infer_objects()  # Infer better data types
```

---

## 📊 Data I/O

### Reading Data
```python
# CSV
df = pd.read_csv('data.csv')

# Excel
df = pd.read_excel('data.xlsx')

# JSON
df = pd.read_json('data.json')

# SQL
df = pd.read_sql('SELECT * FROM table', connection)
```

### Writing Data
```python
# CSV
df.to_csv('data.csv', index=False)

# Excel
df.to_excel('data.xlsx', index=False)

# JSON
df.to_json('data.json')

# SQL
df.to_sql('table', connection, if_exists='replace')
```

---

## 🔄 Data Manipulation

### Filtering
```python
df[df['Column1'] > 2]  # Filter rows
df.query('Column1 > 2')  # Alternative syntax
```

### Sorting
```python
df.sort_values('Column1', ascending=False)
df.sort_index()  # Sort by index
```

### Adding/Removing
```python
# Add column
df['NewColumn'] = df['Column1'] * 2

# Remove column
df.drop('Column1', axis=1, inplace=True)

# Add row
df = df.append({'Column2': 'D'}, ignore_index=True)

# Remove row
df.drop(0, axis=0)
```

### Applying Functions
```python
df['Column1'].apply(lambda x: x**2)
df.applymap(lambda x: x*2)  # Element-wise
```

---

## 🧩 Missing Data

### Handling Missing Values
```python
df.isnull()    # Identify missing values
df.notnull()   # Identify non-missing values
df.dropna()    # Remove missing values
df.fillna(0)   # Fill missing values
df.interpolate()  # Interpolate missing values
```

### Replacing Values
```python
df.replace({1: 'A', 2: 'B'})
df.replace('[A-Za-z]', '', regex=True)
```

---

## 📚 Grouping and Aggregation

### Grouping
```python
grouped = df.groupby('Column2')
grouped.mean()    # Mean of groups
grouped.sum()     # Sum of groups
grouped.describe()  # Descriptive statistics
```

### Aggregation
```python
df.aggregate({'Column1': ['mean', 'std'], 'Column2': 'count'})
df.agg(['mean', 'std'])  # Apply multiple functions
```

---

## 🔄 Merging and Joining

### Concatenation
```python
pd.concat([df1, df2], axis=0)  # Vertical concatenation
pd.concat([df1, df2], axis=1)  # Horizontal concatenation
```

### Merging
```python
pd.merge(df1, df2, on='key')  # Inner join by default
pd.merge(df1, df2, left_on='key1', right_on='key2', how='left')
```

### Joining
```python
df1.join(df2, on='key')  # Join on index
```

---

## 🕒 Time Series

### Creating Date Ranges
```python
pd.date_range(start='2023-01-01', end='2023-01-31')
pd.date_range(periods=30, freq='D')
```

### Resampling
```python
df.resample('W').mean()  # Weekly resampling
```

### Time Zone Handling
```python
df.tz_localize('UTC')  # Localize to UTC
df.tz_convert('US/Eastern')  # Convert time zone
```

---

## 🎨 Visualization

### Basic Plots
```python
df.plot(kind='line', x='Column1', y='Column2')
df.hist()  # Histogram
df.boxplot()  # Box plot
```

### Customization
```python
ax = df.plot(x='Column1', y='Column2', title='My Plot')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
```

---

## ⚡ Performance Optimization

1. **Use Vectorized Operations**: Avoid apply() when possible
2. **Leverage Boolean Indexing**: Faster than filtering with apply()
3. **Chunksize Processing**: For large files
4. **Categorical Data Types**: For memory efficiency
5. **Parallel Processing**: Use swifter or dask for parallel operations
6. **Memory Optimization**: Use appropriate data types
7. **Avoid Copying Data**: Use inplace=True when possible

---

## 📜 Best Practices

1. **Meaningful Variable Names**: Use descriptive names
2. **Chain Operations**: When appropriate for readability
3. **Use Context Managers**: For file operations
4. **Document Your Code**: Especially for complex transformations
5. **Profile Memory Usage**: Use memory_profiler for large datasets
6. **Version Your Data**: Especially when processing large datasets
7. **Test Your Transformations**: Especially for ETL pipelines

---

## 🧩 Advanced Topics

### Window Functions
```python
df.rolling(window=7).mean()  # Rolling mean
df.expanding(min_periods=1).mean()  # Expanding mean
df.ewm(alpha=0.5).mean()  # Exponential moving average
```

### Pivot Tables
```python
pd.pivot_table(df, values='Column1', index=['Column2'], columns=['Column3'])
```

### Categorical Data
```python
df['Column'] = df['Column'].astype('category')
```

### Extension Arrays
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'Column': pd.Categorical(['a', 'b', 'c'])})
```

### Custom Accessors
```python
@pd.api.extensions.register_dataframe_accessor("custom")
class CustomAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def custom_method(self):
        # Custom functionality
        pass
```

---

## 🌟 Learning Roadmap

1. **Basics**: Data structures, indexing, basic operations
2. **Data I/O**: Reading/writing various file formats
3. **Data Manipulation**: Filtering, sorting, transforming
4. **Missing Data**: Handling missing values
5. **Grouping and Aggregation**: Group-by operations
6. **Merging and Joining**: Combining datasets
7. **Time Series**: Working with temporal data
8. **Visualization**: Creating meaningful plots
9. **Performance Optimization**: Efficient data processing
10. **Advanced Topics**: Window functions, categorical data, extension arrays

---

## 🌐 Community & Resources

- [Official Documentation](https://pandas.pydata.org/docs/)
- [Pandas GitHub](https://github.com/pandas-dev/pandas)
- [Pandas Cookbook](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html)
- [Real Python Pandas Tutorials](https://realpython.com/tutorials/pandas/)
- [Kaggle Pandas Micro-Course](https://www.kaggle.com/learn/pandas)

---

## 📝 Conclusion

Pandas is an essential tool for anyone working with data in Python. Its powerful data structures and extensive functionality make it the go-to library for data manipulation and analysis. By mastering pandas, you gain the ability to efficiently process and transform data, enabling you to focus on the insights rather than the mechanics of data handling.

## 🙏 Credits

- NumPy Development Team - [@numpy](https://github.com/numpy/numpy)
- Scientific Python Community
- Icon credits: [Shields.io](https://shields.io), [Twemoji](https://twemoji.twitter.com)
- - Pandas Development Team - [@pandas-dev](https://github.com/pandas-dev/pandas)
- Scientific Python Community
- Icon credits: [Shields.io](https://shields.io), [Twemoji](https://twemoji.twitter.com)
