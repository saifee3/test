# Data Analysis and Visualization with Python [Numpy, Pandas, Matplotlib, Seaborn]

![Python Data Science Stack](https://raw.githubusercontent.com/matplotlib/matplotlib.github.io/master/images/python-data-stack.png)

Welcome to the comprehensive guide for mastering the Data Analysis and Visualization with Python. This repository contains carefully curated Jupyter notebooks designed to take you from the fundamentals of numerical computing to creating publication-quality visualizations. Whether you're a beginner looking to break into data science or an experienced professional seeking to deepen your expertise, this resource will serve as your companion on the journey to Python data science mastery.

## The Data Analysis and Visualization with Python

In today's data-driven world, the ability to work with and understand data has become a crucial skill across various domains. Python has emerged as the premier language for data science, offering a powerful combination of flexibility, readability, and a rich ecosystem of libraries specifically designed for data manipulation, analysis, and visualization.

At the core of this ecosystem are four fundamental libraries:
- **NumPy**: The foundation of numerical computing in Python, providing support for large, multi-dimensional arrays and matrices
- **Pandas**: Built on NumPy, it offers data structures and operations for manipulating numerical tables and time series
- **Matplotlib**: The foundational plotting library for Python, enabling the creation of static, animated, and interactive visualizations
- **Seaborn**: Built on top of Matplotlib, it provides a high-level interface for drawing attractive statistical graphics

These libraries work together seamlessly, forming a powerful stack that allows data scientists to efficiently process, analyze, and visualize data.

## Repository Contents

This repository contains four comprehensive Jupyter notebooks, each dedicated to one of the core Python data science libraries:

### 01-Numpy.ipynb
### 02-Pandas.ipynb
### 03-Matplotlib.ipynb
### 04-Seaborn.ipynb

## Learning Pathway
Each notebook contains:
- Theoretical foundations
- Practical examples
- Code snippets
- Visualizations
- Exercises for reinforcement

## How to Use This Repository

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or Jupyter Lab
- Basic understanding of Python programming

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/python-data-science-mastery.git

# Install required packages
pip install numpy pandas matplotlib seaborn jupyter
```

### Running the Notebooks
```bash
# Navigate to the repository directory
cd python-data-science-mastery

# Start Jupyter Notebook
jupyter notebook
```

## ----- BEGINNING OF NUMPY Guide -----

# 🚀 NumPy: The Ultimate Guide
<img src="https://numpy.org/images/logo.svg" alt="Custom Icon" width="250" height="250">

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

## ----- END OF NUMPY Guide -----
## ----- BEGINNING OF PANDAS Guide -----

# 🐼 Pandas: The Ultimate Guide
<img src="https://pandas.pydata.org/static/img/pandas.svg" alt="Custom Icon" width="250" height="250">

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

## ----- END OF PANDAS Guide -----
## ----- BEGINNING OF MATPLOTLIB Guide -----

# 📊 Matplotlib: The Ultimate Guide
<img src="https://matplotlib.org/stable/_static/logo2_hex.png" alt="Custom Icon" width="250" height="250">

![PyPI version](https://img.shields.io/pypi/v/matplotlib.svg)
![Python versions](https://img.shields.io/pypi/pyversions/matplotlib.svg)
![License](https://img.shields.io/badge/License-BSD_3_Clause-blue.svg)
![Downloads](https://img.shields.io/pypi/dm/matplotlib.svg)

---

## 📖 Table of Contents
1. [Theoretical Foundations](#theoretical-foundations)
2. [Installation Guide](#installation-guide)
3. [Core Concepts](#core-concepts)
4. [Basic Plotting](#basic-plotting)
5. [Customization](#customization)
6. [Advanced Plotting](#advanced-plotting)
7. [3D Plotting](#3d-plotting)
8. [Animation](#animation)
9. [Integration with Other Libraries](#integration-with-other-libraries)
10. [Performance Optimization](#performance-optimization)
11. [Best Practices](#best-practices)
12. [Advanced Topics](#advanced-topics)
13. [Learning Roadmap](#learning-roadmap)

---

## 🧠 Theoretical Foundations

### What is Matplotlib?
Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It is the foundation upon which many other Python visualization libraries are built.

### Key Features
- **Versatile Plotting**: Supports various plot types including line plots, scatter plots, bar charts, histograms, heatmaps, etc.
- **Customization**: Extensive control over every element in a figure
- **Publication Quality**: Produces high-quality figures in various formats
- **Cross-Platform**: Works across different operating systems
- **Integration**: Seamlessly integrates with NumPy, Pandas, and Jupyter notebooks

### Why Matplotlib?
- **Flexibility**: Allows fine-grained control over visual elements
- **Community**: Large and active community support
- **Documentation**: Comprehensive official documentation
- **Ecosystem**: Foundation for other visualization libraries
- **Performance**: Optimized for handling large datasets

---

## 🛠️ Installation Guide

```bash
# Using pip
pip install matplotlib

# Using conda
conda install matplotlib

# For the latest development version from source
git clone https://github.com/matplotlib/matplotlib.git
cd matplotlib
pip install -e .
```

---

## 🧮 Core Concepts

### Basic Components
- **Figure**: The top-level container for all plot elements
- **Axes**: Individual plots (can be multiple in a figure)
- **Axis**: The actual x/y axis
- **Artist**: All visible elements (lines, text, etc.)

### Creating a Basic Plot
```python
import matplotlib.pyplot as plt

# Simple line plot
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Plot')
plt.show()
```

---

## 📊 Basic Plotting

### Line Plots
```python
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro-')
plt.axis([0, 5, 0, 20])
```

### Scatter Plots
```python
plt.scatter([1, 2, 3, 4], [1, 4, 9, 16], c='red', marker='o')
```

### Bar Charts
```python
plt.bar(['A', 'B', 'C', 'D'], [10, 20, 15, 25])
```

### Histograms
```python
plt.hist([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], bins=4)
```

---

## ✨ Customization

### Styling
```python
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'g--', linewidth=2)
plt.rc('lines', linewidth=2, linestyle='--')
```

### Annotations
```python
plt.annotate('Maximum', xy=(4, 16), xytext=(3, 12),
             arrowprops=dict(facecolor='black', shrink=0.05))
```

### Subplots
```python
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 1].scatter(x, y)
axs[1, 0].bar(x, y)
axs[1, 1].hist(y)
```

---

## 🚀 Advanced Plotting

### Heatmaps
```python
import numpy as np

data = np.random.rand(10, 10)
plt.imshow(data, cmap='viridis')
plt.colorbar()
```

### Contour Plots
```python
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.cos(Y)
plt.contourf(X, Y, Z)
```

### Statistical Charts
```python
plt.boxplot(data)
plt.violinplot(data)
```

---

## 🌌 3D Plotting

### 3D Line Plot
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([1, 2, 3], [4, 5, 6], [7, 8, 9])
```

### 3D Surface Plot
```python
ax.plot_surface(X, Y, Z, cmap='viridis')
```

---

## 🎥 Animation

### Creating Animations
```python
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
ani.save('animation.gif', writer='imagemagick')
```

---

## 🤝 Integration with Other Libraries

### With Pandas
```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df.plot(kind='bar')
```

### With NumPy
```python
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
```

---

## ⚡ Performance Optimization

1. **Vectorize Operations**: Use NumPy arrays for data manipulation
2. **Limit Redraws**: Use `plt.ion()` for interactive mode when developing
3. **Simplify Plots**: Reduce the number of elements when possible
4. **Use Rasterization**: For large datasets, rasterize plots
5. **Batch Drawing**: Combine multiple drawing operations

---

## 📜 Best Practices

1. **Consistent Style**: Use style sheets for uniform appearance
2. **Meaningful Labels**: Always label axes and add titles
3. **Appropriate Scales**: Choose the right scale for your data
4. **Color Blind Friendly**: Use color palettes accessible to all
5. **Save Vector Formats**: Use PDF or SVG for publication-quality figures

---

## 🧩 Advanced Topics

### Custom Backends
```python
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
```

### Transformations
```python
from matplotlib import transforms

t = transforms.Affine2D().rotate_deg(30) + ax.transData
text = ax.text(0.5, 0.5, "Text", transform=t)
```

### Custom Projections
```python
from matplotlib.projections import register_projection

class CustomProjection(Axes):
    name = 'custom'
    # ... custom implementation ...
    
register_projection(CustomProjection)
```

---

## 🌟 Learning Roadmap

1. **Basics**: Creating simple plots, understanding figure structure
2. **Customization**: Styling plots, adding annotations
3. **Advanced Plots**: Heatmaps, contour plots, statistical charts
4. **3D Visualization**: Creating and customizing 3D plots
5. **Animation**: Creating and exporting animations
6. **Integration**: Working with Pandas, NumPy, and Jupyter
7. **Performance**: Optimizing plotting for large datasets
8. **Advanced Features**: Custom backends, transformations, projections

---

## 🌐 Community & Resources

- [Official Documentation](https://matplotlib.org/stable/contents.html)
- [Matplotlib GitHub](https://github.com/matplotlib/matplotlib)
- [Matplotlib Examples](https://matplotlib.org/stable/gallery/index.html)
- [Real Python Matplotlib Tutorials](https://realpython.com/tutorials/matplotlib/)
- [Matplotlib Cookbook](https://matplotlib.org/stable/users/explain.html)

---

## 📝 Conclusion

Matplotlib is the cornerstone of data visualization in Python. Its versatility, customization options, and integration capabilities make it an essential tool for anyone working with data. By mastering Matplotlib, you gain the ability to create insightful, publication-quality visualizations that effectively communicate complex information.

## ----- END OF MATPLOTLIB Guide -----
## ----- BEGINNING OF SEABORN Guide -----

# 🎨 Seaborn: The Ultimate Guide
<img src="https://seaborn.pydata.org/_static/logo-large.png" alt="Custom Icon" width="500" height="250">

![PyPI version](https://img.shields.io/pypi/v/seaborn.svg)
![Python versions](https://img.shields.io/pypi/pyversions/seaborn.svg)
![License](https://img.shields.io/badge/License-BSD_3_Clause-blue.svg)
![Downloads](https://img.shields.io/pypi/dm/seaborn.svg)

---

## 📖 Table of Contents
1. [Theoretical Foundations](#theoretical-foundations)
2. [Installation Guide](#installation-guide)
3. [Core Concepts](#core-concepts)
4. [Basic Plots](#basic-plots)
5. [Customization](#customization)
6. [Advanced Plots](#advanced-plots)
7. [Integration with Other Libraries](#integration-with-other-libraries)
8. [Performance Optimization](#performance-optimization)
9. [Best Practices](#best-practices)
10. [Advanced Topics](#advanced-topics)
11. [Learning Roadmap](#learning-roadmap)

---

## 🧠 Theoretical Foundations

### What is Seaborn?
Seaborn is a Python data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. Seaborn integrates closely with Pandas data structures and is particularly powerful for visualizing structured data.

### Key Features
- **Statistical Graphics**: Specialized in creating informative statistical plots
- **High-Level Interface**: Simplifies complex visualization tasks
- **Aesthetic Customization**: Built-in themes and color palettes
- **Integration**: Works seamlessly with Pandas and Matplotlib
- **Advanced Statistical Plots**: Regression plots, distribution plots, and more

### Why Seaborn?
- **Simplifies Complex Visualizations**: Reduces code complexity for sophisticated plots
- **Built-in Themes**: Provides attractive default styles
- **Data-Centric API**: Designed to work directly with DataFrames
- **Statistical Focus**: Built-in support for common statistical visualizations
- **Extensible**: Can be customized with Matplotlib when needed

---

## 🛠️ Installation Guide

```bash
# Using pip
pip install seaborn

# To include optional statistical dependencies
pip install seaborn[stats]

# Using conda
conda install seaborn -c conda-forge
```

---

## 🧮 Core Concepts

### Basic Components
- **Figure**: The canvas where plots are drawn
- **Axes**: Individual plots within a figure
- **Data**: Typically provided as Pandas DataFrames
- **Aesthetics**: Mapping of data variables to visual properties

### Creating a Basic Plot
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load example dataset
tips = sns.load_dataset("tips")

# Create a basic plot
sns.relplot(data=tips, x="total_bill", y="tip", col="time")
plt.show()
```

---

## 📊 Basic Plots

### Distribution Plots
```python
# Histogram with kernel density estimate
sns.histplot(data=tips, x="total_bill", kde=True)

# Box plot
sns.boxplot(data=tips, x="day", y="total_bill")

# Violin plot
sns.violinplot(data=tips, x="day", y="total_bill")
```

### Categorical Plots
```python
# Bar plot
sns.barplot(data=tips, x="day", y="total_bill")

# Point plot
sns.pointplot(data=tips, x="day", y="total_bill", hue="sex")
```

### Regression Plots
```python
# Simple linear regression
sns.regplot(data=tips, x="total_bill", y="tip")

# Logistic regression
sns.regplot(data=tips, x="total_bill", y="smoker", logistic=True)
```

---

## ✨ Customization

### Styling
```python
# Set style
sns.set_style("whitegrid")

# Set color palette
sns.set_palette("viridis")

# Create custom color palette
palette = sns.color_palette("coolwarm", 5)
```

### Annotations and Labels
```python
# Add annotations
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.annotate("High tip", xy=(40, 8), xytext=(20, 10),
             arrowprops=dict(facecolor='black', shrink=0.05))
```

### Faceting
```python
# Create facet grid
g = sns.FacetGrid(tips, col="time", row="smoker")
g.map(sns.scatterplot, "total_bill", "tip")
```

---

## 🚀 Advanced Plots

### Matrix Plots
```python
# Heatmap
flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
sns.heatmap(flights, annot=True, fmt="d")

# Clustermap
sns.clustermap(flights, annot=True, fmt="d")
```

### Multi-Panel Plots
```python
# Pair plot
penguins = sns.load_dataset("penguins")
sns.pairplot(penguins, hue="species")

# Joint plot
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")
```

---

## 🤝 Integration with Other Libraries

### With Pandas
```python
# Use Pandas DataFrame directly
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
sns.lineplot(data=df, x="A", y="B")
```

### With NumPy
```python
# Generate data with NumPy and plot with Seaborn
x = np.linspace(0, 10, 100)
y = np.sin(x)
sns.lineplot(x=x, y=y)
```

---

## ⚡ Performance Optimization

1. **Vectorized Operations**: Use built-in functions that work with arrays
2. **Data Sampling**: Work with samples of large datasets when exploring
3. **Simplify Plots**: Reduce the number of visual elements when possible
4. **Batch Processing**: Process and plot data in batches for very large datasets
5. **Use Appropriate Plot Types**: Choose plot types that efficiently represent your data

---

## 📜 Best Practices

1. **Start Simple**: Begin with basic plots and add complexity as needed
2. **Use Appropriate Scales**: Choose the right scale for your data
3. **Label Clearly**: Always label axes and add titles
4. **Choose Informative Visualizations**: Select plot types that best reveal patterns in your data
5. **Document Your Visualizations**: Comment on what each plot is intended to show

---

## 🧩 Advanced Topics

### Custom Estimators
```python
# Custom estimator in line plot
sns.lineplot(data=tips, x="total_bill", y="tip", estimator=np.median)
```

### Custom Plot Types
```python
# Create a custom plot using object interface
p = sns.Plot(data=tips, x="total_bill", y="tip", color="sex")
p.add(sns.Line())
p.add(sns.Point())
p.render()
```

### Statistical Transformations
```python
# Use statistical transformations
sns.histplot(data=tips, x="total_bill", stat="density")
```

---

## 🌟 Learning Roadmap

1. **Basics**: Creating simple plots, understanding data structures
2. **Customization**: Styling plots, adding annotations
3. **Advanced Plots**: Heatmaps, clustermaps, pair plots
4. **Integration**: Working with Pandas, NumPy, and Jupyter
5. **Performance**: Optimizing visualization for large datasets
6. **Advanced Features**: Custom estimators, statistical transformations

---

## 🌐 Community & Resources

- [Official Documentation](https://seaborn.pydata.org/)
- [Seaborn GitHub](https://github.com/mwaskom/seaborn)
- [Example Gallery](https://seaborn.pydata.org/examples/index.html)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/seaborn)

---

## 📝 Conclusion

Seaborn is a powerful library that simplifies the creation of informative and attractive statistical visualizations in Python. By building on Matplotlib and integrating closely with Pandas, it provides a high-level interface for complex visualizations while maintaining flexibility for customization.


---
---

## 🙏 Credits

- NumPy Development Team - [@numpy](https://github.com/numpy/numpy)
- Scientific Python Community
- Icon credits: [Shields.io](https://shields.io), [Twemoji](https://twemoji.twitter.com)
- Pandas Development Team - [@pandas-dev](https://github.com/pandas-dev/pandas)
- Scientific Python Community
- Icon credits: [Shields.io](https://shields.io), [Twemoji](https://twemoji.twitter.com)
- Matplotlib Development Team - [@matplotlib](https://github.com/matplotlib/matplotlib)
- Scientific Python Community
- Icon credits: [Shields.io](https://shields.io), [Twemoji](https://twemoji.twitter.com)
- Seaborn Development Team - [@mwaskom](https://github.com/mwaskom/seaborn)
- Scientific Python Community
- Icon credits: [Shields.io](https://shields.io), [Twemoji](https://twemoji.twitter.com)
