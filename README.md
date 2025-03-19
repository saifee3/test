# 📊 COVID-19 Exploratory Data Analysis (EDA)
<img src="https://pandas.pydata.org/static/img/pandas.svg" alt="Custom Icon" width="250" height="250">

## Introduction to EDA
Exploratory Data Analysis (EDA) is an approach to analyzing datasets to summarize their main characteristics, often using visual methods. It was promoted by statistician John Tukey in the 1970s as a way to understand data before making assumptions or applying more formal statistical models.

EDA helps data scientists:
- Understand data structure and variables
- Identify patterns, trends, and relationships
- Detect outliers and anomalies
- Test hypotheses
- Generate questions for further analysis

## Why is EDA Important?
1. **Builds intuition about the data**: EDA helps analysts develop a "feel" for the dataset
2. **Identifies data quality issues**: Missing values, inconsistencies, and errors become apparent
3. **Guides modeling decisions**: Understanding data distributions helps select appropriate models
4. **Generates hypotheses**: Patterns observed in EDA can inspire new questions
5. **Communicates insights**: Visualizations from EDA effectively share findings with stakeholders

## Step-by-Step EDA Process
### Step 1: Define Objectives and Questions
Before diving into data, clarify:
- What are the business objectives?
- What specific questions do you want to answer?
- What hypotheses do you want to test?

### Step 2: Data Collection
Gather relevant datasets from various sources:
- CSV files
- Databases
- APIs
- Web scraping
- Cloud storage

### Step 3: Data Cleaning
Address data quality issues:
- Handle missing values (imputation, removal, or flagging)
- Remove duplicates
- Correct data types
- Standardize formats
- Filter irrelevant data

### Step 4: Understand Data Structure
Examine:
- Dimensions (rows and columns)
- Variable types (numerical, categorical, temporal)
- Basic statistics (mean, median, min, max)

### Step 5: Univariate Analysis
Analyze individual variables:
- Numerical variables: distributions, central tendency, spread
- Categorical variables: frequency counts, mode

Common visualizations:
- Histograms
- Box plots
- Bar charts
- Pie charts

### Step 6: Bivariate Analysis
Explore relationships between two variables:
- Numerical vs numerical: correlation, scatter plots
- Numerical vs categorical: group comparisons, box plots
- Categorical vs categorical: cross-tabulation, heatmaps

### Step 7: Multivariate Analysis
Investigate relationships among multiple variables:
- Pairwise correlations
- 3D scatter plots
- Heatmaps
- Faceted plots

### Step 8: Identify Patterns and Anomalies
Look for:
- Seasonal patterns
- Trends
- Outliers
- Unexpected values

## Common EDA Techniques
| Technique          | Description                                  | When to Use                          |
|--------------------|----------------------------------------------|--------------------------------------|
| Summary Statistics | Mean, median, mode, standard deviation      | Initial understanding of numerical data |
| Histograms         | Distribution of a single numerical variable | Check data distribution              |
| Box Plots          | Distribution and outliers of numerical data | Identify outliers and compare groups |
| Scatter Plots      | Relationship between two numerical variables| Check correlations                   |
| Bar Charts         | Compare categories or groups                | Categorical data comparison         |
| Heatmaps           | Visualize correlations or densities         | Show relationships in large datasets |
| Time Series Plots  | Trends over time                            | Temporal data analysis               |

## Tools for EDA

### Programming Languages
- **Python**: Most popular language for EDA with powerful libraries
- **R**: Strong statistical analysis capabilities
- **SQL**: For data extraction and basic analysis

### Python Libraries
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Scipy**: Statistical functions
- **Statsmodels**: Statistical modeling

### IDEs/Jupyter
- **Jupyter Notebook**: Interactive computing environment
- **Google Colab**: Cloud-based Jupyter notebook
- **VS Code**: Code editor with Python support
- **PyCharm**: Python IDE

## Best Practices

1. **Document everything**: Keep track of your findings and thought process
2. **Visualize early and often**: Graphics reveal patterns numbers might miss
3. **Check data quality**: Address missing values and outliers before analysis
4. **Ask questions**: Let the data guide your exploration
5. **Iterate**: EDA is a cyclical process
6. **Communicate insights**: Create clear visualizations for stakeholders

## Example Workflow

Let's walk through a complete EDA workflow using COVID-19 data:

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('covid_data.csv')

# Initial inspection
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Missing values check
print("\nMissing values:")
print(df.isnull().sum())

# Data cleaning
# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Fill missing values in 'recovered' with 0
df['recovered'].fillna(0, inplace=True)

# Calculate new features
df['active_cases'] = df['confirmed_cases'] - df['deaths'] - df['recovered']
df['death_rate'] = (df['deaths'] / df['confirmed_cases']) * 100

# Univariate analysis: Death rate distribution
plt.hist(df['death_rate'], bins=30)
plt.xlabel('Death Rate (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Death Rates Across Regions')
plt.show()

# Bivariate analysis: Confirmed cases vs deaths
plt.scatter(df['confirmed_cases'], df['deaths'])
plt.xlabel('Confirmed Cases')
plt.ylabel('Deaths')
plt.title('Relationship Between Confirmed Cases and Deaths')
plt.show()

# Multivariate analysis: Time series trends
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['confirmed_cases'], label='Confirmed')
plt.plot(df['date'], df['deaths'], label='Deaths')
plt.plot(df['date'], df['recovered'], label='Recovered')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('COVID-19 Trends Over Time')
plt.legend()
plt.show()

# Group analysis: Deaths by region
region_deaths = df.groupby('region')['deaths'].sum().sort_values(ascending=False)
plt.bar(region_deaths.index, region_deaths.values)
plt.xlabel('Region')
plt.ylabel('Total Deaths')
plt.title('Total Deaths by Region')
plt.xticks(rotation=45)
plt.show()

# Correlation matrix
corr_matrix = df[['confirmed_cases', 'deaths', 'recovered', 'active_cases']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of COVID-19 Metrics')
plt.show()
```
---
---

# 📊 COVID-19 Dataset Exploratory Data Analysis (EDA) Project

## 🌐 Project Overview
This EDA investigates COVID-19 progression across countries and provinces, analyzing relationships between confirmed cases, deaths, recoveries, and temporal trends. The analysis focuses on 187 unique countries from a global dataset spanning January 2020 onward.

---
## Dataset Description 📊

The project utilizes multiple datasets:
- **Aggregated Country Data**: Contains confirmed cases, deaths, recoveries, and active cases by country
- **Death Ratio Data**: Provides death ratios calculated as (Deaths / Confirmed Cases) * 100
- **Main COVID-19 Dataset**: Includes time-series data with provincial breakdowns

---

## 📂 Repository Structure
```bash
covid19_eda/
├── data/
│   ├── 01aggregated_country_data.csv    # Country-level aggregated metrics
│   ├── 02death_ratio_data.csv           # Calculated death ratios
│   └── main_covid19_dataset.csv         # Raw COVID-19 time-series data
├── notebooks/
│   └── covid19_eda.ipynb                # Main analysis notebook
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

---

## 🛠️ Technical Setup

### Requirements
- Python 3.8+
- Libraries:
  ```text
  pandas==1.3.5
  matplotlib==3.5.1
  jupyter==1.0.0
  ```

### Installation
```bash
git clone https://github.com/yourusername/covid19_eda.git
cd covid19_eda
pip install -r requirements.txt
jupyter notebook
```

## Exploratory Data Analysis 📈

The project focuses on several key analyses:

### Countries with respect to Death Ratio 🌎💀
```python
def plot_death_ratio_by_country(df, top_n=10):
    sorted_df = df.sort_values(by='Death Ratio (%)', ascending=False).head(top_n)
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_df['Country/Region'], sorted_df['Death Ratio (%)'])
    plt.xlabel('Country')
    plt.ylabel('Death Ratio (%)')
    plt.title(f'Top {top_n} Countries by Death Ratio')
    plt.xticks(rotation=45)
    plt.show()
```

### Specific Country's Province with respect to Death and Recovery 🏞️🚑
```python
def analyze_province_data(country_name, province_data):
    country_provinces = province_data[province_data['Country/Region'] == country_name]
    plt.figure(figsize=(12, 6))
    plt.bar(country_provinces['Province/State'], country_provinces['Deaths'], label='Deaths')
    plt.bar(country_provinces['Province/State'], country_provinces['Recovered'], label='Recovered', alpha=0.7)
    plt.xlabel('Province')
    plt.ylabel('Count')
    plt.title(f'{country_name}: Deaths and Recoveries by Province')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
```

### Trend between Confirmed Cases and Deaths 📉
```python
def plot_confirmed_vs_deaths(df, country_name):
    country_data = df[df['Country/Region'] == country_name]
    plt.figure(figsize=(12, 6))
    plt.scatter(country_data['Confirmed'], country_data['Deaths'])
    plt.xlabel('Confirmed Cases')
    plt.ylabel('Deaths')
    plt.title(f'{country_name}: Confirmed Cases vs Deaths')
    plt.show()
```

### Trend between Time and COVID-19 Metrics ⏳
```python
def plot_time_series(country_name, time_series_data):
    country_time_data = time_series_data[time_series_data['Country/Region'] == country_name]
    plt.figure(figsize=(12, 6))
    plt.plot(country_time_data['Date'], country_time_data['Confirmed'], label='Confirmed')
    plt.plot(country_time_data['Date'], country_time_data['Recovered'], label='Recovered')
    plt.plot(country_time_data['Date'], country_time_data['Deaths'], label='Deaths')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title(f'{country_name}: COVID-19 Trends Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
```

## How to Use 📖

1. Load the datasets using pandas
2. Clean and preprocess the data using functions from `src/data_processing.py`
3. Perform exploratory analysis using visualizations from `src/visualization.py`
4. Execute the Jupyter notebook cells sequentially for a guided analysis

## Results and Insights 📄

After completing the analysis, you'll gain insights into:
- Which countries had the highest death ratios
- How COVID-19 metrics varied across provinces within countries
- The relationship between confirmed cases and deaths
- Temporal patterns in COVID-19 spread and recovery

## Contributing 🤝

Contributions are welcome! Please fork the repository and create a pull request with your improvements.

## 📜 License
MIT License - See [LICENSE](LICENSE) for details.

---

## Credits
- Data Source: Johns Hopkins University CSSE COVID-19 Dataset
- Last Updated: February 2023
- Tools: Python, pandas, matplotlib, Jupyter Notebook
- Inspiration: Standard EDA practices and COVID-19 research papers


[![Open in GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/saifee3/covid19_eda) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/saifee3/covid19_eda/main)
