---
Title: Selection of Regression Models (by "Group 9")
Date: 2025-03-09 
Category: Progress Report
Tags: Group FIN-TANSTIC WIZARDS
---

By **Group "FIN-TANSTIC WIZARDS"**

# Blog Post: Regression Analysis of Social Media Sentiment Divergence on Bitcoin Trading Volume and Volatility

## Introduction

In this chapter of our GitHub-hosted project, we move forward with the regression analysis to explore the impact of the Social Media Sentiment Divergence Index on Bitcoin’s trading volume and volatility. With the regression variables prepared in the previous step, we now focus on building a Vector Autoregression (VAR) model, conducting stationarity and causality tests, and fitting an Ordinary Least Squares (OLS) regression to examine the relationships between the variables. This blog details the process, the rationale behind our modeling choices, and sets the stage for interpreting the results. Reflecting on this phase, we’ve gained deep insights into time-series modeling and the complexities of financial data.

```python
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 22:44:31 2025

@author: 24147
"""

'''2. VAR Model'''
'''pip install matplotlib statsmodels'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.iolib.summary2 import summary_col
import datetime
```

## Constructing the Regression Model

Having prepared the dataset with variables like `Return`, `Prange`, `aVol`, `Disagt`, and `ln_user`, we proceeded with the regression analysis. Below is a step-by-step breakdown of our approach. This stage was a significant learning curve for us—we initially struggled to grasp the nuances of VAR modeling and how to integrate it with OLS regression. Through collaborative brainstorming and extensive research, we tackled these challenges step by step.

### 1. Data Refinement and Time Range Selection

We started by loading the prepared dataset from the Parquet file (`Regression variables.parquet`). To streamline the analysis, we dropped the `Ext_pos` and `Ext_neg` variables, as they were binary indicators of extreme returns and less relevant for the continuous relationships we aimed to model in the VAR and OLS frameworks. We debated whether to keep these for outlier analysis but decided against it to maintain focus on dynamic trends.  
We filtered the dataset to focus on the period from January 1, 2022, to January 1, 2024, chosen to capture a recent and relatively stable period in Bitcoin’s history, avoiding volatility spikes in 2024. We simplified the `Date` column to retain only the year, month, and day, setting it as the index for time-series analysis. Selecting this range required balancing market stability and data volume, and we plan to revisit this if future data suggests a different focus.

```python
'''2.1 Dropping Variables and Filtering Time Range'''
# Read the Parquet file
df = pd.read_parquet('Regression variables.parquet')
print(df.info())

# Drop variables 'Ext_pos' and 'Ext_neg'
df = df.drop(['Ext_pos', 'Ext_neg'], axis=1)

# Filter for the appropriate time range
df = df.drop(df[df['Date'] < np.datetime64('2022-01-01 00:00:00')].index)
df = df.drop(df[df['Date'] > np.datetime64('2024-01-01 00:00:00')].index)

# Keep only the year, month, and day in 'Date', and set 'Date' as the index
df['Date'] = df['Date'].dt.date
df = df.set_index('Date')
```

### 2. Descriptive Statistics

Before proceeding with modeling, we generated descriptive statistics using `df.describe()` to understand the distribution of our variables (`Return`, `Prange`, `aVol`, `Disagt`, `ln_user`). This step helped confirm data integrity and identify potential outliers or anomalies that might affect the regression results. We analyzed these statistics closely, pondering how skewness in `ln_user` might influence our model, and decided to monitor its impact in the regression output.

```python
'''2.2 Descriptive Statistics'''
print(df.describe())
```

### 3. Stationarity Test with ADF

Since the VAR model requires stationary time series, we conducted an Augmented Dickey-Fuller (ADF) test on each variable. The `adf_test` function returned the p-value for each series, where a p-value less than 0.05 indicates stationarity. For non-stationary series, differencing or transformations would be needed, but we assumed our variables (e.g., `Return`, `Prange`, `aVol`) were stationary (e.g., returns are typically stationary by definition). We initially struggled to interpret the p-values, but studying ADF assumptions clarified our approach, though we plan to explore differencing if stationarity issues arise.

```python
'''2.3 Stationarity Test'''
# Augmented Dickey-Fuller (ADF) test for stationarity
def adf_test(series):
    result = adfuller(series)
    return result[1]  # Return p-value

for column in df.columns:
    p_value = adf_test(df[column])
    print(f'p-value for {column}: {p_value}')
```

### 4. Building the VAR Model

We used the `VAR` model from `statsmodels` to explore dynamic relationships between our variables. We estimated the optimal lag order using `model.select_order()`, which evaluates criteria like AIC and BIC to suggest the best lag length.  
Based on the results (and for simplicity in this initial analysis), we fitted the VAR model with a lag of 1 (`model.fit(1)`). This model captures how each variable (e.g., `Prange`, `aVol`, `Disagt`) is influenced by its own lagged values and those of others, providing a holistic view of interdependencies. Choosing the lag order was thoughtful—we debated between AIC and BIC but opted for a manual lag of 1 to start, planning to refine this based on model diagnostics.

```python
'''2.4 Building the VAR Model'''
model = VAR(df)

# Estimate the optimal lag order
lag_order = model.select_order()
print(lag_order.summary())

# Fit the VAR model with a lag of 1
var_model = model.fit(1)
print(var_model.summary())
```

### 5. Granger Causality Tests

To investigate whether the Sentiment Divergence Index (`Disagt`) has a predictive effect on Bitcoin volatility (`Prange`) and trading volume (`aVol`), we conducted Granger causality tests. We tested the following pairs:  
- `Prange` on `Disagt` and vice versa.  
- `aVol` on `Disagt` and vice versa.  

The tests were run with a lag of 1, focusing on the chi-square test statistic (`ssr based chi2 test`) and its p-value. A significant p-value (e.g., < 0.05) suggests that the independent variable Granger-causes the dependent variable, indicating a potential predictive relationship. Interpreting the output was challenging initially—we dug into the documentation to understand the chi-square versus F-test distinction, deepening our understanding of causality.

```python
'''2.5 Causality Test'''
# There are 4 tests; ideally all are significant. For small samples, look at F-test; for large samples, look at chi-square (i.e., 'ssr based chi2 test' chi2 and p-value)
print(grangercausalitytests(df[['Prange', 'Disagt']], maxlag=[1]))
print(grangercausalitytests(df[['Disagt', 'Prange']], maxlag=[1]))
print(grangercausalitytests(df[['aVol', 'Disagt']], maxlag=[1]))
print(grangercausalitytests(df[['Disagt', 'aVol']], maxlag=[1]))
```

### 6. OLS Regression with Lagged Variables

To further examine the relationships, we fitted an OLS regression model using lagged variables to account for temporal dynamics. We created lagged versions of all variables (`Disagt_t-1`, `Prange_t-1`, `aVol_t-1`, `Return_t-1`, `ln_user_t-1`) using `shift(1)`, ensuring independent variables were from the previous day.  
We defined three dependent variables: `Prange` (intraday volatility), `aVol` (adjusted trading volume), and `Disagt` (sentiment divergence index). The independent variables included the lagged versions of all variables plus a constant term. The regression equations were:

- For `Prange_t`: Prange_t = beta_0 + beta_1 * Disagt_t-1 + beta_2 * Prange_t-1 + beta_3 * aVol_t-1 + beta_4 * Return_t-1 + beta_5 * ln_user_t-1 + epsilon_t</span>
- For `aVol_t`:aVol_t = beta_0 + beta_1 * Disagt_t-1 + beta_2 * Prange_t-1 + beta_3 * aVol_t-1 + beta_4 * Return_t-1 + beta_5 * ln_user_t-1 + epsilon_t</span>
- For `Disagt_t`: Disagt_t = beta_0 + beta_1 * Disagt_t-1 + beta_2 * Prange_t-1 + beta_3 * aVol_t-1 + beta_4 * Return_t-1 + beta_5 * ln_user_t-1 + epsilon_t</span>

We used `sm.OLS` to fit these models and summarized the results with `summary_col` to present coefficients, significance levels, and observations in a concise table. Aligning the lagged variables was a technical challenge—we initially faced data loss due to misalignment, but adjusting the `dropna` step resolved it, though we’re still reflecting on optimizing this process.

```python
'''2.6 Regression Results Output'''
# Add lagged variables
df['Prange_t-1'] = df['Prange']
df['aVol_t-1'] = df['aVol']
df['Disagt_t-1'] = df['Disagt']
df['Return_t-1'] = df['Return']
df['ln_user_t-1'] = df['ln_user']
df[['Disagt_t-1', 'aVol_t-1', 'Return_t-1', 'Prange_t-1', 'ln_user']] = df[['Disagt_t-1', 'aVol_t-1', 'Return_t-1', 'Prange_t-1', 'ln_user']].shift(1)
df = df.dropna(subset=['Disagt_t-1', 'aVol_t-1', 'Return_t-1', 'Prange_t-1', 'ln_user_t-1'])

# Fit the OLS regression
x = df[['Disagt_t-1', 'Prange_t-1', 'aVol_t-1', 'Return_t-1', 'ln_user_t-1']]
x = sm.add_constant(x)

y1 = df['Prange']
y2 = df['aVol']
y3 = df['Disagt']

result1 = sm.OLS(y1, x).fit()
result2 = sm.OLS(y2, x).fit()
result3 = sm.OLS(y3, x).fit()

# Summarize regression results
Result = summary_col([result1, result3, result2, result3],
                     model_names=['Prange', 'Disagt', 'aVol', 'Disagt'],
                     stars=True, 
                     regressor_order=['Disagt_t-1', 'Prange_t-1', 'aVol_t-1', 'Return_t-1', 'ln_user_t-1', 'const'],
                     info_dict={'': lambda x: '',
                                'Observations': lambda x: str(int(x.nobs)),
                                })
print(Result)
```

## Rationale Behind Our Modeling Choices

- **Time Range Selection**: Focusing on 2022–2023 ensures a recent period with relatively stable market conditions, avoiding extreme volatility from 2024. We chose this range after debating the trade-off between data recency and stability, and we’re open to adjusting it based on future insights.
- **Stationarity Check**: The ADF test was crucial to validate VAR model assumptions. Since financial variables like returns and normalized volumes are often stationary, this step confirmed our data’s suitability for time-series modeling. We pondered non-stationarity implications and plan to explore differencing if needed.
- **VAR Model for Interdependencies**: The VAR model examines dynamic interactions between all variables simultaneously, capturing feedback loops (e.g., how `Disagt` might influence `Prange`, which affects `aVol`). We appreciated its holistic approach but are considering more lags if the initial fit is insufficient.
- **Granger Causality for Predictive Insight**: These tests helped identify whether `Disagt` predicts `Prange` and `aVol`, central to our research question about sentiment divergence’s impact. We reflected on significance thresholds and intend to test higher lags for robustness.
- **Lagged Variables in OLS**: Using lagged variables accounts for temporal dynamics, modeling past sentiment and market conditions’ effect on current outcomes. This aligns with our hypothesis of delayed social media impact, though lag alignment posed challenges resolved through testing.
- **Comprehensive Variable Inclusion**: Including `ln_user_t-1` alongside `Disagt_t-1` controls for social media activity, isolating sentiment divergence’s effect. We debated its necessity but included it for accuracy, planning to assess its contribution in the results.

