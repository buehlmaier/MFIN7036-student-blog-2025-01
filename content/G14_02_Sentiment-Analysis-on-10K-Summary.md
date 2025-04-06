---
Title: Sentiment Analysis on 10K Summary
Date: 2025-03-09 12:00
Category: Reflective Report
Tags: Group G14
---

By Group "G14"

# Decoding Financial Sentiment: A Deep Dive into 10-K Filings and Stock Performance

"In the world of business, the people who are most fearful of the future are always those who are facing the future with the most inadequate tools." - Warren Buffett

Can the sentiment hidden within a company's annual reports unlock the secrets to its future stock performance? This question drove us to explore the intersection of Natural Language Processing (NLP) and financial analysis, using 10-K filings as our canvas. In this blog post, we detail our journey—from data collection to sentiment analysis and regression modeling—and share the unexpected lessons we uncovered along the way.

## The Data Hunt

Our exploration began with a treasure trove of corporate filings stored in McDonald's Dropbox. Our target: 10-K filings, the annual reports that offer a detailed snapshot of a company's financial health. To identify these files, we turned to Python's `re` library, filtering filenames that contained "10-K".

The next step was to extract the Central Index Key (CIK), a unique identifier assigned by the SEC, from each filename. We designed a regular expression—`r'edgar_data_(\d+)_'`—to pinpoint the CIK within filenames like "edgar_data_789019_10-K_2022.pdf", yielding "789019".

Here's how we did it:

```python
import re

filename = "edgar_data_789019_10-K_2022.pdf"
match = re.search(r'edgar_data_(\d+)_', filename)
if match:
    cik = match.group(1)
    print(cik)  # Output: 789019
```

With a list of unique CIKs, we accessed the WRDS (Wharton Research Data Services) database to match them with PERMNOs (Permanent Identifiers). These PERMNOs allowed us to retrieve historical stock price data from the CRSP database, setting the stage for our performance analysis.

## Sentiment Engineering

To gauge the tone of the 10-K filings, we employed FinBERT, an NLP model tailored for financial text. FinBERT analyzed the documents and produced sentiment scores—positive, negative, and neutral—quantifying the emotional undercurrents in management's narratives. This process transformed dense financial prose into a set of numerical insights we could work with.

## The Performance Puzzle

For each company and filing year, we calculated two performance metrics:
- **Stock Returns**: The percentage change in stock price over the year.
- **Sharpe Ratios**: A risk-adjusted measure of return, calculated as the excess return over the risk-free rate divided by the standard deviation of returns.

These metrics were aligned with the corresponding 10-K filing years, ensuring our analysis stayed temporally consistent.

## Regression Reality

We then performed linear regression to test whether sentiment scores could predict stock returns and Sharpe ratios. The setup was simple: sentiment scores as the independent variable, and performance metrics as the dependent variables. Here's a conceptual snippet of how we might implement this in Python:

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data (replace with actual sentiment scores and returns)
sentiment_scores = [0.6, 0.4, 0.7, 0.3]  # Positive sentiment probabilities
stock_returns = [0.05, -0.02, 0.08, -0.01]  # Annual returns

X = np.array(sentiment_scores).reshape(-1, 1)
y = np.array(stock_returns)

model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)
print(f"R² for stock returns: {r_squared}")
```

The results were revealing:
- **Stock Returns**: R² = -0.0064
- **Sharpe Ratios**: R² = -0.0130

A negative R² means our model performed worse than a baseline that simply predicts the average of the dependent variable. In plain terms, sentiment scores failed to explain the variability in stock performance and even led to poorer predictions.

## Lessons from the Ledger

These findings highlight a critical truth: sentiment scores from 10-K filings alone are not a reliable crystal ball for stock performance. Several factors might explain this:
- **Market Efficiency**: Stock prices reflect a broad array of information—earnings, news, economic conditions—beyond what’s captured in annual filings.
- **Retrospective Focus**: 10-Ks look backward, while stock prices look forward, creating a temporal mismatch.
- **NLP Limitations**: Even advanced models like FinBERT may struggle to fully decode the nuanced, context-heavy language of financial reports.

## Conclusion

Our experiment underscores the limits of relying solely on sentiment analysis for financial forecasting. While the idea of mining 10-K filings for predictive signals is enticing, our negative R² values suggest that sentiment is just one piece of a much larger puzzle. Future efforts could enhance this approach by blending sentiment with financial metrics, market data, or more sophisticated NLP techniques. For now, we’re reminded that in the complex world of finance, no single tool holds all the answers—but each step forward sharpens our understanding.