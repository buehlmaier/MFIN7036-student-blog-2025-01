---
Title: Disagreement index (by "Group 9")
Date: 2025-03-09 
Category: Progress Report
Tags: Group FIN-TANSTIC WIZARDS
---

By **Group "FIN-TANSTIC WIZARDS"**
<br>

# Understanding the Disagreement Index  

In today’s hyper-connected, data-driven world, gauging public opinion is more vital than ever—not just for social scientists or marketers, but for anyone seeking to understand how collective sentiment influences real-world outcomes, such as financial markets. The **Disagreement Index** offers a clever way to measure the dispersion of opinions among social media users. Inspired by Akarsu and Yilmaz’s (2024) research on how social media disagreement affects stock and Bitcoin markets, we’ve been exploring this index as a lens into human behavior. It’s derived from the standard deviation of a binary sentiment index calculated daily—but what does that mean, and why does it matter? Let’s break it down step-by-step and consider its broader implications.


#Defining Disagreement in Financial Markets 
Disagreement, in financial terms, refers to **heterogeneous beliefs among investors about an asset’s future value** (Hong & Stein, 2007). This divergence arises from two key sources:  
- **Asymmetric information**: Investors access different datasets or interpret the same information differently.  
- **Behavioral biases**: Cognitive differences (e.g., overconfidence, herding) lead to conflicting predictions.  

In social media contexts, disagreement reflects the distribution of positive/negative opinions among users. High disagreement indicates polarized views, while low disagreement signals consensus. This metric is critical because it predicts market volatility and trading volume: when investors disagree, they trade more actively, amplifying price swings (Antweiler & Frank, 2004; Akarsu & Yilmaz, 2024).


# Step-by-Step Calculation  

### 1. Sentiment Analysis Tools   
We start by analyzing comment sentiment with two rule-based tools: VADER and TextBlob. Both assign scores from -1 (negative) to +1 (positive), with 0 as neutral. VADER excels at capturing social media nuances—like slang or emojis—while TextBlob provides a straightforward polarity measure. Using both aims for robustness, but it prompts a question: is this dual approach necessary, or are we overcomplicating things?  

```python  
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  
from textblob import TextBlob  
```

**Why use two tools?**  
Combining models reduces bias from algorithm-specific limitations. For example, VADER may overreact to extreme language, while TextBlob might miss sarcasm. Averaging scores balances these quirks (Cookson & Niessner, 2020).


### 2. Averaging Sentiment Scores 
Next, we average VADER and TextBlob scores for each comment, a method borrowed from Akarsu and Yilmaz to balance model-specific quirks. It’s a pragmatic step to reduce noise, but it raises a thought: what if one tool consistently outperforms the other for certain datasets? For now, averaging seems a reasonable compromise.  

```python  
# Initialize sentiment analyzer  
analyzer = SentimentIntensityAnalyzer()  

# Function to compute sentiment score  
def compute_sentiment(text):  
    vader_score = analyzer.polarity_scores(text)["compound"]  # VADER sentiment score  
    textblob_score = TextBlob(text).sentiment.polarity  # TextBlob sentiment score  
    avg_score = (vader_score + textblob_score) / 2  # Average score  

    # Convert to binary sentiment: 0, -1, or 1  
    if avg_score > 0:  
        return 1  
    elif avg_score < 0:  
        return -1  
    else:  
        return 0  
```

**Key Assumption**:  
Averaging assumes both tools are equally reliable. Robustness checks (e.g., using only VADER) could validate this choice.


### 3. Creating Sentiment Dummy Variables  
Here, we transform the averaged scores into a binary index: 1 for positive (avg_score > 0), -1 for negative (avg_score < 0), and we exclude neutral comments (avg_score = 0). This aligns with the study’s view that neutral opinions have less impact on market behavior. I pushed for this strict 1/-1 split to sharpen disagreement measurement, favoring clarity over a continuous scale (-1 to +1). Subtle differences—like 0.2 versus 0.4—might blur the conflict we aim to capture.  

```python  
# Apply sentiment analysis with progress tracking  
from tqdm import tqdm  
tqdm.pandas(desc="Processing Sentiment Scores")  
df["Sentiment Score"] = df["Reply Content"].progress_apply(compute_sentiment)  

# Filter out neutral scores  
df = df[df["Sentiment Score"] != 0]  
```

**Debate Point**:  
Excluding neutrals risks losing nuanced signals. Akarsu and Yilmaz (2024) found that including neutrals weakened the relationship between disagreement and trading volume, suggesting neutrals dilute market-moving sentiment. However, a sensitivity analysis using a three-tier system (-1, 0, 1) could reveal if this trade-off is justified.


### 4. Aggregating Sentiment Values 
We then compute `sent_t`, the daily average of these binary scores. This aggregates individual opinions into a collective signal—powerful yet reductive, like distilling a noisy crowd into a single voice.  

```python  
# Sort by date  
df = df.sort_values(by="Reply Date")  

# Calculate daily average sentiment (sent_t)  
daily_sentiment = df.groupby("Reply Date")["Sentiment Score"].mean().reset_index()  
daily_sentiment.rename(columns={"Sentiment Score": "sent_t"}, inplace=True)  
```

**Interpretation**:  
- `sent_t = 1`: All comments are positive.  
- `sent_t = -1`: All comments are negative.  
- `sent_t ≈ 0`: Polarized opinions.


### 5. Calculating the Disagreement Index   
The Disagreement Index (`disag_t`) is calculated as:  
`disag_t = sqrt(1 - sent_t^2)`  
Here, `sent_t` ranges from -1 to 1. The index peaks (≈ 1) when opinions are evenly split (sent_t ≈ 0) and drops near zero when sentiment is lopsided (sent_t ≈ ±1). It’s an elegant formula, but it assumes a polarized world—what if opinions fall into grayer zones?  

```python  
import numpy as np  

# Calculate disagreement index (disag_t)  
daily_sentiment["disag_t"] = np.sqrt(1 - daily_sentiment["sent_t"]**2)  

# Display initial results  
print(daily_sentiment.head())  
```

**Mathematical Insight**:  
This formula is equivalent to the standard deviation of a binary variable (1/-1), scaled to [0,1]. It quantifies the spread of opinions without assuming normality.

## References  
- Akarsu, S., & Yilmaz, N. (2024). *Social media disagreement and financial markets: A comparison of stocks and Bitcoin*. *Economics and Business Review*, 10(4), 189-213.  
- Antweiler, W., & Frank, M. Z. (2004). *Is all that talk just noise? The information content of Internet Stock Message Boards*. *The Journal of Finance*, 59(3), 1259–1294.  
- Cookson, J. A., & Niessner, M. (2020). *Why don’t we agree? Evidence from a social network of investors*. *The Journal of Finance*, 75(1), 173–228.  
- Hong, H., & Stein, J. C. (2007). *Disagreement and the stock market*. *Journal of Economic Perspectives*, 21(2), 109–128.  

 
