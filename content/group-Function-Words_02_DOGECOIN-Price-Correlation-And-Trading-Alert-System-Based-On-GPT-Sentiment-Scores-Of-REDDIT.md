---
Title: DOGECOIN Price Correlation And Trading Alert System Based On GPT Sentiment Scores Of REDDIT (by Group "Function Words")
Date: 2025-03-9 17:33
Category: Reflective Report
Tags: Group Function Words
---

By Group *"Function Words"*

<p style="font-size: 14px;">
  Codes and Blogs By
  <span style="font-weight: bold; font-style: italic;">Liu Qing Yuan</span>, 
  <span style="font-weight: bold; font-style: italic;">Wang Shu Yu</span>, 
  <a href="https://www.linkedin.com/in/zhenzhe-xiong-453929216/" style="font-weight: bold; font-style: italic; text-decoration: none; color: black;">Xiong Zhen Zhe</a>, 
  and
  <span style="font-weight: bold; font-style: italic;">Zhou Zi Qi</span>.
  This is a <span style="font-weight: bold; font-style: italic;">Second Blog </span> post. If you are looking for our previous post, please move to  <a href="https://buehlmaier.github.io/MFIN7036-student-blog-2025-01/sentiment-analysis-of-dogecoin-based-on-reddit-by-group-function-words.html" style="font-weight: bold; font-style: italic; text-decoration: none; color: Red;"> Sentiment Analysis of Dogecoin Based on Reddit</a>.
</p>




## Abstract

 In our ongoing project, we aim to explore the relationship between social media sentiment and the price movements of Dogecoin, a popular meme cryptocurrency. Our initial focus was on Twitter, but due to the high costs associated with Twitter's API, we pivoted to Reddit, which offers a free API and a wealth of discussion data. This report outlines our progress, methodologies, and findings so far.

Below is a Sequence Diagram of our program.

<div style="text-align: center;">
    <img src="{static}/images/group-Function-Words_02_mermaid1.png" alt="mermaid1" >
</div>
<br>


## 1. Data Collection from Reddit

**Objective**:  
To gather a substantial amount of user-generated content related to Dogecoin for sentiment analysis.

**Methodology**:  
We utilized the Reddit API to scrape posts and comments from subreddits like "CryptoCurrency" that discuss Dogecoin. Here's a breakdown of our approach:

- **API Configuration**: We set up Reddit API credentials (`client_id`, `client_secret`, `user_agent`) to legally access and scrape data.

- **Keyword Search**: We searched for posts containing the keyword "dogecoin" and limited the number of posts to avoid overwhelming our system.

- **Comment Extraction**: For each post, we extracted up to 300 comments, ensuring we captured the most active discussions. We handled "MoreComments" structures to avoid redundant data.

- **Data Storage**: We stored the post titles, comments, and creation dates in a structured format, which we later exported as a CSV file.

**Code Example**:
```python
import praw

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="YOUR_USER_AGENT"
)

subreddit = reddit.subreddit("CryptoCurrency")
posts = subreddit.search("dogecoin", limit=10)

for post in posts:
    print(f"Title: {post.title}")
    print(f"Comments: {post.comments.list()}")

```

## 2. Preliminary Text Visualization: Word Cloud

**Objective**:  
To quickly visualize the most frequently mentioned words in the comments, providing a global overview of the community's focus.

**Methodology**:  

- **Data Preparation**: We concatenated all comments into a single large text.

- **Word Cloud Generation**: Using a word cloud tool, we filtered out common English stopwords (e.g., "the", "and", "of") and visualized the remaining words based on their frequency.

**Visualization**:  
```python
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the uploaded CSV file
file_path = 'dogecoin_reddit_data.csv'
df = pd.read_csv(file_path)

# Combine all comments into a single text string
text_data = ' '.join(df['comment'].dropna())

# Generate word cloud using built-in stopwords from the wordcloud package
wordcloud = WordCloud(stopwords=WordCloud().stopwords, background_color='white', width=800, height=400).generate(text_data)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

<div style="text-align: center;">
    <img src="{static}/images/group-Function-Words_02_wordcloud.png" alt="wordcloud" >
</div>
<br>

**Insight**:  
The word cloud revealed that terms like "dogecoin", "buy", "elon", "moon", and "hold" were predominant, indicating the community's focus on investment strategies and market optimism.



## 3. Sentiment Analysis Using Large Language Models

**Objective**:  
To go beyond simple keyword analysis and employ large language models (LLMs) like GPT to provide nuanced sentiment scores for the comments.

**Methodology**:  
- **Daily Aggregation**: We grouped comments by their posting date.

- **Text Chunking**: For days with a high volume of comments, we split the text into manageable chunks to fit within the LLM's processing limits.

- **Prompt Design**: We crafted detailed prompts instructing the LLM to score the text on metrics like Buy Score, Bullish Score, Hype Score, and to provide a sentiment word (e.g., "bullish", "fearful").

- **Model Interaction**: We submitted the text chunks to the LLM, which returned JSON objects containing the scores and sentiment words.

- **Data Aggregation**: We averaged the scores for each day to create a daily sentiment index.

**Code Example**:
```python
import openai

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Analyze the sentiment of the following text and provide a Buy Score, Bullish Score, Hype Score, and a sentiment word:\n\n'Dogecoin to the moon! Buy now before it's too late.'",
  max_tokens=60
)

print(response.choices[0].text)
```


## 4. Integration with Dogecoin Price Data

**Objective**:  
To align sentiment data with Dogecoin's historical price data for correlation analysis.

**Methodology**:  

- **Price Data Collection**: We obtained a CSV file containing daily Dogecoin prices, including open, high, low, close, and volume.

- **Data Merging**: We merged the sentiment scores with the price data based on the date.

- **Data Processing**: We calculated daily price changes and handled any missing data points.

**Code Example**:
```python
import pandas as pd

# Load sentiment data
sentiment_data = pd.read_csv('sentiment_scores.csv')

# Load price data
price_data = pd.read_csv('dogecoin_prices.csv')

# Merge data on the 'date' column
merged_data = pd.merge(sentiment_data, price_data, on='date', how='inner')

# Calculate daily price changes
merged_data['price_change'] = merged_data['close'].pct_change()

# Handle missing data (if any)
merged_data.dropna(inplace=True)

# Save the merged dataset
merged_data.to_csv('merged_sentiment_price_data.csv', index=False)
```

## 5. Correlation and Extreme Case Analysis

**Objective**:  
To investigate whether there is a significant correlation between Reddit sentiment and Dogecoin price movements.

**Methodology**:  

- **Correlation Analysis**: We calculated Pearson correlation coefficients between sentiment scores (e.g., Buy Score) and price changes.

- **Lag Analysis**: We examined whether sentiment scores from one day could predict price changes on the following day.

- **Extreme Case Analysis**: We identified days with the largest price increases and decreases and analyzed the corresponding sentiment scores.

**Visualization**:  

<div style="text-align: center;">
    <img src="{static}/images/group-Function-Words_02_scatterplot.png" alt="scatterplot" >
</div>
<br>

**Insight**:  
Initial findings suggest a weak correlation between sentiment scores and price changes. However, in extreme cases, such as days with significant price spikes or drops, the sentiment scores showed more pronounced patterns.



## 6. Challenges and Possible Improvement

**Challenges**:  

- **API Limitations**: Reddit's API restricts the number of requests, slowing down data collection.

- **Influential Comments**: Many comments come from casual users, which may not significantly impact market movements.

- **Model Limitations**: The current sentiment analysis model may be too simplistic to capture the full spectrum of emotions.

**Possible Improvement**:  

- **Advanced Sentiment Analysis**: We can use to explore more sophisticated tools like DeepSeek, which offers richer emotional analysis.

- **Expanded Data Sources**: We can incorporate data from other social media platforms to gain a more comprehensive view of market sentiment.

- **Machine Learning Models**: We can explore the use of machine learning models to better predict price movements based on sentiment data.

## The Big Picture of Our Workflow

The overarching process of our project can be summarized as follows:  

1. **Scrape Reddit Data**: Collect social media sentiment data from Reddit related to Dogecoin.  

2. **Clean and Organize Data**: Preprocess the raw data to ensure it is ready for analysis.  

3. **Score Sentiment Using Large Language Models**: Use advanced language models (e.g., GPT) to assign sentiment scores to the collected comments.  

4. **Merge with Dogecoin Price Data**: Combine the sentiment scores with historical Dogecoin price data.  

5. **Perform Time-Series Analysis**: Analyze the data over time to identify trends and patterns.  

6. **Visualize and Gain Insights**: Create visualizations to uncover the relationship between sentiment and price movements.  

Through this method, we aim to provide a data-driven answer to questions like:  

- Does Dogecoin's price rise because of increased discussion, or do people discuss it more after the price rises?  

- Can we detect early signals of price movements from Reddit discussions?  

This approach allows us to explore whether social media sentiment can serve as a leading indicator for cryptocurrency price changes, offering valuable insights for both researchers and investors.

  
<div style="text-align: center;">
    <img src="{static}/images/group-Function-Words_02_mermaid2.png" alt="mermaid2" width="600">
</div>
<br>

## Conclusion

Our project has made significant strides in understanding the relationship between Reddit sentiment and Dogecoin price movements. While initial results show a weak correlation, we believe that with more advanced tools and expanded data sources, we can uncover deeper insights. This research not only contributes to the academic understanding of cryptocurrency markets but also has practical implications for investors and traders.



## Visual Summary

1. **Word Cloud**: Highlights the most frequently discussed terms in Dogecoin-related Reddit comments.
2. **Correlation Scatter Plot**: Illustrates the relationship between sentiment scores and price changes.
3. **Price and Sentiment Over Time**: A time series plot showing how sentiment scores and Dogecoin prices evolve together.




> **Team Function Words**  
> *<span style="color: #FF5733; font-family: 'Courier New', monospace;">Passionate about cryptocurrency and data analysis.</span>*