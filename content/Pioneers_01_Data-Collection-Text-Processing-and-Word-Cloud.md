---
Title: Data Collection, Text Processing and Word Cloud (by Group "Pioneers")
Date: 2025-02-23 18:12
Category: Reflective Report
Tags: Group Pioneers
---

By Group "Pioneers"

## Background and Goal

Our project aims to explore the relationship between the stock prices of Apple Inc. in the U.S.
stock market and the sentiment found in related financial news articles through sentiment 
analysis. We believe that analyzing the emotional tone of news content can provide valuable 
insights into the market performance of Apple.

To achieve this, we selected Apple’s stock prices from January 2016 to December 2024 as our 
financial data source. Additionally, we collected financial news articles related to Apple from 
February 19, 2016, to November 27, 2024, which will serve as the foundation for our sentiment 
analysis. 

Furthermore, we will discuss the challenges encountered during data processing and integration 
to enhance data quality and analysis accuracy. We hope that this research will offer valuable 
insights into financial analysis, investment strategy, and market psychology.

## Data Collection

In order to obtain relevant news text data of APPLE, we visited the official websites of Yahoo, 
StockTwits and Kaggle, analyzed the data structure of the websites, and found that the websites 
were dynamically refreshed and loaded. We decided to use common tools such as Selenium for text 
crawling. Therefore, we resorted to using the request package in Python to crawl the web page, 
and successfully crawled the relevant news headlines and summary information from 2016 to 2024.

A major challenge encountered is that the web page is in scrolling loading mode, so it is 
necessary to simulate mouse scrolling to obtain web page data, and at the same time, organize 
and merge the information obtained from different data sources into the report.

The detailed steps for crawling APPLE text data include:

1. Use Selenium to simulate the web page and obtain the web page source code.

2. Use BeautifulSoup to extract news headlines, time and other data.

Part of the code we used for data collection is as follows:

```python
# Import Necessary Libraries
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import csv

# Set up and Initialize Chrome WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless") 
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
url = "https://finance.yahoo.com/quote/AAPL/press-releases/"
driver.get(url)
time.sleep(3)

# Scroll down and Get News Items
for _ in range(10): 
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)  
    time.sleep(2)  
page_source = driver.page_source
soup = BeautifulSoup(page_source, 'html.parser')
news_items = soup.select('ul li section div a')

# Extract the Title and Summary of Each News Item
news_data = []
for item in news_items:
    title = item.find('h3')
    summary = item.find('p')

    if title and summary:
        news_data.append({
            'title': title.get_text(strip=True),
            'summary': summary.get_text(strip=True)
        })

# Print the Extracted News Data
for i, news in enumerate(news_data, start=1):
    print(f"news {i}:")
    print(f"headers: {news['title']}")
    print(f"intro: {news['summary']}")
    print("=" * 40)

# Save the News Data to a CSV file
csv_file = "apple_news.csv"
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Headers', 'Detail'])
    for news in news_data:
        writer.writerow([news['title'], news['summary']])
```

As for price data of Apple, we directly downloaded the financial data, specifically the daily 
Open, Close, High, Low, Trading Volume and Return of Apple from January 2016 to December 2024 
from Yahoo Finance. In subsequent analysis, the transaction data and news text data will be 
combined to more comprehensively understand the reasons for market fluctuations and the 
driving factors behind them.

## Data Observation

When examining text related to APPLE, we extract information including not only the news title, 
but also the publication date and author. Regarding stock data, it is important to pay special 
attention to the date and adjusted closing price, as well as whether it has risen or fallen.

###### Text Example (News 1)
 
>Headers: Apple debuts iPhone 16e: A powerful new member of the iPhone 16 family
> 
>Intro: CUPERTINO, Calif., February 19, 2025--Apple® today announced iPhone® 16e, 
>a new addition to the iPhone 16 lineup that offers powerful capabilities at a more 
>affordable price. iPhone 16e delivers fast, smooth performance and breakthrough battery 
> life, thanks to the industry-leading efficiency of the A18 chip and the new Apple C1, 
> the first cellular modem designed by Apple. 

###### Text Example (News 2)

>Headers: Introducing Apple Invites, a new app that brings people together for life’s special
> moments
> 
> Intro: CUPERTINO, Calif., February 04, 2025--Apple® today introduced Apple Invites™, a new 
> app for iPhone® that helps users create custom invitations to gather friends and family for 
> any occasion. With Apple Invites, users can create and easily share invitations, RSVP, 
> contribute to Shared Albums, and engage with Apple Music® playlists. Starting today, users 
> can download Apple Invites from the App Store®, or access it on the web through 
> icloud.com/invites. 

## Data Preprocessing and Word Cloud Analysis

In this part, we will demonstrate how to preprocess stock-related news data using Python 
and visually present the main topics of the news articles through a word cloud. We utilize 
several common Natural Language Processing (NLP) libraries such as NLTK and WordCloud, and 
rely on pandas for data manipulation.


##### Necessary Libraryies

First, we import the necessary libraries, including pandas for handling the dataset, re for 
text cleaning, stopwords and lemmatization tools from nltk, and matplotlib and wordcloud for 
generating and displaying the word cloud. Stopwords are common words that we exclude from 
text analysis because they don’t provide significant value, such as "the", "and", etc. To 
refine our analysis, we also add custom stopwords like "news", "article", etc., to further 
filter unnecessary terms.

```python
# Import Necessary Libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define Custom Stopwords
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
additional_stop_words = {'news', 'article', 'subject', 'update'}
stop_words.update(additional_stop_words)
```

##### Tokenization and Stopword Removal

Next, we preprocess the data. We begin by converting the date column to datetime format for 
easier time-based analysis. We then define two main text processing functions: clean_text, 
which cleans the content by removing URLs, non-alphabetic characters, and extra spaces, and 
preprocess_text, which splits the text into words, removes stopwords, and performs 
lemmatization to ensure the text is cleaner and more standardized. These steps ensure the 
quality of the text data and prepare it for further analysis.

```python
# Convert Date to Datetime Format
df['date'] = pd.to_datetime(df['date'])

# Tokenization
def clean_text(text):
    if pd.isna(text): 
        return ''
    text = re.sub(r'http\S+', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = text.lower() 
    text = re.sub(r'\s+', ' ', text).strip()  
    return text
df['cleaned_title'] = df['title'].apply(clean_text)
df['cleaned_content'] = df['content'].apply(clean_text)

# Stopword Removal
def preprocess_text(text):
    if pd.isna(text):  
        return ''
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)
df['processed_title'] = df['cleaned_title'].apply(preprocess_text)
df['processed_content'] = df['cleaned_content'].apply(preprocess_text)

# Concatenate News Content
text = ' '.join(df['processed_content'].dropna())  

```

##### Word Cloud Analysis

Once the text is preprocessed, we concatenate the cleaned news content into a single large 
string, ready for generating the word cloud. A word cloud is a visual representation of 
the most frequent words in the text, which allows us to easily observe the main themes. 
In this case, we use a custom apple-shaped mask image for the word cloud to create a more 
creative visual effect. Using the WordCloud library, we can quickly generate the word cloud 
and display it via matplotlib.

```python
# Load the Apple-shaped Mask
apple_mask = np.array(Image.open('apple_mask.png')) 

# Generate the Word Cloud
wordcloud = WordCloud(
                      width=1600, height=800, 
                      background_color='white', 
                      mask=apple_mask, 
                      contour_color='black', contour_width=0, 
                     ).generate(text)

# Display the Word Cloud Using Matplotlib
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```
After all, we get the word cloud:

![Picture showing the word cloud]({static}/images/Pioneers_01_Word-Cloud.png)

## Conclusion
So far we have clarified our project goals and completed data collection and 
pre-processing. The clean and prepare text data will support us in the following sentiment 
analysis and model construction. The word cloud helps us visually identify the most frequent 
keywords in the news content of Apple. In the next steps, sentiment analysis could be applied 
to assess the market sentiment based on these keywords, which would further aid investment 
decisions.