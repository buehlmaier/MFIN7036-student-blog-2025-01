---
Title: Fed Watching: Web Scraping and Data Preprocessing (by Group "LingoBingo")
Date: 2025-02-23 15:00
Category: Progress Report
Tags: Group LingoBingo
---


## Introduction

The Fed Funds Rate is a benchmark interest rate that influences monetary policy, economic management, financial markets, asset prices, and has a global impact. Its importance is therefore evident, making the study of factors related to the Fed Funds Rate highly meaningful. 

Our project aims to analyze Federal textual information to predict future rate using text analysis and NLP. We will base our analysis on text data from FOMC statements, minutes, SEP reports, and public speeches by officials.

Recent studies have consistently highlighted the critical role of Federal Reserve communications in shaping the Fed Funds Rate by influencing beliefs about monetary policy and other economic fundamentals. Our project will use the latest data to explore the relationship between Federal textual information and the Fed Funds Rate.

We have performed data preprocessing and constructed a FOMC text dataset to provide a high-quality foundation for subsequent interest rate prediction models. This blog will focus on our data collection, preprocessing, and database establishment processes.


## Data Collection

### Financial Data

We obtain the data for both the federal funds target rate and the effective federal funds rate by directly downloading it from [the Federal Reserve Bank of St. Louis website](https://fred.stlouisfed.org/).

### Text Data

We obtain text data from FOMC statements, minutes, SEP reports, and public speeches by officials through [the official website of the Federal Reserve](https://www.federalreserve.gov).

The following table summarizes the basic information for the various documents that we analyze.

![Table: Text Data Info]({static}/images/LingoBingo_Text-Data_image-description.png)

We collect text data through web crawling using Python. The specific process is as follows:

* Import libraries

We use Python's `Requests` and `BeautifulSoup` libraries to parse HTML pages and combine them with `PyPDF2` to process PDF documents. The code we use is as follows:
```python
import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
import re 
from urllib.parse import urljoin
```

* Data Scraping

The time range of the project spans from 2000 to 2024. However, before 2020, statements, minutes, seps and transcripts had a different URL format. Therefore, we need two different URLs. The code we use is as follows:
```python
# URLs for current and historical FOMC data
current_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
historical_url = "https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm"
```

The original text often contains garbled characters due to encoding errors. To handle disturbances from special characters and redundant spaces, we have also defined a function to perform data cleaning.The code we use is as follows:
```python
def clean_text(text):
    text = text.replace("\xa0", " ")   #Replace non line breaking spaces with regular spaces
    text = text.replace("â\x80\x93", "-")   # Fix encoding errors for short dashes (-)
    text = text.replace("â\x80\x99", "'")   # Fix the encoding error of the right single quotation mark (')
    text = text.replace("â\x80\x94", "-")   # Fix encoding errors for long dashes (-)
    text = text.replace("â\x80\x9c", '"')   # Fix encoding error of left double quotation mark (")
    text = text.replace("â\x80\x9d", '"')   # Fix encoding error of left double quotation mark (")
    text = text.replace("â\x80\x98", "'")    # Fix encoding error of left single quotation mark (')
    text = text.replace("â\x80\x9e", '"')   # Fix encoding errors of double quotation marks („) (some language usage)
    text = text.replace("â\x80\x9f", '"')   # Fix encoding errors in double quotes (‟)
    text = text.replace("â\x80\x91", "-")   # Fix encoding errors for hyphens (-)
    
    text = re.sub(r"\s+", " ", text).strip()   #Merge consecutive whitespace characters into a single space and remove the leading and trailing spaces
    
    return text
```

Next, we define two functions(`scrape_fomc_historical_data(url)` and `scrape_fomc_data(url)`) to scrape data.
In these functions, we first make a GET request to the incoming URL to retrieve the page content, which is then parsed using BeautifulSoup. The code we use is as follows:
```python
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
```
Then, the functions traverse all the links on the page and determine the type of document based on the 'href' attribute of each link: statements, minutes, transcripts, or SEP.

Since we use two URLs to obtain data from different time frames, we merge the results after scraping the data. The code we use is as follows:
```python
# Scrape data from the current FOMC calendar
current_statements, current_minutes, current_seps = scrape_fomc_data(current_url)
# Scrape data from the historical FOMC calendar
historical_statements, historical_minutes, historical_seps, transcripts = scrape_fomc_historical_data(historical_url)
# Combine the results
statements = current_statements + historical_statements
minutes = current_minutes + historical_minutes
# Filling a little bug...(some dates have the length of 7 ...)
minutes = [item for item in minutes if len(item["date"]) == 8]
seps = current_seps + historical_seps
```

Finally, we structure the data by organizing each text source into a table, which includes the time and the corresponding text. Then, we export the data to Excel. A sample output is shown below:

![Picture: Sample Output]({static}/images/LingoBingo_Sample-Output_image-description.png)


## Data Preprocessing

To maintain the integrity of data, we preprocess statements, minutes, seps and transcripts separately.

Tokenization, stemming, and lemmatization help extract key emotional vocabulary and enhance the model's ability to understand policy texts, thereby improving the accuracy of sentiment classification. These are important steps in the preprocessing phase. We use the NLTK package to perform these tasks.

First, we import NLTK package and download essential sources. The code we use is as follows:
```python
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

Then take the statements as an example to show our codes for performing tokenization, lower case conversion, punctuation removal, stop word removal, stemming, and lemmatization. The code we use is as follows:
```python
for statements in statements:
   
    # read the text in statements into value
    value = statements['text']
    
    # Tokenization + Lower Case Conversion + Remove Punctuation
    words = [w for w in nltk.word_tokenize(value.lower()) if w.isalpha()]
	   
	#  Stop Word Removal
    no_stops = [t for t in words if t not in stopwords.words('english')] 
	
    #  Stemming
    stems = [stemmer.stem(word) for word in no_stops]

    # Lemmatization
    lemmas = [lemmatizer.lemmatize(word) for word in no_stops]

    # store into new_statements
    new_statements.append({
        'date': statements['date'],
        'type': statements['type'],
        'original': value,
        'tokenized': words,
        'stems': stems,
        'lemmas': lemmas
    })
```

In future work, we will perform additional preprocessing steps, such as using n-Grams, to further enhance the precision and reliability of our project.

## Database Establishment

To facilitate further analysis, we create two SQLite databases(`FOMC_lemmas_data` and `FOMC_stems_data`) to store the processed data. The code we use is as follows:
```python
import sqlite3
conn = sqlite3.connect('FOMC_lemmas_data.db')
conn = sqlite3.connect('FOMC_stems_data.db')

# create a cursor
cursor = conn.cursor()

# create a table for lemmas
cursor.execute('''
CREATE TABLE IF NOT EXISTS FOMC_lemmas_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    date TEXT NOT NULL,
    lemmas TEXT NOT NULL
)
''')

# create a table for stems
cursor.execute('''
CREATE TABLE IF NOT EXISTS FOMC_stems_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    date TEXT NOT NULL,
    stems TEXT NOT NULL
)
''')

# define a function to insert lemmas
def insert_lemmas(data_list):
    cursor.executemany(
        'INSERT INTO FOMC_lemmas_features (type, date, lemmas) VALUES (?, ?, ?)', 
        [(item["type"], item["date"], ', '.join(item["lemmas"])) for item in data_list]
    )
    
# define a function to insert stems
def insert_stems(data_list):
    cursor.executemany(
        'INSERT INTO FOMC_stems_features (type, date, stems) VALUES (?, ?, ?)', 
        [(item["type"], item["date"], ', '.join(item["stems"])) for item in data_list]
    )

# insert our lemmas
insert_lemmas(new_seps)
insert_lemmas(new_minutes)
insert_lemmas(new_statements)
insert_lemmas(new_transcripts)

# insert our stems
insert_stems(new_seps)
insert_stems(new_minutes)
insert_stems(new_statements)
insert_stems(new_transcripts)

# commit the transaction
conn.commit()

# close the cursor and connect
cursor.close()
conn.close()
```


## Word Cloud

To gain an overview of our textual data, we create word clouds based on the two aforementioned databases, which are useful tools for quickly identifying high-frequency words within a body of text. The code we use is as follows:
```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Combine all texts
all_texts = []

#if we use stems for further analysis
#for lst in [stems]:

#if we use lemmas for further analysis
for lst in [lemmas]:   
    for item in lst:
        # Ensure the item is a tuple and has at least 4 elements
        if isinstance(item, tuple) and len(item) >= 4:
            text = item[3]  # Get the fourth index (value at index 3)
            # Ensure 'text' is a string
            if isinstance(text, str):
                all_texts.append(text)
            else:
                print(f"Non-string data: {text}")
        else:
            print(f"Item is not a valid tuple: {item}")

# Join the texts into a single string
text_string = ' '.join(all_texts)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_string)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide the axes
plt.show()
```

We generate two word clouds as follows, highlighting key metrics that the FOMC pays significant attention to, such as monetary policy and the labor market.

![Picture: Word Could_Stem]({static}/images/LingoBingo_Word-Cloud-Stem_image-description.png)             

![Picture: Word Could_Lemmas]({static}/images/LingoBingo_Word-Cloud-Lemmas_image-description.png)


## Future Directions

* Perform more comprehensive data preprocessing.

* Perform sentiment analysis to obtain polarity scores.

* Apply machine learning to establish a relationship between the Fed Funds Rate and polarity scores. Then, use our trained model to predict the Fed Funds Rate.


We will share more details in our next blog post. Thank you for reading our blog!

