---
Title: Scraping Up-to-Date Earnings Call Transcripts (by Group "GenZ AI")
Date: 2025-02-23 09:00
Category: Progress Report
Tags: Group GenZ AI
---

By Group "GenZ AI"

For our recent endeavor focused on earnings call text analysis, the process of data acquisition was a rollercoaster ride filled with challenges and triumphs. Let me take you through our experience.

## Option1: The Drawback of Existing Datasets

When launching our NLP project, we first explored existing datasets to accelerate development. Public resources like Huggingface's [lamini/earnings-calls-qa](https://huggingface.co/datasets/lamini/earnings-calls-qa) and [jlh-ibm/earnings_call](https://huggingface.co/datasets/jlh-ibm/earnings_call) provided valuable labeled examples.

However, a critical gap emerged: the most recent data only covered Q3 2023. Given financial markets' volatility, stale data risks generating outdated insights. This limitation forced us to build our own web scraper for **real-time** earnings call transcripts.


## Option2: The Roadblock at Seeking Alpha

We explored open-source projects and identified [**Seeking Alpha**](https://seekingalpha.com/) as a popular source for earnings call data. Its key advantage: structured access to transcripts via stock tickers (e.g., AAPL). Here's our streamlined approach:

### Step 1: Fetch S&P 100 Tickers
```python
# Get S&P 100 tickers
import pandas as pd
from bs4 import BeautifulSoup
import requests

url = "https://en.wikipedia.org/wiki/S%26P_100"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find_all('table')[2]  # Target table position
df = pd.read_html(str(table))[0]
```

### Step 2: Initial Scraping Attempt 
We then tried scraping transcripts using Selenium:
```python
# Basic scraping attempt (ultimately failed)
from selenium import webdriver

browser = webdriver.Chrome()
browser.get("https://seekingalpha.com/symbol/AAPL/transcripts")
time.sleep(5)  # Wait for page load

# Interactive CAPTCHA triggered here
html = browser.page_source  
# Subsequent parsing became impossible...
```

**Key Obstacle**: The Interactive CAPTCHA system blocked automated access. Even with proxy rotation, request throttling, and header spoofing, we couldn't bypass detection consistently, forcing us to abandon this approach.

![Picture showing the interactive CAPTCHA]({static}/images/group-GenZ-AI_01_image-CAPTCHA.jpeg)
![Picture showing the interactive CAPTCHA](/images/group-GenZ-AI_01_image-CAPTCHA.jpeg)
![Picture showing Powell]({static}/images/group-Fintech-Disruption_Powell.jpeg)

## Option 3: Structured Success at The Motley Fool
We discovered [**The Motley Fool**](https://www.fool.com/earnings-call-transcripts/)'s Earnings Call Transcripts offered a goldmine of well-structured data. Its paginated design (page=1 to page=500) enabled systematic scraping of ~10,000 transcripts. 


**Key Features**

- **Predictable URL patterns**: https://www.fool.com/earnings-call-transcripts/?page=x

- **Consistent CSS classes for data extraction**

- **Minimal anti-scraping mechanisms**



### Step 1: Paginated URL Collection
```python
from bs4 import BeautifulSoup
import requests

# Generate all page URLs
base_url = 'https://www.fool.com/earnings-call-transcripts/?page='
page_urls = [f"{base_url}{x}" for x in range(1, 501)]
```
### Step 2: Transcript Link Extraction
```python
# Extract individual transcript URLs
transcript_urls = []
for page in page_urls:
    soup = BeautifulSoup(requests.get(page).text, "lxml")
    links = soup.select(".flex.py-12px.text-gray-1100 a[href]")
    transcript_urls.extend(f"https://www.fool.com{link['href']}" for link in links)

# Deduplicate and filter (every 3rd link is valid)
clean_urls = list(set(transcript_urls[::3]))
```

### Step 3: Data Extraction Workflow
```python
# Initialize storage
transcripts = []
metadata = []

for url in clean_urls:
    soup = BeautifulSoup(requests.get(url).text, "lxml")
    body = soup.find(class_="article-body")
    
    # Extract structured elements
    header = body.find_all("p")[1].text.strip().split("\n")
    company = header[0].split("(")[0].strip()
    ticker = header[0].split("(")[1].replace(")", "").strip()
    date = soup.find(id="date").text
    
    # Store results
    metadata.append({"company": company, "ticker": ticker, "date": date})
    transcripts.append("\n".join(p.text for p in body.find_all("p")[2:]))
```

This approach demonstrates how structural analysis of target websites can overcome scraping limitations encountered elsewhere. The Motley Fool's consistent formatting made it ideal for large-scale NLP data collection, ultimately providing **9,981** clean transcripts for our analysis.

## Option 4: Augmenting Fresh Data with Historical Records
Our web scraping efforts on The Motley Fool successfully captured nearly **10,000 up-to-date earnings call transcripts**, but this dataset skewed heavily toward post-2020 content. To enable longitudinal analysis of corporate communications, we needed historical context beyond this recent window.

We identified a complementary Kaggle dataset [**Scraped Motley Fool Earnings Call Transcripts**](https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts/data) containing 18,755 transcripts from 2017 Q3 to 2023 Q3. By merging these two sources and deduplicating entries using company tickers and call dates, we can created a unified corpus spanning six continuous years (2017-2023). This combined dataset preserves the latest market insights from our scraping work while incorporating critical pre-pandemic and early-pandemic era discussions, ultimately providing a robust foundation for NLP analysis of corporate narratives.



[def]: {static}/images/group-GenZ_AI_01_image-The-Interactive-CAPTCHA.jpeg