---
Title: Problems and Solutions with Web Scraping (by Group "Byte Bards")
Date: 2025-02-22 15:40
Category: Reflective Report
Tags: Group Byte Bards
---

## Introduction

First of all, I want to briefly introduce the topic we aim to explore, so that future readers can understand our objectives.
Our title is **"From Headlines to Trades: Can News Clusters Predict Markets?"**

Our task is to grab Intel company related news topics and descriptions from the web and use this part of the content as a data source for text analysis. 
Through the **clustering** of news content, Intel related news is divided into several categories (Let's say 3 clusters as example).
Then, we match news articles from each category to their respective publication dates, and the movement of Intel's stock price (as measured by the rise or fall of the company's stock price) is checked the day after the news release.
The final goal is to find out whether a certain type of news has a positive impact on Intel’s stock price, a negative impact, or no impact at all.

Therefore, the first step for our group was to crawl Intel-related news content through the web. 
Below we'll describe in detail the versions of code we tested, the results, and the rationale behind adopting or rejecting each approach. 

## Grabbing from Yahoo News
Our initial idea was to scrape Yahoo Finance stories containing the keyword "Intel" along with their descriptions and use the news descriptions as our data source. However, after successfully obtaining the data, we found that almost all descriptions were incomplete and ended with "..."

We then attempted to capture the missing text by expanding the amount of content we crawled, but this effort ultimately failed. Upon re-examining the structure of Yahoo Finance's webpage, we discovered that the so-called "description" was not extracted from the actual article but was instead a short snippet displayed below the title. If this snippet ended with "...", we could only capture "...", with no way to retrieve the full text.

After manually reviewing a large sample of data, we found that this snippet was merely a truncated portion of the article's first sentence and held little value for analysis. Given this limitation, we decided to abandon the idea of using article descriptions as our data source and instead shifted our focus to a more meaningful target: analyzing article titles.

The so-call description is a line of small print under the title and only means the front part of the first line in article which is meaningless:

![Picture showing Powell]({static}/images/Byte Bards_01_YahooFinance.png)

We use belowed code to grab the news' title on Yahoo Finance with key word “Intel":

```python
def parse_date(date_str):
    """Convert relative dates like '2 days ago' or '1 day ago' to 'YYYY-MM-DD' format."""
    today = datetime.today()
    match = re.search(r'(\d+) days? ago', date_str)
    if match:
        days_ago = int(match.group(1))
        return (today - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    elif 'hour' in date_str or 'minute' in date_str:
        return today.strftime('%Y-%m-%d')
    else:
        try:
            return datetime.strptime(date_str, "%b %d, %Y").strftime('%Y-%m-%d')
        except ValueError:
            return date_str

def get_article(card):
    """Extract article information from the raw HTML."""
    headline = card.find('h4', 's-title').text.strip()
    source = card.find("span", 's-source').text.strip()
    posted_raw = card.find('span', 's-time').text.replace('·', '').strip()
    posted = parse_date(posted_raw)
    
    # Extract full description
    desc_tag = card.find('p', 's-desc')
    if desc_tag:
        description = desc_tag.get('title', desc_tag.text).strip()
        if description.endswith("..."):
            description = desc_tag.text.strip()
    else:
        description = ""
    
    raw_link = card.find('a').get('href')
    unquoted_link = requests.utils.unquote(raw_link)
    pattern = re.compile(r'RU=(.+)/RK')
    match = re.search(pattern, unquoted_link)
    clean_link = match.group(1) if match else None

    article = {
        'Headline': headline,
        'Source': source,
        'Posted': posted,
        'Description': description,
        'Link': clean_link
    }
    return article

def get_the_news(search):
    """Run the main program."""
    template = 'https://news.search.yahoo.com/search?p={}'
    url = template.format(search)
    articles = []
    links = set()

    while True:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        cards = soup.find_all('div', 'NewsArticle')
        
        # Extract articles from page
        for card in cards:
            article = get_article(card)
            link = article['Link']
            if link and link not in links:
                links.add(link)
                articles.append(article)

        # Find the next page
        try:
            url = soup.find('a', 'next').get('href')
            sleep(1)
        except AttributeError:
            break
```
However, there are two questions in the result.

_Question 1_: If the titles are truncated, they will end with ellipses, which may affect content analysis. However, we currently believe that simply removing the ellipses will be sufficient for our analysis.

_Question 2_: Yahoo News employs anti-scraping measures that restrict access to a maximum of 100 pages, with each page containing 10 articles, resulting in a total of 1,000 articles. Out of these, we have successfully retrieved approximately 700, but the available time range covers only one week, which is insufficient for clustering analysis.

While the first issue is **manageable**, the second presents a significant limitation. Given this constraint, we are considering modifying our research topic to: "_Scraping the top 1,000 news headlines from Yahoo News for 50 randomly selected companies and analyzing the frequency of the term 'risk' in relation to stock price volatility over the course of a week."_

After encountering difficulties in collecting data from Yahoo Finance, our group actively explored alternative methods to obtain relevant data.
## Grabbing from Google

Next we try to grab Intel-related news topics from google news with the following code. The key part here is we use multiple ways to get the news released date.

```python
# 1. try to usearticle:published_time meta tag
    date_tag = soup.find('meta', {'property': 'article:published_time'})
    if date_tag and 'content' in date_tag.attrs:
        date = date_tag['content'][:10]  # 获取日期部分 (yyyy-mm-dd)

    # 2.if article:published_time is failed to find date，try other meta labels
    if date == "Unknown date":
        date_tag = soup.find('meta', {'name': 'date'})
        if date_tag and 'content' in date_tag.attrs:
            date = date_tag['content'][:10]

    # 3. if still can't get the date, try to use the text date
    if date == "Unknown date":
        date_tag = soup.find('time')
        if date_tag and date_tag.get('datetime'):
            date = date_tag['datetime'][:10]
        elif date_tag and date_tag.text:
            date = date_tag.text.strip()[:10]
```
The main issue with this result is that incomplete date information poses a significant challenge for clustering analysis. While some news articles include dates, others do not, making it impossible to determine the publication date for each article and, consequently, to match them with stock prices. Currently, approximately 40% of the news articles lack date information.

![Picture showing Powell]({static}/images/Byte Bards_01_GoogleNewsResults.png)
You can see that in the "date" column, 40% of them is "Unknown date".

Another potential issue is that we are currently unaware of Google News' scraping limits, which may restrict our ability to extract the desired volume of data.
## Grabbing from Yahoo Finance

On this website, [Yahoo Finance of INTC](https://finance.yahoo.com/quote/INTC/news/), **article titles are complete, free of ellipses, and of high quality, all of which are finance-related**. However, we are unable to extract date information and can only scrape a maximum of 60 articles. It is unclear whether this limitation stems from a coding issue or a restriction imposed by the website. Ideally, we aim to scrape data covering 1 to 2 months, including dates, titles, and, preferably, the source (which is available on the site). However, the current code is not yet fully developed.

## Conclusion
In summary, if we pursue the first option, it is highly likely that we will need to change our topic. For the second option, we are still uncertain whether we can overcome the data limitations; however, if successful, we could obtain a substantial volume of data (even though 40% of the articles lack dates, we could scrape up to 1,000,000 articles). The third option appears the most promising, but it is also the least developed at this stage.

We look forward to communicating with you

Thanks for reading :D
