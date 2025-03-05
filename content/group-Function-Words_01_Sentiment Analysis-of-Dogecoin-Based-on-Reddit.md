---
Title: Sentiment Analysis of Dogecoin Based on Reddit (by Group "Function Words")
Date: 2025-02-23 18:37
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
  This is a <span style="font-weight: bold; font-style: italic;">First Blog </span> post. If you are looking for more progress, please move to our  <a href="https://buehlmaier.github.io/MFIN7036-student-blog-2025-01/" style="font-weight: bold; font-style: italic; text-decoration: none; color: black;">Second Blog</a><sup><a href="#note1" style="text-decoration: none; color: black;"> [1]</a></sup>.
</p>

<p id="note1" style="font-size: smaller; color: gray;">
  [1] We are still working on it!
</p>



## Abstract

In case you don't remember who we are or what we're up to, we threw together this little recap to help you out.

<p style="font-size: 16px;">
  We are group "Funtion Words", passionating about cryptocurrency and data analysis. <del style="color: red; font-weight: ;">Our goal was to leverage Python programming to scrape and analyze Twitter content from key opinion leaders (KOLs) in the crypto space. By extracting and evaluating their tweets, we aimed to provide data-driven insights to help investors make informed decisions on whether to buy or sell meme coins.</del> Our project combines web scraping, natural language processing, and financial analysis to navigate the volatile world of meme cryptocurrencies.
</p>

You can click [Here](https://www.canva.com/design/DAGS5rbbCLs/x1U6ikQD8CuTpggL89-uoQ/edit?utm_content=DAGS5rbbCLs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) to access our PowerPoint presented in class.

As part of our NLP project, we initially wanted to scrape the sentiment of Dogecoin tweets on Twitter. However, we quickly realized that Twitter's API is too costly, especially for large data scraping. This forced us to look for a less expensive alternative, and this prompted us to explore Reddit's API.

## Our Approach

We were interested in knowing what sort of sentiment people are using in social media comments and posts about Dogecoin and whether it correlates with the price movements of Dogecoin. We first thought of using Twitter to get social media sentiment since it's a very popular platform for talking about financial topics. However, Twitter's API is too costly, so we figured we'd do something different and do Reddit instead, which has a free API and lots of discussion data.

The code we use is as follows:
```python
# List to store the fetched results
data = []

# Iterate through each selected post
for idx, submission in enumerate(selected_submissions):
    print(f"Fetching submission {idx + 1}/{len(selected_submissions)}: {submission.title}") # Print the title of the post being fetched

    # Handle the post's comments to avoid loading too many redundant comments
    submission.comments.replace_more(limit=0) # Replace "MoreComments" with empty, to avoid fetching excessive comments
    comments = submission.comments.list() # Get all comments (excluding redundant parts)

    # Get the text of the comments, limited to 'comment_limit' number of comments
    comment_texts = [comment.body for comment in comments[:comment_limit]]

    # Get the post creation date and format it as a string (Year-Month-Day Hour:Minute:Second)
    created_date = datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')

    # Add the current post and its comment data to the result list
    data.append({
        'headline': submission.title, # The title of the post
        'comments': comment_texts, # The content of the comments
        'date': created_date # The date the post was created
    })

    # Sleep for 'sleep_time' seconds after fetching each post to avoid making requests too quickly
    time.sleep(sleep_time)

return data
```

We started by extracting data from the "CryptoCurrency" subreddit, as well as posts and comments mentioning Dogecoin. We extracted Dogecoin posts using Reddit's search API and examined the titles and comments for sentiment. Sentiment analysis was done by implementing the TextBlob library, which provides a sentiment score for each post and comment between <span style="font-weight: bold; font-style: None; color: Red;">-1</span> (<span style="font-weight: bold; font-style: None; color: Red;">Negative</span>) and <span style="font-weight: bold; font-style: None; color: Black;">1</span> (<span style="font-weight: bold; font-style: None; color: Black;">Positive</span>).

## Dataset

1. The dataset of reddit comments: 
<table style="width:100%">
  <tr>
    <th style="background-color: #f2f2f2; text-align: center;">Headline</th>
    <th style="background-color: #f2f2f2; text-align: center;">Comment</th>
    <th style="background-color: #f2f2f2; text-align: center;">Date</th>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">[deleted]</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">Surviving hackers is bullish</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">As long as degens can still buy and sell DOGE they don't care what it actually does or doesn't do lol</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">"69%? Nice"</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">Doge investors: pump the news</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">Already been patched on most networks. price isn't changing</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">Why didn't hacker just click and drag the price to $1,000?</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">So buy the dip?</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">Wait, isn't DOGE's source code basically a fork of Litecoin? Does that mean LTC has this kind of flaw too?</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">Who would imagine that a meme coin without proper development team and security updates is vulnerable to attacks... keep on buying, it will pump tomorrow for sure!</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">DOGE hodlers have no clue what any of this means so they'll just buy more and keep dick riding Elmo</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">So, Moon Soon?</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">All this does is improve DOGE in the long run and works as a non-security short?</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">Blockchair's node count is very inaccurate. <a href="https://what-is-dogecoin.com/nodes/">https://what-is-dogecoin.com/nodes/</a> shows 14563 nodes.  There is no evidence of any flaw.  Some rando shorted hard, made a tweet and everyone is taking it as facts.  If there was a node crash it would have blown up during the crash supposedly on Dec 4.   I for one run an older version node, never taken offline, never crashed.</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">Seems like FUD. Based on other discussion threads there was no significant change in the number of nodes. Anyone got proof of this besides this tweet?</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
  <tr>
    <td style="text-align: center;">Hacker exploits DOGECOIN flaw, crashing 69% of nodes and exposing a vulnerability that could have taken down the entire network.</td>
    <td style="text-align: center;">Glad DOGE people don't interest themselves with anything informative!</td>
    <td style="text-align: center;">2024/12/12 14:18</td>
  </tr>
</table>
2. The dataset of price:
<table style="width:100%">
  <tr>
    <th style="background-color: #f2f2f2;">Date</th>
    <th style="background-color: #f2f2f2;">Price</th>
    <th style="background-color: #f2f2f2;">Open</th>
    <th style="background-color: #f2f2f2;">High</th>
    <th style="background-color: #f2f2f2;">Low</th>
    <th style="background-color: #f2f2f2;">Vol.</th>
    <th style="background-color: #f2f2f2;">Change %</th>
  </tr>
  <tr>
    <td>02/21/2025</td>
    <td>0.246771</td>
    <td>0.254737</td>
    <td>0.261063</td>
    <td>0.246746</td>
    <td style="text-align: center;">1.31B</td>
    <td style="text-align: center;">-3.13%</td>
  </tr>
  <tr>
    <td>02/20/2025</td>
    <td>0.254740</td>
    <td>0.254866</td>
    <td>0.257420</td>
    <td>0.250098</td>
    <td style="text-align: center;">1.06B</td>
    <td style="text-align: center;">-0.05%</td>
  </tr>
  <tr>
    <td>02/19/2025</td>
    <td>0.254876</td>
    <td>0.251209</td>
    <td>0.255351</td>
    <td>0.248996</td>
    <td style="text-align: center;">913.84M</td>
    <td style="text-align: center;">1.42%</td>
  </tr>
  <tr>
    <td>02/18/2025</td>
    <td>0.251300</td>
    <td>0.258189</td>
    <td>0.259660</td>
    <td>0.242390</td>
    <td style="text-align: center;">1.55B</td>
    <td style="text-align: center;">-2.67%</td>
  </tr>
  <tr>
    <td>02/17/2025</td>
    <td>0.258182</td>
    <td>0.265716</td>
    <td>0.268627</td>
    <td>0.254015</td>
    <td style="text-align: center;">1.26B</td>
    <td style="text-align: center;">-2.84%</td>
  </tr>
  <tr>
    <td>02/16/2025</td>
    <td>0.265728</td>
    <td>0.271776</td>
    <td>0.274075</td>
    <td>0.263873</td>
    <td style="text-align: center;">842.34M</td>
    <td style="text-align: center;">-2.23%</td>
  </tr>
  <tr>
    <td>02/15/2025</td>
    <td>0.271794</td>
    <td>0.271835</td>
    <td>0.282934</td>
    <td>0.268399</td>
    <td style="text-align: center;">1.50B</td>
    <td style="text-align: center;">-0.01%</td>
  </tr>
</table>





## Challenges Encountered 

1. Issue to Access Influential Comments: The first issue that we faced was that Reddit has numerous common users whose comments might not be as influential in their effect on the market as those of professionals or influential figures on sites such as Twitter. Thus, some of the comments that we fetched were not very insightful or informative regarding sentiment analysis for Dogecoin price changes. This restriction became increasingly obvious as we examined the data.
<br>

2. API for Reddit has a restriction in that you can only do 300 requests at once. Because we wanted much more data, we had to execute many requests to get sufficient information. This limitation slowed down our data collection, but we had to do it in order to ensure we had enough information for some good analysis.
<br>
3. Observing a Weak Relationship Between Sentiment and Price: Once we executed the sentiment analysis using TextBlob, we merged the sentiment scores with the historical price data of Dogecoin. 

![overtimetrend]({static}/images/group-Function-Words_01_overtimetrend.png)

However, we found that the sentiment scores were not strongly correlated with Dogecoin's price movements. The simple polarity-based sentiment analysis did not seem to pick up on how the discussion topics in the subreddit influenced the trends in the market. This result brought us to the conclusion that our current sentiment analysis approach is too simplistic.

## Looking for a Better Model: DeepSeek for Emotional Understanding

Because rudimentary sentiment analysis has its limitations, we sought to explore more advanced tools. DeepSeek is one of them: it's a free API offering rich emotional analysis that we can use to analyze comments and rate their emotional tone of text into feelings like happiness, sadness, anger, and so on. For example, we can simply start a request of" rating this comment about how surprised they are from 0-10". We believed this would help us capture the full spectrum of sentiment on Reddit postings and ascertain more precisely how emotion might affect the price of Dogecoin. So, we're thinking of analyzing daily comments from a range of different emotional angles, with a score from 0 to 10 to indicate how strong each emotion is in DeepSeek. With that data, we might be able to apply a regression model to see if there is any relationship between those emotional environments and how Dogecoin's price moves.

We're not really looking to devise a profitable trading strategy, but we think this is a fascinating research project into how social media sentiment relates to cryptocurrency prices.

## Conclusion

Despite these difficulties, we have gained a great deal of insight into the nuances of sentiment analysis, especially regarding social media data. Our next steps involve leveraging more advanced tools like DeepSeek in order to further refine our sentiment analysis and investigate potential correlations with price movement. Though the current goal is research-oriented and exploration in nature, we are excited to continue to investigate how sentiment from platforms like Reddit may influence market trends, especially in the volatile world of cryptocurrency. We're aiming to dive deeper and help everyone better understand how social media moods can impact market movements.
