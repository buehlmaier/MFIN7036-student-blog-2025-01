---
Title:Data Collection in Financial NLP：Progress, Challenges, and Next Steps (By Group PuzzleBOT)
Date: 2025-02-23 23:59
Category: Reflective Report
Tags: Group 5 - PuzzleBOT
---

## **Abstract**

This blog post presents our progress in data collection for our project, which investigates the feasibility of using NLP to extract sentiment-related words from conference calls and analyze their impact on stock returns. In order to fulfill the research objectives, at this stage we have completed a literature review and identified relevant methods and variables from previous studies. Based on these findings, we obtained data from WRDS database and stored them as CSV file pattern for further analysis. However, the data collection process posed several challenges, including database selection adjustments, data inconsistencies, and time alignment issues. We discussed these obstacles and resolved some of them. Upon completion of data preprocessing, we anticipate additional challenges in data cleaning and sentiment extraction, which will be addressed in the next phase of the study.

![cover]({static}/images/PuzzleBOT_01_cover.png){: style="width: 100%; height: auto;"}


## **Data Collection**

Capital IQ provides professional conference calls & earnings call transcription services, outputting well-structured, high-quality transcripts text data, and WRDS collects these transcripts in batches. As such, we gather data from WRDS.

WRDS provides a database access API, and we downloaded some sample data based on our needs.

```python
import wrds

db = wrds.Connection()

query1 = "SELECT * FROM ciq_common.wrds_ticker LIMIT 5"
wrds_ticker = db.raw_sql(query1)

query2 = "SELECT * FROM ciq_transcripts.wrds_transcript_detail LIMIT 5"
wrds_transcript_detail = db.raw_sql(query2)

query3 = "SELECT * FROM ciq_transcripts.ciqtranscriptcomponent LIMIT 5"
ciqtranscriptcomponent = db.raw_sql(query3)

db.close()
```

Table wrds_ticker is used to merge Capital IQ's internal companyid and stock ticker. Table wrds_transcript_detail contains the basic information of each transcript, including companyid, release time, meeting type and corresponding transcriptid. Table ciqtranscriptcomponent stores the text content information of transcripts.

We need to download three tables and merge them according to the relationship. The data dictionaries and data samples of the three tables are as follows:

### 1. wrds_ticker

| Variable Name | Type    | Length | Description   |
| ------------- | ------- | ------ | ------------- |
| companyid     | Decimal | 11     | Company ID    |
| companyname   | Char    | 400    | Company Name  |
| enddate       | Date    |        | End Date      |
| startdate     | Date    |        | Start Date    |
| ticker        | Char    | 15     | Ticker Symbol |

![wrds_ticker]({static}/images/PuzzleBOT_01_wrds_ticker.png)


### 2. wrds_transcript_detail

| Variable Name                   | Type    | Length | Description                       |
| ------------------------------- | ------- | ------ | --------------------------------- |
| audiolengthsec                  | Decimal | 11     | Audio Length in Seconds           |
| companyid                       | Decimal | 11     | Company ID                        |
| companyname                     | Char    | 400    | Company Name                      |
| headline                        | Char    | 381    | Event Headline                    |
| keydeveventtypeid               | Decimal | 11     | Key Development/Event Type ID     |
| keydeveventtypename             | Char    | 400    | Key Development/Event Type Name   |
| keydevid                        | Decimal | 11     | Key Dev ID                        |
| mostimportantdateut             | Date    |        | Most Important Date UTC           |
| mostimportanttimeutc            | Time    | 53     | Most Important Time UTC           |
| transcriptcollectiontypeid      | Int     | 32     | Transcript Collection Type ID     |
| transcriptcollectiontype-name   | Char    | 200    | Transcript Collection Type Name   |
| transcriptcreationdate_utc      | Date    |        | Transcript Creation Date UTC      |
| transcriptcreationtime_utc      | Time    | 53     | Transcript Creation Time UTC      |
| transcriptid                    | Decimal | 11     | Transcript ID                     |
| transcriptpresentation-typeid   | Int     | 32     | Transcript Presentation Type ID   |
| transcriptpresentation-typename | Char    | 200    | Transcript Presentation Type Name |

![wrds_transcript_detail]({static}/images/PuzzleBOT_01_wrds_transcript_detail.png)


### 3. ciqtranscriptcomponent

| Variable Name              | Type | Length | Description |
| -------------------------- | ---- | ------ | ----------- |
| componentorder             | Int  | 16     | None        |
| componenttext              | Char |        | None        |
| transcriptcomponentid      | Int  | 32     | None        |
| transcriptcomponenttype-id | Int  | 16     | None        |
| transcriptid               | Int  | 32     | None        |
| transcriptpersonid         | Int  | 32     | None        |

![ciqtranscriptcomponent]({static}/images/PuzzleBOT_01_ciqtranscriptcomponent.png)

We query a transcript text sample and found that the text data is well-structured.

```python
print(ciqtranscriptcomponent.loc[0, 'componenttext'])
```

![text_data_template]({static}/images/PuzzleBOT_01_text_data_template.png)


## **Problems solved**

### 1. Massive Workload when Downloading Transpricts from Capital IQ

Capital IQ provides professional conference calls & earnings call transcription services, outputting well-structured, high-quality transcripts text data. However, we can only download a single quarter's transcript for a single company at a time.

We looked at GitHub, Hugging Face, Kaggle and other third-party datasets, but these were not used in the end because there were worries about the data quality and how difficult it would be to process the data later. In the end, we found that there are those transcripts inside Capital IQ collected and organised on WRDS.

### 2. Slow Data Downloads and Inefficient Calls

When working with large datasets in WRDS, retrieving all data at once can be inefficient and memory-intensive. A better approach is to fetch data in chunks using SQL’s `LIMIT` and `OFFSET`. This ensures efficient processing and avoids overloading memory.

**How It Works**

1. Use a Loop:
   - Fetch data in chunks with `LIMIT`.
   - Increment `OFFSET` to avoid duplicates.
   - Stop when no more data is returned.
2. Combine All Chunks into a single DataFrame for analysis.

**Python Implementation**
```python
import wrds
import pandas as pd

db = wrds.Connection()
chunk_size = 10000
offset = 0
data = []

while True:
    query = f"SELECT * FROM ciq_transcripts.ciqtranscriptcomponent LIMIT {chunk_size} OFFSET {offset}"
    df = db.raw_sql(query)
    if df.empty:
        break
    data.append(df)
    offset += chunk_size

final_data = pd.concat(data)
db.close()
```

## **Key Points to Note**

### 1. Noise Issues in Data Cleaning and Preprocessing

When working with WRDS transcript data, it is important to recognize potential noise that could impact research accuracy. Although WRDS transcript data is of high quality, it includes transcripts from global markets, with North American companies accounting for only 55%. Additionally, the dataset is not limited to earnings calls and conference calls but also includes other types of events, such as M&A calls, which may introduce unintended noise. Moreover, each transcript contains multiple edited versions, which could lead to inconsistencies if not properly handled during preprocessing.

To ensure data reliability, it is necessary to clearly define the research sample in advance, repeatedly verify the meanings of variables, and design logical filtering criteria to exclude irrelevant data. Additionally, care should be taken to avoid look-ahead bias to maintain the validity of the analysis.

### 2. Time Matching Issues Between Earnings Calls and Financial Market Data

The second challenge we face when analyzing both earnings call data and financial market data is aligning the two across different companies. Sometimes, earnings calls are released after the end of a fiscal quarter or even in a different fiscal year, which leads to timing mismatches. This inconsistency makes it difficult to match earnings call data with financial market data, affecting the accuracy of our analysis. Ensuring precise timing between the two datasets is essential for obtaining reliable results.

To address this, we plan to establish a time window—typically a few days before and after the earnings call release. By focusing on stock price fluctuations within this window, we can reduce the biases introduced by timing mismatches. This approach should help better align market reactions with the content of the earnings calls.

## **Next Steps**

- First, we will clean the dataset by handling missing values, removing inconsistencies, and ensuring proper formatting. 

- Next, we will extract sentiment-related words from the conference call transcripts using NLP techniques. 

- After extraction, we will classify sentiment words into categories such as positive, negative, or neutral, potentially leveraging existing sentiment dictionaries or machine learning models. 

- Finally, we will conduct regression analysis to examine the relationship between extracted sentiment and stock returns.
