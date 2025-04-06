---
Title: Data Cleaning Process (by “Group 9”)
Date: 2025-02-23 18:12
Category: Progress Report
Tags: Group FIN-TANSTIC WIZARDS
---
# Data Cleaning Process: Transforming Unstructured Text into Structured Data

In our project, data cleaning is a crucial step to ensure the accuracy and reliability of downstream analyses. Our raw data is sourced from web-scraped posts and comments, which often contain noise such as special characters, links, and formatting artifacts. These issues increase processing complexity and can interfere with natural language processing (NLP) or machine learning (ML) models. Therefore, we designed a comprehensive text-cleaning function, `clean_text`, to remove irrelevant content while preserving meaningful information.

This blog post details our data-cleaning process, explaining each step's purpose and implementation with corresponding code snippets. All code is self-contained, allowing readers to test and verify results. We also discuss two major challenges encountered and their solutions.

## Overview of the Data Cleaning Process

Our goal is to transform messy raw text into a clean, structured format by performing the following steps:

1. Handle encoding artifacts and excessive whitespace.
2. Fix escape characters.
3. Remove non-standard characters (e.g., non-ASCII symbols).
4. Delete links, addresses, and sensitive data.
5. Eliminate formatted tables, dates, and placeholders.
6. Normalize text formatting.

Each step is detailed below, with explanations and corresponding code implementations.

---

## Detailed Cleaning Steps

### 1. Handle Encoding Artifacts and Excessive Whitespace

Raw text may contain invisible encoding artifacts (e.g., `\xa0`) and unnecessary whitespace characters.

```python

def clean_text(text):
    # Replace encoding artifacts and whitespace
    text = text.replace('\xa0', ' ').replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### 2. Fix Escape Characters

Fixing improperly escaped characters ensures text consistency.

```python

def clean_text(text):
    # Restore escape characters
    text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    text = text.replace('\\', '')  # Remove extra backslashes
    return text
```

### 3. Remove Non-Standard Characters

Deleting non-ASCII characters standardizes the dataset.

```python

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'\\x[0-9A-Fa-f]{2}', '', text)  # Remove hex escapes
    text = re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)  # Remove Unicode escapes
    return text
```

### 4. Delete Links and Sensitive Information

URLs, blockchain addresses, and private keys should be removed.

```python

def clean_text(text):
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = re.sub(r'0x[0-9a-fA-F]+', '', text)  # Remove hexadecimal addresses
    text = re.sub(r'1[1-9A-HJ-NP-Za-km-z]{25,34}', '', text)  # Remove Bitcoin addresses
    text = re.sub(r'[0-9a-fA-F]{64}', '', text)  # Remove potential private keys
    return text
```

### 5. Remove Tables, Dates, and Placeholders

Structured text artifacts such as tables and timestamps must be eliminated.

```python

def clean_text(text):
    text = re.sub(r'\|.*?\|.*?\|.*?\|.*?\|.*?\|', '', text)  # Remove tables
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text)  # Remove dates
    text = re.sub(r'____-__-__', '', text)  # Remove placeholders
    return text
```

### 6. Normalize Text Formatting

Final cleanup ensures consistent spacing and structure.

```python

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

---

## Challenges and Solutions

### Challenge 1: Handling Quoted Content

**Issue:** Extracting the original comment while removing quoted text was challenging.

| Reply Content |
|---------------|
| Quote from: franky1 on February 19, 2025, 08:05:22 PMc. you cant steal digital art ..its digital. .. copy & paste of the image is not theft But we can steal an NFT, Sir. Quote from: JollyGood on February 20, 2025, 02:07:05 PM I did not like it. What a pity, maybe you will like one of my next stories. Quote from: JollyGood on February 20, 2025, 02:07:05 PM How many accounts do you operate in the forum? I'm only operating my account, Sir. |

*This is the code we initially used to remove the quoted content, but later we found issues with the data, so we abandoned this version.*

```python
import re
import pandas as pd

def remove_quoted_text_from_reply(row, df):
    """
    Remove quoted text from the reply by finding the original referenced reply based on 
    Reply Author and Reply Date.
    :param row: Current row data
    :param df: DataFrame containing all comments
    :return: Cleaned comment content
    """
    # Ensure the Reply Content is of string type
    reply_content = str(row['Reply Content'])

    # Optimize the regular expression to match 24-hour time format and ensure no letter follows AM/PM
    pattern = r"Quote from: (?P<author>.*?) on (?P<date>.*?) (?P<time>\d{2}:\d{2}:\d{2})(?=\s*PM|\s*AM|\b)"
    match = re.search(pattern, reply_content)

    if match:
        quoted_author = match.group('author')
        quoted_date = match.group('date')
        quoted_time = match.group('time')

        # Convert date format to match Reply Date format
        quoted_date = pd.to_datetime(quoted_date, errors='coerce').strftime('%Y/%m/%d')  # Convert to a uniform date format
        
        # Convert time to 24-hour format
        quoted_time_24hr = convert_to_24hr_format(f"{quoted_time} PM")  # If there is AM/PM, convert directly

        # Print the extracted time and date for debugging purposes
        print(f"Quoted Author: {quoted_author}")
        print(f"Quoted Date: {quoted_date}")
        print(f"Quoted Time: {quoted_time_24hr}")

        # Ensure Reply Date format matches the extracted date and time
        df['Reply Date'] = pd.to_datetime(df['Reply Date'], errors='coerce').dt.strftime('%Y/%m/%d %H:%M:%S')  # Convert Reply Date to a unified date-time format
        
        # If Reply Date doesn't have seconds, add seconds
        df['Reply Date'] = df['Reply Date'].apply(lambda x: x if len(x.split()[1]) == 8 else x + ":00")

        quoted_reply = df[(df['Reply Author'] == quoted_author) & 
                           (df['Reply Date'].str.contains(quoted_date)) & 
                           (df['Reply Date'].str.contains(quoted_time_24hr))]

        # Print debug info to check if quoted_reply matches the correct record
        print(f"Quoted Reply: {quoted_reply}")

        if not quoted_reply.empty:
            quoted_content = quoted_reply['Reply Content'].iloc[0]
            # Remove quoted portion
            cleaned_text = reply_content.replace(f"Quote from: {quoted_author} on {quoted_date} {quoted_time}", "")
            cleaned_text = cleaned_text.replace(quoted_content, "")
            return cleaned_text.strip()

    return row['Reply Content']  # If no quoted content, return original content

# Process each comment to ensure the Reply Content is converted to string type
df['clean Content'] = df.apply(lambda row: remove_quoted_text_from_reply(row, df), axis=1)

# Set pandas display option to ensure full display of text content
pd.set_option('display.max_colwidth', None)  # Set to None to display full text

# Print the results
print(df[['Reply Author', 'Reply Date', 'Reply Content', 'clean Content']])
```
**Solution:** Initially, we attempted to resolve this by tracking comment timestamps and user IDs to find the referenced comments and then remove quoted text. However, since each post contained numerous comments, we did not scrape the entire discussion thread, making it impossible to match all quoted content. Therefore, we adopted an alternative approach: modifying our web scraping logic to exclude quoted content and only extract the main body of each comment.The detailed code can be viewed directly from the previous blog's data scraping section.

### Challenge 2: Preserving Useful Symbols

**Issue:** Removing all non-ASCII characters risked deleting meaningful symbols like emojis.

**Solution:** Instead of eliminating all special characters, we adjusted the regex to retain common emojis and symbols relevant to our dataset (e.g., 😊, 👍, ©, ™). This ensures that meaningful symbols are preserved while still eliminating unwanted noise.

---

## Conclusion

This structured approach to data cleaning ensures that our text is free of irrelevant information while preserving key insights for analysis. The `clean_text` function effectively removes unnecessary elements, standardizes formatting, and enhances the dataset’s usability for NLP and ML applications.


---

## 完整代码
```python
import re

def clean_text(text):
    
    # 删除乱码和多余空格
    text = text.replace('\xa0', ' ').replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 修复转义字符
    text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    text = text.replace('\\', '')
    
    # 删除非标准字符（可根据需求保留 emoji）
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\\x[0-9A-Fa-f]{2}', '', text)
    text = re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)
    
    # 删除链接和敏感信息
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'0x[0-9a-fA-F]+', '', text)
    text = re.sub(r'1[1-9A-HJ-NP-Za-km-z]{25,34}', '', text)
    text = re.sub(r'[0-9a-fA-F]{64}', '', text)
    
    # 删除表格、日期和占位符
    text = re.sub(r'\|.*?\|.*?\|.*?\|.*?\|.*?\|', '', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'____-__-__', '', text)
    text = re.sub(r'========================== U N K N O W N =========================', '', text)
    
    # 规范化文本
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


```
