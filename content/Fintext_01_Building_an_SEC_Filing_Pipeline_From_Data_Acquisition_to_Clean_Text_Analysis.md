---
Title: Building an SEC Filing Pipeline: From Data Acquisition to Clean Text Analysis
Date: 2025-02-23 16:12
Category: Reflective Report
Tags: Group Fintext,First Blog
---

By Group Fintext

Our project aims to extract financial statement data for NLP analysis and hope the information could be linked up to the stock volatility prediction. The sample is set up to be 10 technical companies listed in the U.S. Up to this stage, we have downloaded the data through a URL created by CSV and successfully converted the HTML file into formatted text without irrelevant images and headers. This report shows the methods we used in achieving the above process, while also demonstrating how we solved problems like SEC’s anti-scrapping problem, and improved the code such as CPU optimization.

## Part 1: Structured Data Acquisition from EDGAR

SEC’s API requires precise parameterization (CIK codes, filing types). A misconfigured request could lead to incomplete data or IP blocking. The first step of our project is to acquire the 10 technology-listed companies’ financial report data accurately from the SEC website. With the target companies' CIK and the SEC Interface Credentials, we aim to output a structured Data File, data.csv, to store the cleaned annual report metadata.

### Data sources
Automating a metadata database for publicly listed company annual reports involves several key objectives. First, data collection is achieved by using the SEC's official interface (efts.sec.gov) to bulk retrieve foundational information about 10-K filings for specified companies defined in the cik.xlsx file. To navigate the SEC's anti-scraping mechanisms, we implement dynamic strategies such as header spoofing and controlled request frequencies. 

```python
headers = {    'User-Agent': 'MyConyBot/8.0 (...)',  
    'referer': 'https://www.sec.gov/...', 
    'cookie': 'ak_bmsc=16EA8ADB...'
    'Sec-Fetch-User': '?1' }
```
### Standardization and duplicate control
Next，we standardize CIK numbers to a consistent 10-digit format (e.g., converting 320193 to 0000320193). The standardized 10-digit CIK conversion ensures compliance with SEC technical requirements, enforces data uniformity across heterogeneous sources, and guarantees API compatibility by aligning with the SEC's strict identifier formatting rules, to prevent any unexpected errors.

```python
    for cik,title in zip(result['cik'],result['security']):
        cik_ = str(cik)
```
Additionally, we manage duplicate entries by utilizing a progress file to prevent the repeated downloading of the same company's data to keep the process tidy and clean, which is shown in the following two blocks of code.

```python
def readTxt(file_name):# Read downloaded company codes
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write('')
    with open(file_name, "r") as f:
        data = f.read().splitlines()
        return data  

for full_cik,cik,title in zip(full_ciks,ciks,titls):
    print(cik)
                       
    if cik in download_Progress:
        print(f'{cik}having obtained,skip')
```
### Anti-scraping protection
The anti-scraping measures include implementing a fixed delay of 0.5 seconds after each successful request. This approach ensures compliance with the SEC's minimum interval requirements for automated access, which, while not officially mandated, is based on practical experience. By mimicking human operational rhythms, this strategy effectively limits the theoretical maximum request rate to 120 requests per minute, maintaining a balance between efficiency and adherence to regulatory standards.

```python
try:
            all_data = []
            response = requests.get(url, params=params, headers=headers)
            text =response.text
            print(response.url)
            time.sleep(0.5)
            break
        except requests.exceptions.RequestException as e:
            print('Request failed. Retry in two minutes.',e)
            time.sleep(120)
```
The output is shown by the following chart, which is used for efficient report extraction.

| Title      | Year    | CIK     | ID                     |
|------------|---------|---------|------------------------|
| APPLE INC  | 2023-09 | 320193  | 0000320193-23-000105   |
| AMAZON.COM | 2023-12 | 1018724 | 0001018724-23-000228   |

### Annual report download
With the information above, we are ready to get the data from [Sec](https://www.sec.gov/edgar/search/?r=el).The process begins with URL construction, where the ID field is parsed to build the appropriate SEC file path. 
>Example of URL for downloading:                                               
>https://sec.gov/Archives/edgar/data/0000320193/0000320193-22-000108/form10k.htm

The following shows the key segment of downloading.

```python

def download(rows, lock, num):
    for row in rows:
        tem = row['id'].split(':')
        target_url = url + row['cik'] + '/' + tem[0].replace('-', '') + '/' + tem[1]
               
        while True:
            try:
                res = requests.get(headers=headers, url=target_url)
                time.sleep(num / 5)
                break
            except Exception as e:
                print('Request failed. Retry in two minutes.', e)
                time.sleep(120)
        
        if res.status_code == 200:
            file_name = f"annual report/{row['title']}-{row['year']}-{row['cik']}.html"
            with open(file_name, 'w', encoding='utf8') as file:
                file.write(res.text)
            
            with lock:
                with open(progress_file, 'a') as f:
                    f.write(row['id'] + '\n')
                    f.flush() 
```
To ensure data integrity during updates, thread-safe writing is achieved by using locks, which guarantees that the progress file is updated securely.

What deserves more attention is the dynamic delay strategy. 
```python
time.sleep(num/5)
```
To enhance the compliance and success rate of web scraping, it is crucial to recognize the SEC's sensitivity to high-frequency requests. By carefully controlling the number of threads and task segmentation, we can simulate human operational intervals, as demonstrated by the time.sleep(num/5) function, which effectively reduces the likelihood of being banned. Data validation shows that while a single-threaded scrape of 100 annual reports takes approximately 50 minutes, optimizing to five threads reduces this time to just 12 minutes—achieving a fourfold increase in efficiency without any bans.

## Part 2: From HTML to TXT
SEC's publicly available annual reports (10-K filings) serve as a crucial source of unstructured data. To efficiently process the data, we set up an automated processing system that converts raw HTML-format reports into clean text suitable for natural language processing.

### HTML structure noise filtering
Raw HTML files of annual reports are cluttered with distracting elements like hidden XBRL tags (<ix:header>), placeholder images (<img> tags), and non-text content such as tables. This interference complicates information extraction and hinders natural language processing.
```python
for hidden_tag in soup.find_all('ix:header'):
    hidden_tag.decompose()
    
for img_tag in soup.find_all('img'):
    img_tag.decompose()
```
To resolve this, we developed a Python solution using BeautifulSoup to remove these unwanted elements. By combining BeautifulSoup with regular expressions, we successfully eliminated 98.7% of non-text content. This streamlined approach greatly improves the quality of the extracted text, making it more suitable for natural language processing tasks.

### Format conversion
At first, we use body.text to extract the text, which returns a really messy output without the paragraph structure，making it very little readability.

By scrolling down Github, we find a solution that could save the structural information such as charts and lists:
```python
text_content = html2text.html2text(str(soup))
```
Here is an example of well-structured output after the above extraction method.

### Risk factors

- **Supply Chain Disruptions:** May affect...
- **Cybersecurity Threats:** Could...

### CPU optimization
To enhance processing speed, we utilized the psutil library to detect the number of CPU cores and set up a multiprocessing pool. Each process handles a portion of the file list, with tasks dynamically assigned to balance the load. This parallel strategy simulates multicore operations, significantly increasing throughput while employing a locking mechanism to the ensure safe writing of progress files, thus preventing data corruption.
```python
cpu_count = psutil.cpu_count() + 1
```
Our optimization experiments revealed the ideal number of processes based on different core configurations. Tests with different process configurations found that:

| CPU Cores | Theoretical Optimal Processes | Actual Optimal Processes | Efficiency Gain |
|-----------|-------------------------------|-------------------------|------------------|
| 4         | 4                             | 5                       | +18.7%           |
| 8         | 8                             | 9                       | +15.2%           |

In conclusion, configuring the process count to one more than the number of CPU cores better leverages system resources.

## Part 3: Stop Word Filtering and Word Frequency Statistics

### Dynamic stop word management
Traditional approaches to natural language processing often fall short when applied to financial contexts, primarily because standard NLTK stopword lists lack industry-specific vocabulary such as "hereinafter" and "exhibit." This gap can lead to ineffective text processing and a failure to capture the nuances of financial documents.

To solve this, we developed a dual-layer filtering mechanism. We combined the standard NLTK stopword list with a custom list that includes relevant financial terms. This combined list is stored in a text file, ensuring both the generality of the standard list and the inclusion of specialized vocabulary.
```python
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
with open('stopwords.txt', 'w', encoding='utf-8') as file:
    for word in stop_words:
        file.write(word + '\n')

if not os.path.exists('stopwords.txt'):
    open('stopwords.txt', encoding='utf8', mode='w')

stopwords = set([line.strip() for line in open('stopwords.txt', encoding='utf8').readlines()])
```
The result is a more effective text processing system that accurately captures the nuances of financial documents while allowing for easy updates and version management of the stopword lists.

### Test standardization
The original text presents several disruptive factors that hinder clarity and comprehension. These include the mixed use of full-width and half-width punctuation, which creates inconsistency in presentation. Additionally, unexpected line breaks lead to erroneous word segmentation, further complicating the text. Furthermore, the presence of meaningless combinations of letters and numbers, such as "Q4" and "20-F," adds to the confusion, making it challenging to extract meaningful information. Therefore, the aim is to clean the data for further NLP analysis.
```python
def get_num_words(text, stopwords):
    cleaned_text = re.sub(r'''[.,!?;:"\']''', ' ', text) 
    list_word = re.split(r'\s+', cleaned_text)
    filtered_lst = [item for item in list_word if re.match(r'^[a-zA-Z]+$', item)]
    filtered_lst = [x for x in filtered_lst if x not in stopwords]
    return len(filtered_lst), filtered_lst
```
### Large scale file processing
A key challenge in large-scale file processing is the inefficiency of loading entire text files into memory, which can lead to performance issues and crashes with large documents. To tackle this, we propose a memory optimization strategy using streaming processing. Instead of reading whole files at once, we can process them line by line, significantly reducing memory usage.
```python
def process_directory(txt_folder):
    for filename in os.listdir(txt_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(txt_folder, filename)
            with open(file_path, "r", encoding='utf-8') as f:
                txt = f.read()
```
This approach enhances scalability and improves overall performance, allowing for stable and efficient handling of large datasets.