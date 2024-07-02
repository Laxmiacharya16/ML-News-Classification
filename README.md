# ML-News-Classification
README
Overview

This script performs several tasks on articles from the Malay Mail website. It scrapes article content, preprocesses the text, performs text summarization and sentiment classification, and generates topic headlines. The processed data is saved to AWS S3 and local files.
Steps Performed

    Web Scraping:
        Scrapes articles from the Malay Mail website.
        Extracts titles, dates, and content of the articles.

    Data Preprocessing:
        Removes HTML tags, smilies, and punctuation from the article content.

    Data Storage:
        Saves raw and cleaned data to AWS S3 buckets.
        Loads data back from S3 for further processing.

    Text Summarization:
        Uses the Pegasus model to summarize the articles.

    Sentiment Classification:
        Uses the DistilBERT model to classify the sentiment of the articles.
        Labels articles as "Hot news" based on a probability threshold.

    Topic Modeling:
        Uses the T5 model to generate topic headlines for the articles.

    Evaluation:
        Evaluates the generated headlines using the ROUGE metric.

    Results Storage:
        Saves the processed data with summaries and headlines to CSV files.

Prerequisites

    Python 3.x
    Required Python packages:
        pandas
        BeautifulSoup4
        requests
        pickle
        boto3
        nltk
        transformers
        torch
        rouge_score

AWS Credentials

You need AWS credentials with S3 access to upload and download files from S3.

python

aws_access_key = 'YOUR_AWS_ACCESS_KEY'
aws_secret_key = 'YOUR_AWS_SECRET_KEY'

Instructions

    Web Scraping:
        Modify the url variable to change the target website if needed.
        The script scrapes articles from the first 5 pages.

    Data Preprocessing:
        Ensure the correct format for the 'Date' column is specified.

    AWS S3 Storage:
        Specify the S3 bucket names and object keys for raw and cleaned data.

    Text Summarization:
        Modify the summarization function summarize_tex1 if you want to use a different model or parameters.

    Sentiment Classification:
        Modify the classify_sentiment function if you want to use a different model or parameters.

    Topic Modeling:
        Modify the generate_headline function if you want to use a different model or parameters.

    Run the Script:
        Execute the script to perform all tasks sequentially.
        The processed data will be saved locally as CSV files.

Example Usage

To run the script, execute the following command in your terminal:

bash

python script_name.py

Replace script_name.py with the actual name of your script file.
Output

    malaymail_articles.csv: Final processed data with summaries and headlines.
    generated_summaries.csv: Summaries generated for each article.
    generated_headline.csv: Headlines generated for each article.

Notes

    Ensure you have the required permissions to access the S3 buckets.
    The script uses the google/pegasus-cnn_dailymail, lxyuan/distilbert-base-multilingual-cased-sentiments-student, and JulesBelveze/t5-small-headline-generator models from the Hugging Face Transformers library.
    The script saves the summarization and topic modeling models using pickle for future use.
