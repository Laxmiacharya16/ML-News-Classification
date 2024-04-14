import pandas as pd
from bs4 import BeautifulSoup
import requests
import pickle



## url of the malaymail website
url = "https://www.malaymail.com/morearticles/malaysia?pgno=1"

# Initialize lists to store data
data_content_titles = []
timestamps = []
content_texts = []

# creating loop for the different page URL
for i in range(1, 6):
    page_url = f"{url}&pgno={i}"
    print("Page URL:", page_url)
    r = requests.get(page_url)
    soup = BeautifulSoup(r.text, "html.parser")

    article_urls = []

    # Find all div tags with class "col-md-3 article-item"
    div_tags = soup.find_all("div", class_="col-md-3 article-item")

    # Iterate through each div tag
    for div_tag in div_tags:
        # Find the h2 tag with class "article-title" inside each div tag
        h2_tag = div_tag.find("h2", class_="article-title")

        # Check if h2_tag is found
        if h2_tag:
            # Find the <a> tag inside the h2 tag
            a_tag = h2_tag.find("a")

            # Check if the <a> tag is found
            if a_tag:
                # Get the href attribute and append to the list
                href_value = a_tag.get("href")
                article_urls.append(href_value)

    # Now article_urls contains all the href values
    print(article_urls)

    # let's get soup from the URL
    for articleurl in article_urls:
        r = requests.get(articleurl)
        new_soup = BeautifulSoup(r.text, "html.parser")

        # title
        title = new_soup.find("h1", class_="article-title").get_text(strip=True)

        # date
        date = new_soup.find("div", class_="article-date").get_text(strip=True)

        # content
        article_body_div = new_soup.find("div", class_="article-body")

        # Check if the article_body_div is found
        if article_body_div:
            # Find all paragraphs (<p>) within the article_body_div
            paragraphs = article_body_div.find_all("p")

            # Extract text from each paragraph and concatenate
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            print("Title:", title)
            print("Date:", date)
            print("Content:", content)
            print("\n" + "="*50 + "\n")  # Separate articles for better visibility

            # Append data to lists
            data_content_titles.append(title)
            timestamps.append(date)
            content_texts.append(content)

        else:
            print(f"Article body not found for URL: {articleurl}")

# Create a DataFrame
malaymail_news = pd.DataFrame({
    "Title": data_content_titles,
    "Date": timestamps,
    "Content": content_texts
})

# Display the DataFrame
print(malaymail_news.head())

import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from io import BytesIO
from datetime import datetime

import boto3

## we have to collect only the latest news so for that we have to chANGE THE date format

# Convert the 'Date' column to datetime objects
malaymail_news['Date'] = pd.to_datetime(malaymail_news['Date'], format='%A, %d %b %Y %I:%M %p MYT')

# Format the 'Date' column as a new date string
malaymail_news['Date'] = malaymail_news['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Get the current date
current_date = datetime.now().strftime('%Y-%m-%d')

# Filter the DataFrame for the latest news
malaymail_latest_news = malaymail_news[malaymail_news['Date'].str.startswith(current_date)]

# AWS credentials
aws_access_key = 'AKIASUYK4WETMTSWA55T'
aws_secret_key = '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'

# S3 bucket details for raw data
bucket_name_raw = 'laxmicleandata'
object_key_raw = 'malaymail_rawdata_1'

# Convert DataFrame to CSV format
csv_buffer = BytesIO()
malaymail_latest_news.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

# Upload CSV data to S3 for raw data
s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
s3.upload_fileobj(csv_buffer, bucket_name_raw, object_key_raw)

print(f"Raw Dataframe uploaded to s3://{bucket_name_raw}/{object_key_raw}")

# Download the raw data file from S3
response_raw = s3.get_object(Bucket=bucket_name_raw, Key=object_key_raw)
content_raw = response_raw['Body'].read()

# Convert content to DataFrame (assuming it's a CSV file)
malaymail_raw = pd.read_csv(BytesIO(content_raw))

# Display the raw DataFrame
malaymail_raw

import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords resource
nltk.download('stopwords')

def preprocess_text(df, column_name):
   
    # Remove HTML tags
    def remove_html_tags(text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    df[column_name] = df[column_name].apply(lambda x: remove_html_tags(x))

    # Remove smilies
    def remove_smilies(text):
        smilies = [':)', ':(', ';)', '<3', ':/', ':D', ':P', ':-)']
        for smiley in smilies:
            text = text.replace(smiley, '')
        return text

    df[column_name] = df[column_name].apply(lambda x: remove_smilies(x))

    # Remove punctuation
    def remove_punctuation(text):
        no_punct = [words for words in text if words not in string.punctuation]
        words_wo_punct = ''.join(no_punct)
        return words_wo_punct

    df[column_name] = df[column_name].apply(lambda x: remove_punctuation(x))
    return df

# Preprocess the text in the 'Content' column without stemming and tokenizer
preprocessed_malaymail = preprocess_text(malaymail_raw, 'Content')

# S3 bucket details for cleaned data
bucket_name_clean = 'laxmicleandata'
object_key_clean = 'malaymailcleandata_1'

# Convert cleaned DataFrame to CSV format
csv_buffer_clean = BytesIO()
preprocessed_malaymail.to_csv(csv_buffer_clean, index=False)
csv_buffer_clean.seek(0)

# Upload CSV data to S3 for cleaned data
s3.upload_fileobj(csv_buffer_clean, bucket_name_clean, object_key_clean)

print(f"Cleaned Dataframe uploaded to s3://{bucket_name_clean}/{object_key_clean}")

# Download the cleaned data file from S3
response_clean = s3.get_object(Bucket=bucket_name_clean, Key=object_key_clean)
content_clean = response_clean['Body'].read()

# Convert content to DataFrame (assuming it's a CSV file)
cleanmalaymail_news = pd.read_csv(BytesIO(content_clean))

# Display the cleaned DataFrame
print(cleanmalaymail_news.head())

#####################################################################################################################
####################################################################################################################
########## Extracting the clean malaymail data from s3 bucket##################

import boto3
import pandas as pd
from io import BytesIO

# S3 bucket details
aws_access_key = 'AKIASUYK4WETMTSWA55T'
aws_secret_key = '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'
bucket_name_clean = 'laxmicleandata'
object_key_clean = 'malaymailcleandata_1'
csv_buffer_clean = BytesIO()
s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

response_clean = s3.get_object(Bucket=bucket_name_clean, Key=object_key_clean)
content_clean = response_clean['Body'].read()

# Convert content to DataFrame (assuming it's a CSV file)
cleanmalaymail_news = pd.read_csv(BytesIO(content_clean))

# Display the cleaned DataFrame
print(cleanmalaymail_news.head())


cleanmalaymail_news. to_csv()


############################ text summarization ###########################

###                model building #######################################

#################################################################################################################################
##################################################################################################################################


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = "google/pegasus-cnn_dailymail"
summarize_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
summary_model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")

def summarize_tex1(text):
    # Tokenize the input text and generate summaries using the Pegasus model
    inputs = summarize_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summary_model.generate(inputs, num_beams=6,max_length=150, min_length=50,length_penalty = 2.0)
    
    # Decode the summary
    summary1= summarize_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # Post-process the summary to remove the <n> characters and join all lines into a single paragraph
    summary_para = summary1.replace('<n>', ' ').strip()
    return summary_para
  
generated_summaries = cleanmalaymail_news['Content'].apply(summarize_tex1)

# Add the generated summaries to the DataFrame
cleanmalaymail_news['summary'] = generated_summaries


cleanmalaymail_news.to_csv('generated_summaries.csv', index=False)


cleanmalaymail_news['summary'][5]

# Saving the Pegasus model and tokenizer
pickle.dump(summarize_tokenizer, open('summarize_tokenizer.pkl', 'wb'))
pickle.dump(summary_model, open('summary_model.pkl', 'wb'))

# Optionally, you can save the cleanmalaymail_news dataframe with the appended generated summaries to a CSV file.
cleanmalaymail_news.to_csv("cleanmalaymail_news.csv", index=False)  # Save the dataframe as CSV file

import os
cwd = os.getcwd()




#####################        TEXT CLASSIFICATION       #############################

#########################################################################################################################3

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load the sentiment classification model and tokenizer
tokenizer_sentiment = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")

label_map = {
    0: "positive",
    1: "neutral",
    2: "negative"
}

import torch

def classify_sentiment(text, threshold=0.7):
    # Tokenize the text
    inputs = tokenizer_sentiment(text, return_tensors="pt", padding=True, truncation=True)

    # Apply the model
    outputs = model_sentiment(**inputs)

    # Interpret the output (e.g., get predicted label)
    predicted_label = outputs.logits.argmax().item()
    
    # Get the probabilities for each class
    probabilities = torch.softmax(outputs.logits, dim=1).tolist()[0]
    
    # Map the numeric label to sentiment category
    sentiment_category = label_map[predicted_label]
    
    # Check if the maximum probability is greater than the threshold
    is_hot_news = probabilities[predicted_label] >= threshold

    # Concatenate classification, predicted label, and probabilities into a single string
    classification_result = f"{sentiment_category}, {predicted_label}, {probabilities}"
    
    # Return the concatenated classification result and hot news flag
    return classification_result, "Hot news" if is_hot_news else "Not hot news"

# Apply the classify_sentiment function to each row of the 'Content' column
results = cleanmalaymail_news['Content'].apply(lambda x: classify_sentiment(x, threshold=0.7))

# Unpack the results into separate lists
classification_result, hot_news_flag = zip(*results)

# Add the classification result and hot news flag as new columns in the DataFrame
cleanmalaymail_news['Classification_Result'] = classification_result
cleanmalaymail_news['classification'] = hot_news_flag

# Print the DataFrame to see the added columns
print(cleanmalaymail_news)

pickle.dump(model_sentiment, open('model_sentiment.pkl', 'wb'))
pickle.dump(tokenizer_sentiment, open('tokenizer_sentiment.pkl', 'wb'))

###############################################################################################################################
#################################################################################################################################
####### ######################       topic modelling     ###################################

import re 
# train test split of the data
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(cleanmalaymail_news, test_size=0.2, random_state=42)

# Further split the test set into test and validation sets (50% test, 50% validation)
test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)

from transformers import AutoTokenizer, T5ForConditionalGeneration
topic_model_name = "JulesBelveze/t5-small-headline-generator"
topic_tokenizer = AutoTokenizer.from_pretrained("JulesBelveze/t5-small-headline-generator")
model = T5ForConditionalGeneration.from_pretrained("JulesBelveze/t5-small-headline-generator")


def generate_headline(article_text):
    # Tokenize the input text and generate headline using the model
    input_ids = topic_tokenizer(
    article_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=384
    )["input_ids"]
    output_ids = model.generate(
    input_ids=input_ids,
    max_length=84,
    no_repeat_ngram_size=2,
    num_beams=4
    )[0]
    summary = topic_tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False) 
    return summary

# Apply the function to the Content column of the train_data so that we can evaluate the model using rouge score

generated_headline = train_data['Content'].apply(generate_headline)

train_data['generated_headline'] = generated_headline

#### Evaluating the model ##################

#######    ROUGE     ###################
###################

############################################### evaluation in aggregate  #################33

from rouge_score import rouge_scorer

# Prepare reference and system-generated summaries
reference_summaries = train_data['Title'].tolist()
candidate_summaries = train_data['generated_headline'].tolist()

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Initialize variables to store aggregate scores
aggregate_scores = {metric: {'precision': 0, 'recall': 0, 'fmeasure': 0} for metric in ['rouge1', 'rouge2', 'rougeL']}

# Iterate over each pair of reference and system-generated summaries
for reference_summary, system_summary in zip(reference_summaries, candidate_summaries):
    # Calculate ROUGE scores
    scores = scorer.score(reference_summary, system_summary)
    
    # Aggregate scores
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        aggregate_scores[metric]['precision'] += scores[metric].precision
        aggregate_scores[metric]['recall'] += scores[metric].recall
        aggregate_scores[metric]['fmeasure'] += scores[metric].fmeasure

# Calculate average scores
num_samples = len(reference_summaries)
for metric in ['rouge1', 'rouge2', 'rougeL']:
    aggregate_scores[metric]['precision'] /= num_samples
    aggregate_scores[metric]['recall'] /= num_samples
    aggregate_scores[metric]['fmeasure'] /= num_samples

# Print aggregate scores
print("Aggregate ROUGE Scores:")
for metric, scores in aggregate_scores.items():
    print(f"{metric}: Precision={scores['precision']}, Recall={scores['recall']}, F-measure={scores['fmeasure']}")

# score obtained is normal so we will go with the same model in the whole data set
########################################################################33

##lets apply the above model for the whole dataset

# Apply the function to the Content column of the malaymail_data 

generated_headline = cleanmalaymail_news['Content'].apply(generate_headline)

cleanmalaymail_news['Headline'] = generated_headline

cleanmalaymail_news.to_csv('generated_headline.csv', index=False)

cleanmalaymail_news['Content'][2]

## saving the topic modelling model

pickle.dump(model, open('topic_model.pkl', 'wb'))
pickle.dump(topic_tokenizer, open('topic_tokenizer.pkl', 'wb'))


## saving results 
cleanmalaymail_news.to_csv('malaymail_articles.csv', index=False)
