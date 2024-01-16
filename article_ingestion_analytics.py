############### Libraries ###############

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import os
import time
import json

import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import boto3
from dateutil.relativedelta import relativedelta

############### Config & Data Ingestion ###############
s3_client = boto3.client('s3')
testing = False

def generate_synthetic_data(num_entries):
    current_timestamp = round(time.time())
    current_datetime = datetime.fromtimestamp(current_timestamp)

    three_months_ago = current_datetime - relativedelta(months=3)
    three_months_ago_timestamp = int(time.mktime(three_months_ago.timetuple()))

    three_months_future = current_datetime + relativedelta(months=3)
    three_months_future_timestamp = int(time.mktime(three_months_future.timetuple()))

    synthetic_data = []
    for _ in range(num_entries):
        data_point = {
            "articlesRemoved": random.randint(100, 600),       # Random integer between 100 and 600
            "rssFeeds": random.randint(1400, 1600),            # Random integer between 1400 and 1600
            "articlesProcessed": random.randint(50, 600),      # Random integer between 50 and 600
            "articleAmount": random.randint(29000, 32000),     # Random integer between 29000 and 32000
            "time": random.randint(three_months_ago_timestamp, three_months_future_timestamp) # Random time within 3 months
        }
        synthetic_data.append(data_point)
    return synthetic_data

main_root = 'analysis'
# Example usage
if testing:
    synthetic_data = generate_synthetic_data(200)  # Generate 10 synthetic data points
    data = synthetic_data
else:
    bucket_name = 'production-logs-tokenized-toast'
    file_key = 'article_analytics.json'
    # Fetch the file from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    content = response['Body'].read().decode('utf-8')
    data = json.loads(content)

############### Analysis Functions ###############
def analyze_seasonality(df:pd.DataFrame, selected_column:str, datetime_col:str="datetime"):
    root = f'{main_root}/{selected_column}-seasonality/'
    try:
        os.mkdir(root)
    except:
         pass

    # Ensure datetime_col is in datetime format
    # df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Extract day of the week and hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['hour'] = df[datetime_col].dt.hour

    # Check for daily seasonality
    daily_avg = df.groupby('day_of_week')[selected_column].mean()
    decompose_daily = seasonal_decompose(daily_avg, model='additive', period=1)
    decompose_daily.plot()
    plt.title('Daily Seasonality')
    plt.savefig(root+'daily_seasonality.png')

    # Check for hourly seasonality
    hourly_avg = df.groupby('hour')[selected_column].mean()
    decompose_hourly = seasonal_decompose(hourly_avg, model='additive', period=1)
    decompose_hourly.plot()
    plt.title('Hourly Seasonality')
    plt.savefig(root+'hourly_seasonality.png')

def analyze_normality(df:pd.DataFrame, selected_column:str):
    # Calculate difference
    root = f'{main_root}/{selected_column}-normality/'
    try:
        os.mkdir(root)
    except:
         pass


    # Mean and Standard Deviation
    mean_diff = df[selected_column].mean()
    std_diff = df[selected_column].std()

    # Normality Test (Shapiro-Wilk Test)
    shapiro_test = stats.shapiro(df[selected_column])
    normality_passed = shapiro_test.pvalue > 0.05  # Assuming alpha = 0.05

    # Unit Root Test (Augmented Dickey-Fuller Test)
    adf_test = adfuller(df[selected_column])

    # Autocorrelation Function Plot
    fig, ax = plt.subplots()
    sm.graphics.tsa.plot_acf(df[selected_column], lags=40, ax=ax)
    plt.savefig(root+'autocorrelation_plot.png')
    plt.close()

    # Binned Plot with Normal Distribution Curve
    sns.histplot(df[selected_column], kde=True, bins=30)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean_diff, std_diff)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit Results: mu = %.2f,  std = %.2f" % (mean_diff, std_diff)
    plt.title(title)
    plt.savefig(root+'binned_plot.png')
    plt.close()

    # Save results to a text file
    with open(root+'analysis_results.txt', 'w') as file:
        file.write(f"Mean Difference: {mean_diff}\n")
        file.write(f"Standard Deviation: {std_diff}\n")
        file.write(f"Normality Test Passed: {normality_passed}\n")
        file.write(f"Unit Root Test (ADF): {adf_test}\n")

    return root+'analysis_results.txt', root+'autocorrelation_plot.png', root+'binned_plot.png'

def save_plot(x, y, title, filename):
    root = f'{main_root}/charts/'
    try:
        os.mkdir(root)
    except Exception as e:
         print(e)
         pass


    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(title)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(root+filename+'.png')



############### Main ###############
# Convert JSON data to a DataFrame
df = pd.DataFrame(data)

if len(df) > 40:
    # Over Time
    df['datetime'] = pd.to_datetime(df['time'], unit='s')

    save_plot(df['datetime'], df['articleAmount'], 'Article Amount Over Time', 'article_amount_over_time')
    save_plot(df['datetime'], df['rssFeeds'], 'RSS Feeds Over Time', 'rss_feeds_over_time')
    save_plot(df['datetime'], df['articlesRemoved'], 'Articles Removed Over Time', 'articles_removed_over_time')
    save_plot(df['datetime'], df['articlesProcessed'], 'Articles Processed Over Time', 'articles_processed_over_time')
    df['processedMinusRemoved'] = df['articlesProcessed'] - df['articlesRemoved']
    save_plot(df['datetime'], df['processedMinusRemoved'], 'Difference between Added & Removed Articles', 'processed_minus_removed_over_time')

    # Normality Testing
    analyze_normality(df, 'processedMinusRemoved')
    analyze_normality(df, 'articlesRemoved')
    analyze_normality(df, 'articlesProcessed')
    analyze_normality(df, 'articleAmount')

    analyze_seasonality(df, 'processedMinusRemoved')
    analyze_seasonality(df, 'articlesRemoved')
    analyze_seasonality(df, 'articlesProcessed')
    analyze_seasonality(df, 'articleAmount')
else:
    print("Data does not meet lag criterai at length: " + len(df) + ". Must be of length 40.")