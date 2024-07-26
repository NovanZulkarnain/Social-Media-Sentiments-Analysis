import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'D:/SkillUp/Social Media Sentiments Analysis/sentimentdataset.csv'
data = pd.read_csv(file_path)

# Clean up the data by removing leading and trailing whitespaces in the 'Sentiment' column
data['Sentiment'] = data['Sentiment'].str.strip()

# Count the number of each sentiment type
sentiment_counts = data['Sentiment'].value_counts()

# Plotting the sentiment distribution as a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=['green', 'red', 'blue'])
plt.title('Sentiment Distribution')
plt.show()

# Optionally, you can add more visualizations, such as bar charts for sentiment per platform or country
# Sentiment per platform
platform_sentiment = data.groupby('Platform')['Sentiment'].value_counts().unstack().fillna(0)

platform_sentiment.plot(kind='bar', stacked=True, figsize=(12, 6), color=['red', 'blue', 'green'])
plt.title('Sentiment Distribution per Platform')
plt.xlabel('Platform')
plt.ylabel('Number of Sentiments')
plt.legend(title='Sentiment')
plt.show()

# Sentiment per country
country_sentiment = data.groupby('Country')['Sentiment'].value_counts().unstack().fillna(0)

country_sentiment.plot(kind='bar', stacked=True, figsize=(12, 6), color=['red', 'blue', 'green'])
plt.title('Sentiment Distribution per Country')
plt.xlabel('Country')
plt.ylabel('Number of Sentiments')
plt.legend(title='Sentiment')
plt.show()
