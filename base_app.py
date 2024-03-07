"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies

import streamlit as st
import joblib,os



# Data dependencies

import emoji 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from PIL import Image

# Text Processing Libraries.

from nltk.corpus import stopwords  
from nltk.stem import WordNetLemmatizer  
from nltk import download as nltk_download  
import regex  
import string  
import unicodedata  
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack  

# Vectorizer

news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

#with open('resources/TFIDF_Vec.pkl', 'rb') as file:
        #tf_vect = pickle.load(file)	
#with open('resources/TFIDF_Vec.pkl', 'rb') as file:
        #tf_vect = pickle.load(file)
		# 		

#new vectorizer

news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

#preprocess function
def preprocess_tweet(tweets):
	# function to determine if there is a retweet within the tweet
	def is_retweet(tweet):
		word_list = tweet.split()
		if "RT" in word_list:
			return 1
		else:
			return 0
	tweets["is_retweet"] = tweets["message"].apply(is_retweet, 1)

	# function to extract retween handles from tweet	
	def get_retweet(tweet):
		word_list = tweet.split()
		if word_list[0] == 'RT':
			handle = word_list[1]
		else:
			handle = ''
		handle = handle.replace(':', "")

		return handle
	tweets['retweet_handle'] = tweets['message'].apply(get_retweet,1)

	# function to count the number of hashtags within the tweet
	def count_hashtag(tweet):
		count = 0

		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				count += 1

		return count
	tweets["hashtag_count"] = tweets["message"].apply(count_hashtag, 1)

	# function to extract the hashtags within the tweet
	def get_hashtag(tweet):
		hashtags = []
		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				hashtags.append(word)

		returnstr = ""

		for tag in hashtags:
			returnstr + " " + tag

		return returnstr
	tweets["hashtags"] = tweets["message"].apply(get_hashtag, 1)

	# function to count the number of mentions within the tweet
	def count_mentions(tweet):
		count = 0
		word_list = tweet.split()
		if "RT" in word_list:
			count += -1 # remove mentions contained in retweet from consideration
		
		for word in word_list:
			if word[0] == '@':
				count += 1
		if count == -1:
			count = 0
		return count
	tweets["mention_count"] = tweets["message"].apply(count_mentions, 1)

	def get_mentions(tweet):
		mentions = []
		word_list = tweet.split()

		if "RT" in word_list:
			word_list.pop(1) # Retweets don't count as mentions, so we remove the retweet handle from consideration

		for word in word_list:
			if word[0] == '@':
				mentions.append(word)

		returnstr = ""

		for handle in mentions:
			returnstr + " " + handle

		return returnstr
	tweets["mentions"] =  tweets["message"].apply(get_mentions, 1)

	# function to count the number of web links within tweet
	def count_links(tweet):
		count = tweet.count("https:")
		return count 
	tweets["link_count"] = tweets["message"].apply(count_links, 1)

	# function to replace URLs within the tweet
	pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
	subs_url = r'url-web'
	tweets['message'] = tweets['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
	
	# function count number of newlines within tweet
	def enter_count(tweet):
		count = tweet.count('\n')
		return count
	tweets["newline_count"] = tweets["message"].apply(enter_count, 1)

	# function to count number of exclaimation marks within tweet
	def exclamation_count(tweet):
		count = tweet.count('!')
		return count 
	tweets["exclamation_count"] =  tweets["message"].apply(exclamation_count, 1)
	
	# Remove handles from tweet
	def remove_handles(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == "@":
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + " "

		return returnstr
	tweets['message'] = tweets['message'].apply(remove_handles)

	# Remove hashtags from tweet
	def remove_hashtags(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == '#':
				wordlist.remove(word)
		returnstr = ''
		for word in wordlist:
			returnstr += word + " "

		return returnstr
	
	tweets["message"] = tweets["message"].apply(remove_hashtags)

	# Remove RT from tweet
	def remove_rt(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word == 'rt' or word == 'RT':
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + ' '

		return returnstr
	
	tweets['message'] = tweets['message'].apply(remove_rt)

	# function to translate emojis and emoticons
	def fix_emojis(tweet):
		newtweet = emoji.demojize(tweet)  # Translates 👍 emoji into a form like :thumbs_up: for example
		newtweet = newtweet.replace("_", " ") # Beneficial to split emoji text into multiple words
		newtweet = newtweet.replace(":", " ") # Separate emoji from rest of the words
		returntweet = newtweet.lower() # make sure no capitalisation sneaks in

		return returntweet
	tweets["message"] = tweets['message'].apply(fix_emojis)

	# function to remove punctuation from the tweet
	def remove_punctuation(tweet):
		return ''.join([l for l in tweet if l not in string.punctuation])
	
	tweets['message'] = tweets['message'].apply(remove_punctuation)
	
	#transform tweets into lowercase version of tweets
	def lowercase(tweet):
		return tweet.lower()
	tweets["message"] = tweets["message"].apply(lowercase)

	# remove stop words from the tweet
	def remove_stop_words(tweet):
		words = tweet.split()
		return " " .join([t for t in words if t not in stopwords.words('english')])
	tweets["message"] = tweets['message'].apply(remove_stop_words)

	# function to replace contractions
	def fix_contractions(tweet):
		expanded_words = []
		for word in tweet.split():
			expanded_words.append(contractions.fix(word))

		returnstr = " ".join(expanded_words)
		return returnstr
	tweets["message"] = tweets['message'].apply(fix_contractions)

	# function to replace strange characters in tweet with closest ascii equivalent
	def clean_tweet(tweet):
		normalized_tweet = unicodedata.normalize('NFKD',tweet)

		cleaned_tweet = normalized_tweet.encode('ascii', 'ignore').decode('utf-8')

		return cleaned_tweet.lower()
	tweets["message"] = tweets['message'].apply(clean_tweet)

	#function to remove numbers from tweet
	def remove_numbers(tweet):
		return ''.join(char for char in tweet if not char.isdigit())
	
	tweets["message"] = tweets['message'].apply(remove_numbers)

	# Create a lemmatizer object
	lemmatizer = WordNetLemmatizer()

	# Create function to lemmatize tweet content
	def tweet_lemma(tweet,lemmatizer):
		list_of_lemmas = [lemmatizer.lemmatize(word) for word in tweet.split()]
		return " ".join(list_of_lemmas)
	tweets["message"] = tweets["message"].apply(tweet_lemma, args=(lemmatizer, ))

	# Make dataframe of all word counts in the data
	twt_wordcounts = pd.DataFrame(tweets['message'].str.split(expand=True).stack().value_counts())
	twt_wordcounts.reset_index(inplace=True)
	twt_wordcounts.rename(columns={"index": "word", 0:"count"}, inplace=True)
	
	# Extract unique words from data
	twt_unique_words = twt_wordcounts[twt_wordcounts["count"]==1]

	# make a list of unique words
	unique_wordlist = list(twt_unique_words["word"])

	# Function to remove unique words from data
	def remove_unique_words(tweet):
		words = tweet.split()
		return ' '.join([t for t in words if t not in unique_wordlist])
	tweets['message'] = tweets['message'].apply(remove_unique_words)

	#function to add retweets to message 
	def add_rt_handle(row):
		if row["retweet_handle"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " rt " + row["retweet_handle"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis=1)

	# Function to add retweets to message
	def add_hashtag(row):
		if row["hashtags"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["hashtags"]
		return ret
	tweets["message"] = tweets.apply(add_hashtag, axis = 1)

	# Function to add mentions to message
	def add_rt_handle(row):
		if row["mentions"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["mentions"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis = 1)

	# drop redundant columns
	tweets = tweets.drop(["retweet_handle", "hashtags", "mentions"], axis=1)

	return tweets

def preprocess_csv(tweets):
	# function to determine if there is a retweet within the tweet
	def is_retweet(tweet):
		word_list = tweet.split()
		if "RT" in word_list:
			return 1
		else:
			return 0
	tweets["is_retweet"] = tweets["message"].apply(is_retweet, 1)

	# function to extract retween handles from tweet	
	def get_retweet(tweet):
		word_list = tweet.split()
		if word_list[0] == 'RT':
			handle = word_list[1]
		else:
			handle = ''
		handle = handle.replace(':', "")

		return handle
	tweets['retweet_handle'] = tweets['message'].apply(get_retweet,1)

	# function to count the number of hashtags within the tweet
	def count_hashtag(tweet):
		count = 0
		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				count += 1

		return count
	tweets["hashtag_count"] = tweets["message"].apply(count_hashtag, 1)

	# function to extract the hashtags within the tweet
	def get_hashtag(tweet):
		hashtags = []
		word_list = tweet.split()
		for word in word_list:
			if word[0] == "#":
				hashtags.append(word)

		returnstr = ""

		for tag in hashtags:
			returnstr + " " + tag

		return returnstr
	tweets["hashtags"] = tweets["message"].apply(get_hashtag, 1)

	# function to count the number of mentions within the tweet
	def count_mentions(tweet):
		count = 0
		word_list = tweet.split()
		if "RT" in word_list:
			count += -1 # remove mentions contained in retweet from consideration
		
		for word in word_list:
			if word[0] == '@':
				count += 1
		if count == -1:
			count = 0
		return count
	tweets["mention_count"] = tweets["message"].apply(count_mentions, 1)

	def get_mentions(tweet):
		mentions = []
		word_list = tweet.split()

		if "RT" in word_list:
			word_list.pop(1) # Retweets don't count as mentions, so we remove the retweet handle from consideration

		for word in word_list:
			if word[0] == '@':
				mentions.append(word)

		returnstr = ""

		for handle in mentions:
			returnstr + " " + handle

		return returnstr
	tweets["mentions"] =  tweets["message"].apply(get_mentions, 1)

	# function to count the number of web links within tweet
	def count_links(tweet):
		count = tweet.count("https:")
		return count 
	tweets["link_count"] = tweets["message"].apply(count_links, 1)

	# function to replace URLs within the tweet
	pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
	subs_url = r'url-web'
	tweets['message'] = tweets['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
	
	# function count number of newlines within tweet
	def enter_count(tweet):
		count = tweet.count('\n')
		return count
	tweets["newline_count"] = tweets["message"].apply(enter_count, 1)

	# function to count number of exclaimation marks within tweet
	def exclamation_count(tweet):
		count = tweet.count('!')
		return count 
	tweets["exclamation_count"] =  tweets["message"].apply(exclamation_count, 1)
	
	# Remove handles from tweet
	def remove_handles(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == "@":
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + " "

		return returnstr
	tweets['message'] = tweets['message'].apply(remove_handles)

	# Remove hashtags from tweet
	def remove_hashtags(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word[0] == '#':
				wordlist.remove(word)
		returnstr = ''
		for word in wordlist:
			returnstr += word + " "

		return returnstr
	
	tweets["message"] = tweets["message"].apply(remove_hashtags)

	# Remove RT from tweet
	def remove_rt(tweet):
		wordlist = tweet.split()
		for word in wordlist:
			if word == 'rt' or word == 'RT':
				wordlist.remove(word)
		returnstr = ''

		for word in wordlist:
			returnstr += word + ' '

		return returnstr
	
	tweets['message'] = tweets['message'].apply(remove_rt)

	# function to translate emojis and emoticons
	def fix_emojis(tweet):
		newtweet = emoji.demojize(tweet)  # Translates 👍 emoji into a form like :thumbs_up: for example
		newtweet = newtweet.replace("_", " ") # Beneficial to split emoji text into multiple words
		newtweet = newtweet.replace(":", " ") # Separate emoji from rest of the words
		returntweet = newtweet.lower() # make sure no capitalisation sneaks in

		return returntweet
	tweets["message"] = tweets['message'].apply(fix_emojis)

	# function to remove punctuation from the tweet
	def remove_punctuation(tweet):
		return ''.join([l for l in tweet if l not in string.punctuation])
	
	tweets['message'] = tweets['message'].apply(remove_punctuation)
	
	#transform tweets into lowercase version of tweets
	def lowercase(tweet):
		return tweet.lower()
	tweets["message"] = tweets["message"].apply(lowercase)

	# remove stop words from the tweet
	def remove_stop_words(tweet):
		words = tweet.split()
		return " " .join([t for t in words if t not in stopwords.words('english')])
	tweets["message"] = tweets['message'].apply(remove_stop_words)

	# function to replace contractions
	def fix_contractions(tweet):
		expanded_words = []
		for word in tweet.split():
			expanded_words.append(contractions.fix(word))

		returnstr = " ".join(expanded_words)
		return returnstr
	tweets["message"] = tweets['message'].apply(fix_contractions)

	# function to replace strange characters in tweet with closest ascii equivalent
	def clean_tweet(tweet):
		normalized_tweet = unicodedata.normalize('NFKD',tweet)

		cleaned_tweet = normalized_tweet.encode('ascii', 'ignore').decode('utf-8')

		return cleaned_tweet.lower()
	tweets["message"] = tweets['message'].apply(clean_tweet)

	#function to remove numbers from tweet
	def remove_numbers(tweet):
		return ''.join(char for char in tweet if not char.isdigit())
	
	tweets["message"] = tweets['message'].apply(remove_numbers)

	# Create a lemmatizer object
	lemmatizer = WordNetLemmatizer()

	# Create function to lemmatize tweet content
	def tweet_lemma(tweet,lemmatizer):
		list_of_lemmas = [lemmatizer.lemmatize(word) for word in tweet.split()]
		return " ".join(list_of_lemmas)
	tweets["message"] = tweets["message"].apply(tweet_lemma, args=(lemmatizer, ))

	# Make dataframe of all word counts in the data
	twt_wordcounts = pd.DataFrame(tweets['message'].str.split(expand=True).stack().value_counts())
	twt_wordcounts.reset_index(inplace=True)
	twt_wordcounts.rename(columns={"index": "word", 0:"count"}, inplace=True)
	
	# Extract unique words from data
	twt_unique_words = twt_wordcounts[twt_wordcounts["count"]==1]

	# make a list of unique words
	unique_wordlist = list(twt_unique_words["word"])

	# Function to remove unique words from data
	def remove_unique_words(tweet):
		words = tweet.split()
		return ' '.join([t for t in words if t not in unique_wordlist])
	tweets['message'] = tweets['message'].apply(remove_unique_words)

	#function to add retweets to message 
	def add_rt_handle(row):
		if row["retweet_handle"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " rt " + row["retweet_handle"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis=1)

	# Function to add retweets to message
	def add_hashtag(row):
		if row["hashtags"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["hashtags"]
		return ret
	tweets["message"] = tweets.apply(add_hashtag, axis = 1)

	# Function to add mentions to message
	def add_rt_handle(row):
		if row["mentions"] == "":
			ret = row["message"]
		else:
			ret = row["message"] + " " + row["mentions"]
		return ret
	tweets["message"] = tweets.apply(add_rt_handle, axis = 1)

	# drop redundant columns
	tweets = tweets.drop(["retweet_handle", "hashtags", "mentions"], axis=1)

	return tweets

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	image = Image.open('resources/imgs/Blue White Creative Professional Modern Business Agency Pitch Deck Presentation Template (2).png')

	col1, col2 = st.columns([3, 3])
	with col1:
		st.image(image, use_column_width=True)
	with col2:
		st.title("Lead Engine")
		st.title("Twitter Sentiment Classifier App")
	
	# Creating sidebar with selection box -
	# you can create multiple pages this way
		
	st.sidebar.title('App Navigation')

	options = ["About Us", "Information", "Explore The Data","Model Explanations","Predictions", "Contact Us"]
	selection = st.sidebar.radio("Selectons", options)

	# Building out the home company page
	if selection == "About Us":
		st.info('Welcome to Lead Engine (PTY) LTD ')
		st.markdown("At Lead Engine, we're more than just a data management solution – we're your trusted partners in growth.")
		st.markdown("We understand the challenges you face in today's ever-evolving business landscape. That's why we're dedicated to taking the burden of data management off your shoulders, freeing you to focus on what matters most: cultivating strong customer relationships and driving results.")
		st.markdown("Our team of passionate experts doesn't just handle your data – they become an extension of your own. We meticulously ensure the accuracy and security of your information, allowing you to make data-driven decisions with confidence. Our expertise goes beyond data security; it's about unlocking the full potential of your customer data. We empower you to gain deeper insights into your customers' needs and behaviors, enabling you to personalize experiences and build lasting loyalty. Our ultimate goal isn't just to maximize your productivity or increase your profits in the short term. It's about future-proofing your business by providing the tools and insights you need to thrive in the ever-changing digital age.")
		st.markdown("With Lead Engine as your partner, you'll be equipped to adapt to new trends, anticipate customer demands, and stay ahead of the competition.")
		st.write('To access the codebase for this application, please visit the following GitHub repository:https://github.com/BuhleNonjojo/2309_Classification_NM3')

		st.subheader("Meet the team")

		# Director 
		col1, col2 = st.columns([1, 6])
		with col1:
			image_k = Image.open('resources/imgs/IMG_6150.JPG')
			st.image(image_k, use_column_width=True,caption = 'Director: Buhle Nonjojo')

		# Business Intelligence Developer
		col1, col2 = st.columns([1, 6])
		with col1:
			image_m = Image.open('resources/imgs/20240221_084117.jpg')
			st.image(image_m, use_column_width=True, caption = 'BI Developer: Koketso Mahlangu')

		# Data Analyst
		col1, col2 = st.columns([1, 6])
		with col1:
			image_h = Image.open('resources/imgs/IMG_5666.jpg')
			st.image(image_h, use_column_width=True, caption = 'Data Analyst: Ngcebo Khumalo')
		
		# Data Scientist
		col1, col2 = st.columns([1, 6])
		with col1:
			image_t = Image.open('resources/imgs/IMG_1707418875180.jpg')
			st.image(image_t, use_column_width=True, caption = 'Data Scientist: Maliviwe Mahambi')

		# Machine Learning Engineer
		col1, col2 = st.columns([1, 6])
		with col1:
			image_kg = Image.open('resources/imgs/WhatsApp_Image_2024-02-28_at_10.32.53_a4977d04.jpg')
			st.image(image_kg, use_column_width=True, caption = 'ML Engineer: Noluthando Mtshali')

		# Data Engineer
		col1, col2 = st.columns([1, 6])
		with col1:
			image_i = Image.open('resources/imgs/IMG_2911.jpg')
			st.image(image_i, use_column_width=True, caption = 'Data Engineer: Onkarabile Maele')


	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("We are Lead Engine")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the "Explore The Data" page
	if selection == "Explore The Data":
		st.info("Exploring The Data")
		st.subheader("Dataset")
		st.subheader("Overview of dataset")
		st.info("The dataset consists of three columns namely: Sentiment, Tweet and Tweet ID")
		st.markdown("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43,943 tweets were collected. Each tweet is labelled as one of 4 classes.")

		st.subheader("Distribution of data per sentiment class")
		st.info("The dataset consists of three columns namely: Sentiment, Tweet and Tweet ID")
		st.image('resources/imgs/distribution_of_data_in_each_class.png')
		st.markdown("From the figures above, we see that the dataset we are working with is very unbalanced. More than half of our dataset is people having pro-climate change sentiments, while only  8% of our data represents people with anti-climate change opinions. This might lead our models to become far better at identifying pro-climate change sentiment than anti-climate change sentiment, and we might need to consider balancing the data by resampling it.")

		st.subheader("Proportion of retweets")
		st.info("The dataset consists of three columns namely: Sentiment, Tweet and Tweet ID")
		st.image('resources/imgs/proportion_of_retweets_hashtags_and_original_mentions.png')
		st.markdown("We see that a staggering  60% of all our data is not original tweets, but retweets! This indicates that extracting more information from the retweets could prove integral to optimizing our model\'s predictive capabilities.")

		st.subheader("Popular retweet handles per sentiment group in a word cloud")
		st.info("The dataset consists of three columns namely: Sentiment, Tweet and Tweet ID")
		st.image('resources/imgs/wordcloud_of_popular_retweet_handles_per_sentiment_group.png')
		st.markdown("From the above, we see a clear difference between every sentiment with regards to who they are retweeting. This is great news, since it will provide an excellent feature within our model. Little overlap between categories is visible, which points to the fact that this feature could be a very strong predictor.")
		st.markdown('We see that people with anti-climate change sentiments retweets from users like @realDonaldTrump and @SteveSGoddard the most. Overall retweets associated with anti-climate science opinions are frequently sourced from prominent Republican figures such as Donald Trump, along with individuals who identify as climate change deniers, like Steve Goddard.')
		st.markdown('In contrast to this, people with pro-climate change views often retweet Democratic political figures such as @SenSanders and @KamalaHarris. Along with this, we see a trend to retweet comedians like @SethMacFarlane. The most retweeted individual for this category, is @StephenSchlegel.')
		st.markdown('Retweets in the factual news category mostly contains handles of media news organizations, like @thehill, @CNN, @wasgingtonpost etc...')
		st.markdown('People with neutral sentiments regarding climate change seems to not retweet overtly political figures. Instead, they retweet handles unknown to the writer like @CivilJustUs and @ULTRAVIOLENCE which no longer currently exist on twitter. The comedian @jay_zimmer is also a common retweeted incividual within this category.')

		st.subheader("Popular hashtags in per sentiments group")
		st.info("The dataset consists of three columns namely: Sentiment, Tweet and Tweet ID")
		st.image('resources/imgs/popular_hashtags_per_sentiment_group_wordcloud.png')
		st.markdown("Finally there is some hashtags that are more prominent within certain sentiment groups. Take #MAGA and #fakenews in the anti-climate change category, or #ImVotingBecause in the pro-climate change category. This indicates that some useful information can be extracted from this feature, and should remain within the model.")

		st.subheader("Popular mentions per sentiment group")
		st.info("The dataset consists of three columns namely: Sentiment, Tweet and Tweet ID")
		st.image('resources/imgs/popular_hashtags_per_sentiment_group_wordcloud.png')
		st.markdown("As was the case when we considered hashtags, we see that some handles get mentioned regardless of sentiment class. An example of this is @realDonaldTrump, which is prominent in every sentiment category, and as such should be removed before training our models, since it adds no value towards our data.")
		st.markdown("Furthermore, there is some mentions that are more prominent in certain classes than others. Take @LeoDiCaprio for example, which features heavily in both pro-climate change as well as neutral towards climate change sentiment, but is not represented in the other two categories. This indicates that this feature could be beneficial for categorizing our data, and should remain within the dataset.")		
 
	# Building out the "Model Explanations" page
	if selection == "Model Explanations":
		st.info("Learning more about our models")

		st.subheader("Logistic Regression")
		st.markdown('Logistic regression is a classification algorithm used to predict the probability of a binary outcome based on one or more input features. It models the relationship between the input variables and the probability of the outcome belonging to a particular class. Logistic regression uses the logistic function (also known as the sigmoid function) to map the output of a linear combination of the input features to a value between 0 and 1, representing the probability of belonging to the positive class.')
		st.markdown('In simpler terms, logistic regression aims to find the best-fitting S-shaped curve that separates the two classes. It estimates the coefficients (weights) of the input features through a process called maximum likelihood estimation, optimizing the parameters to maximize the likelihood of the observed data.')
		st.markdown('Once trained, logistic regression can make predictions by calculating the probability of the positive class based on the input features. A threshold is then applied to determine the final predicted class.')
		st.markdown('Logistic regression models are known for their simplicity and interpetability. Since they are more simplistic models, they are relatively quick to train and computationally efficient. They can also be expanded to handle multiclass classification as is the case for our data. This model does assume a linear relationship between the features and the log-odds of the outcome, however, which does not necessarily hold true in many cases. It is also sensitive to outliers and irrelevant features.')

		st.subheader("CatBoost Classifer")
		st.markdown('CatBoost is a machine learning model that tackles prediction tasks by combining multiple decision trees, a technique known as gradient boosting. Unlike some other models, CatBoost excels at handling data that includes categorical features, like text or zip codes.')
		st.markdown('It achieves this by using a special technique called "Ordered Boosting" that analyzes these features directly, without needing to convert them into numerical codes first. This allows CatBoost to capture the relationships within the categories more effectively.')
		st.markdown('Additionally, CatBoost incorporates regularization methods to prevent the model from overfitting to the training data, ensuring it performs well on unseen examples.')
		st.markdown('Overall, CatBoost\'s strength in handling categorical data, combined with its decision tree ensemble approach and effective regularization, makes it a powerful tool for various machine learning applications.')

		st.subheader("Decision Tree Classifier")
		st.markdown('A decision tree model is a machine learning method that learns by creating a tree-like structure. This structure mimics a series of if-then-else questions, where each question splits the data into smaller and more specific groups.')
		st.markdown('It starts at the root node, which represents the entire dataset. Here, the algorithm identifies the most important feature that best separates the data according to the target variable (like predicting whether an email is spam).')
		st.markdown('The data is then split based on this feature, with branches leading to child nodes containing data that share a specific value of that feature.')
		st.markdown('This process continues at each child node, using different features to further divide the data until it reaches leaf nodes. These leaf nodes contain the final predictions, representing the most likely outcome based on the features used throughout the tree.')		

		st.subheader("Linear Support Vector Classifier")
		st.markdown('Linear Support Vector Classification (LinearSVC) is a variant of Support Vector Machines (SVMs) used for classification tasks. It employs a linear kernel to create a hyperplane in the high-dimensional feature space to separate different classes of data.')
		st.markdown('Unlike other svms that might use various kernels, LinearSVC assumes a linear relationship between the features and the target variable. This model is robust and efficient in numerous applications, offering good performance even with less training data. However, its effectiveness can be sensitive to feature selection and requires careful preprocessing of the data.')
		st.markdown('Key parameters include the C parameter, controlling the trade-off between a smooth decision boundary and classifying training points correctly. Despite its underlying assumption of a linear relationship, LinearSVC has proven to be versatile in its performance, with careful tuning and preprocessing.')
		st.markdown('Lastly unlike most machine learning models, linear support vector machines offer a visually intuitive geometric interpretation by finding a hyperplane that maximizes the separation between data points in different classes, making them particularly interesting for their interpretability, efficiency, and fast training speeds, even though their application is limited to data that can be separated linearly.')
		
		st.subheader("Random Tree Classifier")
		st.markdown('Random forest consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model\'s prediction. The reason behind is that a large number of relatively uncorrelated models (decision trees) operating as a group will outperform any of the individual constituent models. The low correlation between models is the key, with trees protecting each other from their individual errors(as long as they do not deviate in the same direction)')
		st.markdown('While some trees may be wrong, many other trees will be right, so as a group the trees are able to move in the correct direction.')
		st.markdown('Random forest ensures that the behavior of each individual is not too correlated with the behavior of any other tree in the model by using bagging or bootstrap aggregation which allows each individual tree to randomly sample from the dataset with replacement resulting in different trees. Random forest also make use of feature randomness, in decision trees when it is time to split at each mode(a splitting point), we consider every possible feature and pick the one that produces the most separation between the observation in the left node vs those in the right node')
		st.markdown('In contrast, each tree in the random forest can only pick from a random subset of features, forcing even more variation amongst the trees resulting in lower correlation and more diversification.Son in random forest, we end up with trees that are not only trained on different sets of data through bagging but also use different features to make decisions(compared to individual trees which consider every feature to make a decision).')

		st.subheader("Support Vector Gemma Classifier")
		st.markdown('Gemma, also known as the Support Vector Classifier (SVC), is a powerful algorithm for classification tasks. It operates by finding a hyperplane in high-dimensional space that best separates the data points of different classes with the maximum margin.')
		st.markdown('For data that is not naturally separable in the original feature space, Gemma can employ a technique called the kernel trick. This trick essentially maps the data points to a higher-dimensional space where they become linearly separable. This allows Gemma to find an effective hyperplane even for complex datasets.')
		st.markdown('Once the data is mapped (if necessary), Gemma identifies the data points closest to the hyperplane from each class. These points are called support vectors as they play a crucial role in defining the margin and the model''s decision boundary. The core principle of Gemma is to maximize the margin. This margin refers to the distance between the hyperplane and the closest support vectors from each class. By maximizing this margin, Gemma aims to create a robust decision boundary that can effectively generalize to unseen data.')
		st.markdown('During prediction, Gemma takes a new data point and maps it to the same high-dimensional space (if applicable). The model then determines on which side of the hyperplane the new point lies, classifying it into the corresponding class based on the hyperplane''s orientation.')
		

		st.subheader("Support Vector Poly Classifier")
		st.markdown('Support Vector Classifier (SVC) with a polynomial kernel, also known as SVC(poly), is a powerful tool for handling non-linearly separable data in classification tasks.SVC(poly) first takes the original data points and maps them into a higher-dimensional space using a specific polynomial function. This transformation allows the data points, which might not be linearly separable in the original space, to become linearly separable in the higher-dimensional space.')
		st.markdown('Similar to a linear SVC, SVC(poly) then seeks to find a hyperplane within this higher-dimensional space that maximizes the margin between the data points belonging to different classes. This margin still represents the distance between the hyperplane and the closest data points from each class, called support vectors.')
		st.markdown('Once the optimal hyperplane is found in the higher-dimensional space, the model effectively maps it back to the original feature space. This creates a non-linear decision boundary in the original space, allowing for classification of data points that couldn''t be separated linearly before.')
		st.markdown(' Interestingly, SVC(poly) avoids explicitly performing the high-dimensional mapping for every data point during training and prediction. Instead, it utilizes a technique called the kernel trick. This trick allows the model to work directly with the data points in the original space while still leveraging the benefits of the higher-dimensional representation, making it computationally efficient even for large datasets. In essence, SVC(poly) overcomes the limitation of linear separation by transforming data into a higher-dimensional space, finding an optimal hyperplane there, and mapping it back to the original space to create a non-linear decision boundary for classification in various machine learning tasks.')
		
		st.subheader("Multinominal Naives Bayes Classifier")
		st.markdown('Naive Bayes is a probabilistic classification model based on Bayes theorem, which calculates the probability of a class given the input features. It assumes that the features are conditionally independent, meaning that the presence of one feature does not affect the presence of another feature. Despite this simplifying assumption, Naive Bayes can be surprisingly effective in many real-world scenarios.')
		st.markdown('Multinomial Naive Bayes is a variant of Naive Bayes that is specifically designed for classification tasks with discrete features. It is commonly used for text classification, where the input features are typically word frequencies or counts. Unlike Gaussian Naive Bayes, which assumes a Gaussian distribution for continuous features, Multinomial Naive Bayes assumes a multinomial distribution for discrete features..')
		st.markdown('In Multinomial Naive Bayes, the model learns the probability distribution of each feature given the class label. It estimates the probabilities using the training data, where the feature values represent the frequencies or counts of each feature in the documents of each class. To predict the class of a new instance, the model calculates the likelihood of observing the given feature values for each class and combines it with the prior probability of the class using Bayes theorem. The class with the highest probability is chosen as the predicted class.')
		st.markdown('The key difference between this model, and the Gaussian Naive Bayes is in the assumptions made about the data. Multinomial Naive Bayes assumes a multinomial distribution for discrete features, whereas Gaussian Naive Bayes assumes a Gaussian distribution for continuous features. Multinomial Naive Bayes is appropriate for discrete features, such as word counts, while Gaussian Naive Bayes is suitable for continuous or normally distributed features.This model is generally known to be efficient, even in large feature spaces. It also works well with unbalanced data, which is handy in our case. It is also able to handle multiclass classification problems, which our classification falls into.')

		st.subheader("XGBoost Classifier")
		st.markdown('XGBoost stands for Extreme Gradient Boosting and is a gradient boosting algorithm known for its high performance and accuracy in various machine learning tasks, including classification. It is an ensemble method that combines the predictions of multiple weak predictive models, usually decision trees, to create a strong predictive model. XGBoost builds an ensemble of decision trees sequentially, where each new tree is trained to correct the mistakes made by the previous trees.')
		st.markdown('It uses a gradient-based optimization technique to minimize a specific loss function, such as logistic loss for classification tasks. The algorithm calculates gradients and hessians to update the model parameters, ensuring that each subsequent tree focuses on the areas where the previous trees performed poorly.')
		st.markdown('Additionally, XGBoost incorporates several regularization techniques, such as shrinkage (learning rate) and tree pruning, to prevent overfitting and improve generalization. It also supports parallelization and distributed computing, making it efficient for training on large datasets.')
		st.markdown('XGBoost boasts exceptional predictive performance and accuracy, and is robust to outliers in the data. It also supports feature importance estimation, allowing for better understanding of feature contributions. It does however require careful hyperparameter tuning for optimal performance. It is also computationally intensive, especially with a large number of trees and complex datasets.')

	# Building out the Predictions page
	if selection == 'Predictions':
		st.write('Predict the sentiment of each twitter using various models with each tweet falling into one of 4 categories: anti-man made climate change, neutral, pro-man made climate change and lastly, whether a tweet represent factual news!')
	
		pred_type = st.sidebar.selectbox("Predict sentiment of a single tweet or submit a csv for multiple tweets", ('Single Tweet', 'Multiple Tweets'))

		if pred_type == "Single Tweet":
			st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter tweet here","Type Here")

			options = ["Logistic Regression Classifier","CatBoost Classfier", "Decision Tree Classifier","Linear Support Vector Classifier","Random Tree Forest Classifier","Support Vector Gemma Classifier", "Support Vector Poly Classifier", " Multinomial Naive Bayes Classifier", "XGBoost Classifier"] 
			selection = st.selectbox("Choose Your Model", options)

			if st.button("Classify Tweet"):
				#process single tweet using our preprocess_tweet() function

				# create dataframe for tweet
				text = [tweet_text]
				df_tweet = pd.DataFrame(text, columns=['message'])

				processed_tweet = preprocess_tweet(df_tweet)
				
				# Create a dictionary for tweet prediction outputs
				dictionary_tweets = {'[-1]': "A tweet refuting man-made climate change",
                     				  '[0]': "A tweet neither supporting nor refuting the belief of man-made climate change",
                     				  '[1]': "A pro climate change tweet",
                     				  '[2]': "This tweet refers to factual news about climate change"}

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = None
				X_pred = None
				if selection == "Logistic Regression Classifier":
					lr = pickle.load(open("resources/Logistic_regression.pkl",'rb'))
					predicton= joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))

				if selection == "CatBoost Classfier":
					cb = pickle.load(open("resources/.CB.pkl",'rb'))
					predicton= joblib.load(open(os.path.join("resources/.CB.pkl"),"rb"))

				if selection == "Decision Tree Classifier":
					dt = pickle.load(open("resources/DT.pkl",'rb'))
					predicton= joblib.load(open(os.path.join("resources/DT.pkl"),"rb"))

				if selection == "Linear Support Vector Classifier":
					lsv = pickle.load(open("resources/linear_svc_model.pkl,'rb"))
					predicton= joblib.load(open(os.path.join("resources/linear_svc_model.pkl"),"rb"))

				if selection == "Random Tree Forest Classifier":
					rt = pickle.load(open("resources/.RFC.pkl",'rb'))
					predicton= joblib.load(open(os.path.join("resources/.RFC.pkl"),"rb"))

				if selection == "Support Vector Gemma Classifier":
					svg = pickle.load(open("resources/svc_gemma.pkl",'rb'))
					predicton= joblib.load(open(os.path.join("resources/svc_gemma.pkl"),"rb"))

				if selection == "Support Vector Poly Classifier":
					svp = pickle.load(open("resources/svc_poly.pkl",'rb'))
					predicton= joblib.load(open(os.path.join("resources/svc_poly.pkl"),"rb"))

				if selection == "Multinomial Naive Bayes Classifier":
					mnb = pickle.load(open("resources/multinomial_nb_model.pkl",'rb'))
					predicton= joblib.load(open(os.path.join("resources/multinomial_nb_model.pkl"),"rb"))

				if selection == "XGBoost Classifier":
					xgb = pickle.load(open('resources/.XGB.pkl'))
					predicton= joblib.load(open(os.path.join("resources/.XGB.pkl"),"rb"))

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.

	# Building out the home company page
	if selection == "Contact Us":
		st.info('Lead Engine (PTY) LTD ')
		st.markdown("Contact Us:")
		st.markdown("Website: www.leadengine.com.")
		st.markdown("Telephone: +123-456-7890.")
		st.markdown("Email: contact@leadengine.com")
		st.markdown('Location : 123 Anywhere St., Any City, ST 12345')
		st.write('We are just a email or phone call or vist away.')

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()