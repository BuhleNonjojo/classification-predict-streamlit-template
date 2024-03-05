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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from PIL import Image

# Text Processing Libraries.

import emoji  
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

news_vectorizer = open("Modelb.pkl","rb")
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
	st.title("Tweet Classifer")
	st.title("Lead Engine.")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "About Us", "Twitter", "Sentiments"]
	selection = st.sidebar.selectbox("Options", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")
		st.markdown("We are Lead Engine")
		st.image (IMG_6150.JPG)

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models today")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			predictor = joblib.load(open(os.path.join("resources/linear_svc_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			predictor = joblib.load(open(os.path.join("resources/multinomial_nb_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
