import nltk
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords #read corpus files in a variety of formats.
from nltk.tag import pos_tag #part-of-speech tagging, It uses the Penn Treebank tagset, www.nltk.org/api/nltk.tag.html
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist #Count Frequency distribution of words.
import pandas as pd
import re, string
import random

#Removing Noise from the Data
#Remove Hyperlinks, '@', punctuation and special characters in text.
#Convert all letters to lower case.
def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        #searches for a substring that matches a URL starting with http:// or https://
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

#takes a list of tweets as an argument to provide a list of words in all of the tweet tokens joined.
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

#Arguments format: pandas detaFrame, list of strings, float 
def findEmotion(samples, testSencs, ratio):

    #tokenization, a list of list, inner list contains strings of a sentence.
    #Prepare to train model with samples with known sentiment scores extracted from database.
    comment_tokens = [x.split() for x in list(samples.iloc[:,1])]
    
    #Normalizing the Data
    #NNP: Noun, proper, singular
    #NN: Noun, common, singular or mass
    #IN: Preposition or conjunction, subordinating
    #VBG: Verb, gerund or present participle
    #VBN: Verb, past participle
    
    #A list of 'meaningless' words in English. e.g. on, a, the.
    stop_words = stopwords.words('english') 
    
    #use the remove_noise() function to clean the words in samples.
    cleaned_tokens_list = []
    for token in comment_tokens:
        cleaned_tokens_list.append(remove_noise(token, stop_words))
    
    #Determining Word Density
    #Combine all words present in cleaned_tokens_list in to a single list.
    all_pos_words = get_all_words(cleaned_tokens_list) 
    
    #Preparing Data for the Model
    #Converting Tokens to a Dictionary
    #use the Naive Bayes classifier in NLTK to perform the modeling exercise.
    #Convert to a list of dictionary, each sentence is a dictionary. Every word (key) has initial boolean value = True.
    tokens_for_model = get_tweets_for_model(cleaned_tokens_list)
    
    #Splitting the Dataset for Training and Testing the Model
    #dataset = [(tweet_dict, "Positive") for tweet_dict in tokens_for_model]
    #Assignment corresponding sentiment score to processed sentences. Format: [({dict}, 1)]
    dataset = [(tweet_dict, samples.iloc[i,0]) for i, tweet_dict in enumerate(tokens_for_model)] 
    
    #dataset = positive_dataset + negative_dataset
    train_data = dataset[:int(len(dataset)*(1 - ratio))]
    test_data = dataset[int(len(dataset)*ratio):]
    
    #Step 7: Building and Testing the Model
    #Use the .train() method to train the model and the .accuracy() method to test the model on the testing data.
    classifier = NaiveBayesClassifier.train(train_data)
    
    #Process sentences to be classified. Output format = list of list, innter list contains strings.
    custom_tokens = [remove_noise(word_tokenize(testSenc)) for testSenc in testSencs]
    #Classify emotion of a sentence.
    emotion_scores = [classifier.classify(dict([token, True] for token in custom_token)) for custom_token in custom_tokens]
    return pd.DataFrame({'Sentiment': emotion_scores, 'Phrase': testSencs})
  
