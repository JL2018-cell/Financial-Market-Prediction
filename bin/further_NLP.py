#Further performing Sentiment Analysis using Text Classification
#-1 - negative sentiment, 1 - positive sentiment
import pandas as pd
import numpy as np

#Convert a collection of text documents to a matrix of token counts. 
#converting raw text to a numerical vector representation of words and n-grams. 
#This makes it easy to directly use this representation as features (signals) in 
#Machine Learning tasks such as for text classification and clustering.
from sklearn.feature_extraction.text import CountVectorizer

#Tokenizers divide strings into lists of substrings.
from nltk.tokenize import RegexpTokenizer
#Separate a dataset to train data and test data.
from sklearn.model_selection import train_test_split
#Used for classification with discrete features, 
from sklearn.naive_bayes import MultinomialNB
#Convert a collection of raw documents to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer
#Split dataset to train data and test data.
from sklearn.model_selection import train_test_split 
#Probability distribution function of multinomial distribution.
from sklearn.naive_bayes import MultinomialNB 
#quantifying the quality of predictions.
from sklearn import metrics 
#Local file, return the weight of each piece of news
import EntityRecognition

def emotionTrainMdl_MNB(data, testNum, weights):
    #Remove unwanted elements from out data like symbols and numbers
    #Splits a long string into substrings using a regular expression.
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize) #Convert text data to sparse matrix.
    #input = text
    text_counts= cv.fit_transform(data['Sentence'])

    #Split train and test set
    X_train, X_test, y_train, y_test = train_test_split(text_counts, data['Sentm_Scr'], test_size = testNum, shuffle = False)

    #Model Building and Evaluation:
    #After converting text to sparse matrix, 
    #fit sentiment distribution and text distribution 
    #according to multinomial probaility distribution.
    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_test) #Format = Numpy array.
    #return the fraction/number of correctly classified samples
    
    #result is: sentiment score = sum(sentiment score * weight)
    return np.dot(predicted, weights)

def emotionTrainMdl_IDF(data, testNum, weights):
    #Feature Generation using TF-IDF
    #TF-IDF(Term Frequency-Inverse Document Frequency) normalizes the document term matrix. It is the product of TF and IDF. Word with high tf-idf in a document, it is most of the times occurred in given documents and must be absent in the other documents.
    #Convert a collection of raw documents to a matrix of TF-IDF features. Equivalent to CountVectorizer followed by TfidfTransformer.
    tf = TfidfVectorizer()
    #text data used for training.
    text_tf = tf.fit_transform(data['Sentence'])
    
    #Split dataset into train data and test data.
    X_train, X_test, y_train, y_test = train_test_split(text_tf, data['Sentm_Scr'], test_size = testNum, shuffle = False)

    #Model Building and Evaluation (TF-IDF)
    #Model Generation Using Multinomial Naive Bayes
    #After converting text to sparse matrix, 
    #fit sentiment distribution and text distribution 
    #according to multinomial probaility distribution.
    clf = MultinomialNB().fit(X_train, y_train)
    predicted= clf.predict(X_test) #Format = Numpy array.

    return np.dot(predicted, weights) #sentiment score = sum(sentiment score * weight)

def rateEmotionTrain(trainText, testSencs, entityName): #testSencs = Comments taken from stock forum
    #If there is no sentence to be tested.
    if (len(testSencs) == 0): 
        return 0

    #Combine trainText with testSencs, pd.concat([df1, df2]). Sentiment Score of testSencs is randomly assigned because it will be revised finally in later NLP processing.
    testSencsDataFrame = pd.DataFrame({'Sentm_Scr':[i for i in range(len(testSencs))], 'Sentence': testSencs})

    #Combine traning text and sentences to be tested together.
    txtData = pd.concat([trainText, testSencsDataFrame])
    #Count number of data to be test data.
    testNum = len(testSencs)

    #Recognise entities, choose the last 'testNum' data. They are the text web-scrapped from Internet. Others are training data.
    #Give different weights to each news according to entity recognition result.
    #Return type = Numpy array.
    weights = EntityRecognition.findEntity(testSencs, entityName)
    #data.iloc[:0] = sentiment score
    predicted_MNB = emotionTrainMdl_MNB(txtData, testNum, weights)
    predicted_IDF = emotionTrainMdl_IDF(txtData, testNum, weights)

    #Average of predicted score form 2 methods.
    return (predicted_IDF + predicted_MNB)/2
