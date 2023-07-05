import datetime
import spacy #A natural language processing library.
import numpy as np

#Store values with repeated keys in entity recognition.
#e.g. {'Company': ['Tesla', 'Alibaba'], 'Region': ['China', 'USA']}
class Dictlist(dict):
    def __setitem__(self, key, value):
        try:
            self[key]
        except KeyError:
            super(Dictlist, self).__setitem__(key, [])
        self[key].append(value)

#textData: Pandas series, entityName: String
def findEntity(textData, entityName):
    #Use Natural language processing to interpret text.
    spacy.require_cpu()

    #Load tagger, parser, lemmatizer for English language.
    nlp = spacy.load("en_core_web_sm")
    entityDict = Dictlist()
    entity_weight = np.array([])

    for sentence in textData:
        #Process input sentence "sentence".
        sentence_nlp = nlp(sentence) 
        
        for word in sentence_nlp:
            #If this is a recognised entity in nltk package.
            if word.ent_type_:
                entityDict[word.ent_type_] = word.text.lower()
            #If this is not a recognised entity in nltk package.
            else:
                print('Not word.ent_type_:', word.ent_type_)

        #find out the most frequent named entities.
        try:
            if (entityName in entityDict['ORG'] or entityName in entityDict['PERSON']):
                #Weight of this news is 1.77
                entity_weight = np.concatenate((entity_weight, np.array([1.77])), axis=0)
            else:
                #Weight of this news is 1
                entity_weight = np.concatenate((entity_weight, np.array([1])), axis=0)
        #Target entity does not appear in the news.
        except KeyError:
                #Weight of this news is 1
                entity_weight = np.concatenate((entity_weight, np.array([1])), axis=0)

        #Clear content in entityDict.
        entityDict = Dictlist()

    #Numpy array, content = [weight of news 1, weight of news 2, ...]  
    return entity_weight
