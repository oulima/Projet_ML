import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
import pickle
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from stop_words import get_stop_words
import nltk
import re
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


my_stop_word_list = get_stop_words('french')
final_stopwords_list = stopwords.words('french')

s_w=list(set(final_stopwords_list+my_stop_word_list))
s_w=[elem.lower() for elem in s_w]

fr = SnowballStemmer('french')

#le corpus
df=pd.read_csv('corpus.csv')
df['label']=df['rating']
positif=df[df['label']>3].sample(391)
negatif=df[df['label']<3]
Corpus=pd.concat([positif,negatif],ignore_index=True)[['review','label']]

for ind in Corpus['label'].index:
    if Corpus.loc[ind,'label'] > 3:
        Corpus.loc[ind,'label']=1
    elif Corpus.loc[ind,'label'] < 3:
        Corpus.loc[ind,'label']=0



def nettoyage(string):
   
    l=[]
    string=unidecode(string.lower())
    string=" ".join(re.findall("[a-zA-Z]+", string))
    
    for word in string.split():
        if word in s_w:
           continue
        else:
            l.append(fr.stem(word))
    return ' '.join(l)

Corpus['review_net']=Corpus['review'].apply(nettoyage)

def prediction(phrase):
   
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    user = transformer.fit_transform(loaded_vec.fit_transform([nettoyage(phrase)]))

    cls=pickle.load(open("cls.pkl", "rb"))
 
    return (cls.predict(user)[0],cls.predict_proba(user).max())


  
def entrainement():
    vectorizer = TfidfVectorizer()
    vectorizer.fit(Corpus['review_net'])
    X=vectorizer.transform(Corpus['review_net'])
    #sauver le voabulaire 
    pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

    y=Corpus['label']
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    cls=LogisticRegression(max_iter=300).fit(x_train,y_train)
    #sauver cls
    pickle.dump(cls,open("cls.pkl","wb"))

    return(cls.score(x_val,y_val))




