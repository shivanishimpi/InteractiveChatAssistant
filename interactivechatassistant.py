import pandas as pd
import nltk
import numpy as np
import re
from nltk.stem import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from sklearn.metrics import pairwise_distances
from nltk import word_tokenize
from nltk.corpus import stopwords

df=pd.read_excel('textCorpus.xlsx')
df.ffill(axis = 0,inplace=True) 
df1=df.head(10)
def step1(x):
    for i in x:
        a=str(i).lower()
        p=re.sub(r'[^a-z0-9]',' ',a)
        print(p)
s='tell me about your personality'
words=word_tokenize(s)
lemma = wordnet.WordNetLemmatizer() 
def text_normalization(text):
    text=str(text).lower()
    spl_char_text=re.sub(r'[^ a-z]','',text)
    tokens=nltk.word_tokenize(spl_char_text)
    lema=wordnet.WordNetLemmatizer()
    tags_list=pos_tag(tokens,tagset=None)
    lema_words=[]
    for token,pos_token in tags_list:
        if pos_token.startswith('V'):
            pos_val='v'
        elif pos_token.startswith('J'):
            pos_val='a'
        elif pos_token.startswith('R'):
            pos_val='r'
        else:
            pos_val='n'
        lema_token=lema.lemmatize(token,pos_val)
        lema_words.append(lema_token)
    return " ".join(lema_words)
df['lemmatized_text']=df['Context'].apply(text_normalization)
stop = stopwords.words('english')
cv = CountVectorizer()
X = cv.fit_transform(df['lemmatized_text']).toarray()

features = cv.get_feature_names()
df_bow = pd.DataFrame(X, columns = features)
Question ='Will you help me and tell me about yourself more'

Q=[]
a=Question.split()
for i in a:
    if i in stop:
        continue
    else:
        Q.append(i)
    b=" ".join(Q)

Question_lemma = text_normalization(b)
Question_bow = cv.transform([Question_lemma]).toarray()

cosine_value = 1- pairwise_distances(df_bow, Question_bow, metric = 'cosine' )

df['similarity_bow']=cosine_value
df_simi = pd.DataFrame(df, columns=['Text Response','similarity_bow'])
df_simi_sort = df_simi.sort_values(by='similarity_bow', ascending=False) 
threshold = 0.2
df_threshold = df_simi_sort[df_simi_sort['similarity_bow'] > threshold]

index_value = cosine_value.argmax() 

Question1 ='Tell me about yourself.'

tfidf=TfidfVectorizer() 
x_tfidf=tfidf.fit_transform(df['lemmatized_text']).toarray()
Question_lemma1 = text_normalization(Question1)
Question_tfidf = tfidf.transform([Question_lemma1]).toarray()

df_tfidf=pd.DataFrame(x_tfidf,columns=tfidf.get_feature_names())

cos=1-pairwise_distances(df_tfidf,Question_tfidf,metric='cosine')
df['similarity_tfidf']=cos 
df_simi_tfidf = pd.DataFrame(df, columns=['Text Response','similarity_tfidf'])

df_simi_tfidf_sort = df_simi_tfidf.sort_values(by='similarity_tfidf', ascending=False)

threshold = 0.2 
df_threshold = df_simi_tfidf_sort[df_simi_tfidf_sort['similarity_tfidf'] > threshold]

index_value1 = cos.argmax() # returns the index number of highest value

def stopword_(text):   
    tag_list=pos_tag(nltk.word_tokenize(text),tagset=None)
    stop=stopwords.words('english')
    lema=wordnet.WordNetLemmatizer()
    lema_word=[]
    for token,pos_token in tag_list:
        if token in stop:
            continue
        if pos_token.startswith('V'):
            pos_val='v'
        elif pos_token.startswith('J'):
            pos_val='a'
        elif pos_token.startswith('R'):
            pos_val='r'
        else:
            pos_val='n'
        lema_token=lema.lemmatize(token,pos_val)
        lema_word.append(lema_token)
    return " ".join(lema_word)

def chat_bow(text):
    s=stopword_(text)
    lemma=text_normalization(s)
    bow=cv.transform([lemma]).toarray()
    cosine_value = 1- pairwise_distances(df_bow,bow, metric = 'cosine' )
    index_value=cosine_value.argmax()
    return df['Text Response'].loc[index_value]

def chat_tfidf(text):
    lemma=text_normalization(text)
    tf=tfidf.transform([lemma]).toarray()
    cos=1-pairwise_distances(df_tfidf,tf,metric='cosine')
    index_value=cos.argmax()
    return df['Text Response'].loc[index_value]

flag = True
print('Hey there. I will be happy to help you out, if you wanna exit, type Bye')
while(flag==True):
  user_response=input()
  user_response=user_response.lower()
  if (user_response!='Bye' and user_response!='bye'):
    print(chat_tfidf(user_response))
  else:
    flag=False
    print('I hope to see you soon again, bye.')
