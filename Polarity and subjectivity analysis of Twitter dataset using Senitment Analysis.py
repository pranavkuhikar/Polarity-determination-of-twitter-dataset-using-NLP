#!/usr/bin/env python
# coding: utf-8

# ### Importing the necessary libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import json

from textblob import TextBlob
from wordcloud import WordCloud

plt.style.use('fivethirtyeight')


# ### reading the given json file and converting it to dataframe

# In[3]:


t = open(r"C:\Users\Pranav\Downloads\tweets.json", mode='r',encoding='cp1252')


# In[4]:


data = json.load(t)


# In[5]:


df1 = pd.DataFrame(data)


# In[6]:


df1


# ### we can see that the converted dataframe needs to be transposed.
# 

# In[7]:


df2 = df1.T


# In[8]:


df2.head()


# In[9]:


df2.shape


# ### converting the dataframe to csv file format

# In[10]:


df2.to_csv('tweets.csv')


# In[11]:


twitter = pd.read_csv(r"C:\Users\Pranav\Downloads\tweets.csv")


# In[12]:


twitter.head()


# In[13]:


twitter.shape


# In[14]:


twitter.info()


# In[15]:


twitter.isnull().sum()


# In[16]:


twitter.tweet_author.value_counts()


# *** we can see that "Patient Power" has the most number of tweets. 

# In[17]:


twitter.tweet_text.describe()


# *** it is evident that the most frequent tweet_text is "chronic lymphocytic leukemia慢性リンパ性白血病"

# In[18]:


twitter = twitter.drop(['_key'],axis=1)


# In[19]:


twitter.shape


# In[20]:


twitter.head()


# In[21]:


from collections import Counter


# In[22]:


col_list =  list(twitter["tweet_text"])


# In[23]:


print("Given List:\n",col_list)
res = max(set(col_list), key = col_list.count)
print("Element with highest frequency:\n",res)


# #### We need to create a user defined function to free the text from special symbols and unwanted characters

# In[24]:


def cleanText(text):
    text =re.sub(r'@[A-Za-z0-9]+','', text)
    text =re.sub(r'#','',text)
    text =re.sub(r'RT[\s]+','',text)
    text =re.sub(r'https?:\/\/\S+','',text)
    
    return text

twitter['tweet_text'] = twitter['tweet_text'].apply(cleanText)

#show clean text

twitter


# ### creating subjectivity and polarity scores for each record

# In[25]:


# create a function to analyze subjectivity

def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# create a function to analyze polarity

def getpolarity(text):
    return TextBlob(text).sentiment.polarity

# create new columns
twitter['Subjectivity'] = twitter['tweet_text'].apply(getsubjectivity)
twitter['Polarity'] = twitter['tweet_text'].apply(getpolarity)

twitter


# ### plotting word cloud to see the most frequent word

# In[26]:


# plot the word cloud

allwords = ' '.join([twts for twts in twitter['tweet_text']])
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=119).generate(allwords)

plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# ### analysing positive, neutral and negative sentiment based on polarity scores

# In[27]:


# function to create negative, neutral, positive analysis.

def getanalysis(score):
    if score <0:
        return "Negative"
    elif score==0:
        return "Neutral"
    else:
        return "Positive"

twitter['Analysis'] = twitter['Polarity'].apply(getanalysis) 

twitter


# ### Below is a function to print all the positive and negative tweets which were identified by the algorithm.

# In[28]:


# print all positive tweets

j=1
sortedDF = twitter.sort_values(by=['Polarity'])
for i in range(0, sortedDF.shape[0]):
    if(sortedDF['Analysis'][i] =='Positive'):
        print(str(j) +') '+sortedDF['tweet_text'][i])
        print()
        j = j+1


# In[29]:


# print negative tweets:

j=1
sortedDF = twitter.sort_values(by=['Polarity'], ascending=False)
for i in range(0, sortedDF.shape[0]):
    if(sortedDF['Analysis'][i]=='Negative'):
        print(str(j) +')'+ sortedDF['tweet_text'][i])
        print()
        j=j+1


# #### Plotting the polarity and subjectivity score on a scatterplot to visualise the distribution/variation of the data points

# In[30]:


# plot the polarity and subjectivity
plt.figure(figsize=(8,6))
for i in range(0, twitter.shape[0]):
    plt.scatter(twitter['Polarity'][i], twitter['Subjectivity'][i], color='red')
    
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()


# In[31]:


# get percentage of positive tweets
ptweets = twitter[twitter.Analysis=='Positive']
ptweets = ptweets['tweet_text']

round((ptweets.shape[0]/twitter.shape[0])*100,1)


# In[32]:


# get percentage of positive tweets
ntweets = twitter[twitter.Analysis=='Negative']
ntweets = ntweets['tweet_text']

round((ntweets.shape[0]/twitter.shape[0])*100,1)


# #### We can observe that the total no of positive tweets (in percentage) is 42.1%. 
# #### For negative tweets, the number is 7.3%
# #### and the remaining i.e. 50.6% belongs to the neutral class.

# In[33]:


twitter['Analysis'].value_counts()

# plot and visualize the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
twitter['Analysis'].value_counts().plot(kind='bar')
plt.show()


# We can see that the most commonly identified tweet belongs to the Neutral class followed by Positive and Negative classes

# #### We need to save our work in a csv format for easy accessibility. 

# In[53]:


twitter.to_csv('raw_data.csv', index=False)


# #pwd
# you can use this command to see the path of your saved file.

# ### creating entities

# In[55]:


import spacy
from spacy import displacy

nlp= spacy.load("en_core_web_sm")

def spacy_ner(text):
    text = text.replace('\n',' ')
    doc = nlp(text)
    entities=[]
    labels=[]
    position_start = []
    position_end = []
    
    for ent in doc.ents:
        if ent.label_ in ['PERSON','ORG','GPE']:
            entities.append(ent)
            labels.append(ent.label_)
    return entities, labels

def fit_ner(df):
    # the dataframe should have a column named text
    print("fitting spacy ner model")
    ner = df['text'].apply(spacy_ner)
    ner_org ={}
    ner_pe ={}
    ner_gpe ={}
    
    for x in ner:
        #print(list(x))
        for entity, label in zip(x[0],x[1]):
            # print(type(entity.text))
            if label =='ORG':
                ner_org[entity.text] = ner_org.get(entity.text,0)+1
            elif label =='Person':
                ner_per[entity.text] = ner_per.get(entity.text,0)+1
            else:
                ner_gpe[entity.text] = ner_gpe.get(entity.text,0)+1
            
    return {'ORG':ner_org,'PER':ner_per,'GPE':ner_gpe}  


# In[37]:


words = nltk.word_tokenize(allwords)


# In[38]:


pos_tags = nltk.pos_tag(words)


# In[39]:


pos_tags


# In[40]:


chunks = nltk.ne_chunk(pos_tags, binary=True)
for chunk in chunks:
    print(chunk)


# In[41]:


entities = []
labels=[]
for chunk in chunks:
    if hasattr(chunk, 'label'):
        # print chunk
        entities.append(' '.join(c[0] for c in chunk))
        labels.append(chunk.label())
        
entities_labels = list(set(zip(entities, labels)))
entities_df = pd.DataFrame(entities_labels)
entities_df.columns=['Entities','Label']


# In[42]:


entities_df


# In[43]:


entities_df.count(axis='columns')


# ### converting entities dataframe to csv file

# In[44]:


entities_df.to_csv('Entities.csv')


# In[45]:


df = twitter


# In[46]:


df.tweet_text = df.tweet_text.str.lower()

from nltk.corpus import stopwords

stopwords.words('english')


# In[47]:


import string
string.punctuation


# #### creating a function to clean the text from punctuation, stopwords.

# In[48]:


def text_process(mess):            ### creating a function
    """                                                        ## a docstring
    1. remove the punctuation
    2. remove the stopwords
    3. return the list of clean textwords
    
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = "".join(nopunc)
    
    return [ word for word in nopunc.split() if word not in stopwords.words("english")]


# In[49]:


from sklearn.feature_extraction.text import CountVectorizer

after_transf= CountVectorizer(analyzer=text_process).fit(df['tweet_text'])


# In[50]:


after_transf.vocabulary_


# In[ ]:


We can see the count of words after the transformation.


# ### saving the objectives in the required csv file format

# In[56]:


twitter.to_csv('objective2.csv')


# In[57]:


entities_df.to_csv('objective1.csv')

