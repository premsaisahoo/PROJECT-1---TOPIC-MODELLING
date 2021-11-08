# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:34:43 2021

@author: Ranjeet
"""

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd # not used in gnews
import urllib3
from urllib.request import Request,urlopen
import html5lib


#################### Web Scraping ####################

root = "https://www.google.com/"
link = "https://www.google.com/search?q=social+events&rlz=1C1CHBF_enIN810IN810&biw=1280&bih=609&tbs=qdr%3Am&tbm=nws&sxsrf=ALeKk02o0f3hHmvRVA0lzXLcbFDpNNi3RA%3A1627020038578&ei=Blv6YIG9Iu-4mAWK0qmACg&oq=social+events&gs_l=psy-ab.3..0i433k1j0l6j0i10k1j0l2.845410.848467.0.848776.13.10.0.1.1.0.367.1253.2-1j3.4.0....0...1c.1.64.psy-ab..8.5.1255...0i433i131k1j0i433i67k1j0i433i131i67k1.0.Fw1DxCUyxVs"

def news(link):
    req = Request(link,headers={'User-Agent':'Chrome/91.0.4472.114'})
    webpage = urlopen(req).read()
    with requests.Session() as c:
        soup = bs(webpage,'html5lib')
        #print(soup)
        for item in soup.find_all('div',attrs={'class':'ZINbbc xpd O9g5cc uUPGi'}):
            raw_link = (item.find('a',href=True)['href'])
            link = (raw_link.split("/url?q=")[1]).split('&sa=U&')[0]
            #print(item)
            title = (item.find('div',attrs={'class':'BNeawe vvjwJb AP7Wnd'}).get_text()) # BNeawe vvjwJb AP7Wnd # JheGif nDgy9d
            description = (item.find('div',attrs={'class':'BNeawe s3v9rd AP7Wnd'}).get_text())
            
            title = title.replace(",","")
            description = description.replace(",","")
            
            time = description.split(" � ")[0]
            descript = description.split(" � ")[1]
            print(title)
            print(time)
            print(descript)
            print(link)
            document = open("data.csv","a",encoding="utf-8")
            document.write("{}, {}, {}, {} \n".format(title, time, descript, link))
            document.close()
        next = soup.find('a',attrs={'aria-label':'Next page'})
        next = (next['href'])
        link = root + next
        news(link)
news(link)


######################### START HERE #########################

col_names = ['Title','Time','Description','Link','Unnec']
data = pd.read_csv("C:\\Users\\Ranjeet\\myenv\\data.csv", names=col_names) #encoding="utf-8"

data[0:10]

data['news'] = data['Title'] + data['Description']

final = data.loc[:,['news']]


######################### DATA CLEANING #########################

final
final.info()
final[final.news.duplicated()].shape # 2732 duplicates

final_nd = final.drop_duplicates(subset = None, keep = 'first', inplace = False) # droppping duplicate

w = [news.strip() for news in final_nd.news] # removes leading and trailing characters
w = [news for news in w if news] # removes empty strings
w[0:10]


#### Lowercase, Removal - numbers,puctiations ####

import re
import string

def clean_text(text):
 ''', , and '''
#Make text lowercase   
 text = text.lower()
#remove text in square brackets
 text = re.sub(r'\[.*?\]', '', text)
#remove punctuation   
 text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
 text = re.sub(r'�',"",text)
#remove words containing numbers
 text = re.sub(r'\w*\d\w*', '', text)
 return text

news = map(clean_text, w) # map command used for lsit
news = list(news)

def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

news_a = map(remove_punctuation,news)
news_a = list(news_a)


#### Removal - Short words ####

def remove_short_word(text):
    text = ' '.join([w for w in text.split() if len(w) > 3])
    return text

news1 = map(remove_short_word, news_a) 
news1 = list(news1)

#### Tokenization ####

import nltk
from nltk.tokenize import word_tokenize

news2 = map(word_tokenize,news1)
news2 = list(news2)

#### Removal - Stopwords ####

# import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

stop_words = stopwords.words('english')
# stop_words.extend(['say', 'announce', 'reuter', 'daily', 'use','istthe','news'])

def remove_stop_word(text):
    text = [word for word in text if not word in stop_words]
    return text
    
news3 = map(remove_stop_word,news2)
news3 = list(news3)

#### Lemmatization ####

import spacy
nlp = spacy.load("en_core_web_sm")

def lemmatize(text):
    doc = nlp(' '.join(text))
    lemma_news = [token.lemma_ for token in doc]
    return lemma_news

news4 = map(lemmatize,news3)
news4 = list(news4)

news4 = map(remove_stop_word,news4)
news4 = list(news4)


######################### EDA #########################

### feature extraction ###

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

clean_news = [' '.join(ele) for ele in news4]
clean_news = [news for news in clean_news if news]

X = vectorizer.fit_transform(clean_news)

print(vectorizer.vocabulary_)
count_vect = vectorizer.vocabulary_  # 5920 unique words

X.toarray().shape

#### News Length ####

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

plt.figure(figsize=(10,6))
doc_lens = [len(d) for d in clean_news]
plt.hist(doc_lens, bins = 100)
plt.title('Distribution of News character length')
plt.ylabel('Number of News')
plt.xlabel('News character length')
sns.despine();

#### Wordcloud ####

import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off"); 

stopwords = STOPWORDS

wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(str(clean_news))
# Plot
plot_cloud(wordcloud)

#### Unigram ####

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(clean_news, 10)
unigram = pd.DataFrame(common_words, columns = ['unigram' , 'count'])

### Trigram ###

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3,3) , stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_trigram(clean_news, 10)
trigram = pd.DataFrame(common_words, columns = ['trigram' , 'count'])



######################### MODEL BUILDING #########################

from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


data_lemmatized = news4

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

id2word[0] # to see what word a given id corresponds to, for eg, here what 0 corresponds to

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

### LDA Model ###

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,                                           
                                           chunksize=100,
                                           passes=10,      
                                           per_word_topics=True)

# Print the Keyword in the 10 topics

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]



######################### MODEL EVALUATION #########################

# Compute Perplexity 

print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # new -8.0966 , a measure of how good the model is. lower the better.

# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda) # 0.43325



######################### MODEL OPTIMISATION #########################

### INVESTIGATING COHERENCE BY VARYING KEY PARAMETERS ###

### Coherence values for varying number of topics ###

def compute_coherence_values_TOPICS(corpus, dictionary, texts, start, limit, step):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics,random_state=100, passes = 10)
        model_list.append(model)
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values_TOPICS(corpus=corpus, dictionary=id2word, texts=data_lemmatized, start=2, limit=20, step=3)

# Plot graph of coherence values by varying number of topics

limit=20; start=2; step=3;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence"), loc='best')
plt.show()

# Print the coherence scores

for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

# go for 14 topics, by observing graph - 0.4033


### Coherence values for varying alpha ###

def compute_coherence_values_ALPHA(corpus, dictionary, num_topics, texts, start, limit, step):
    coherence_values = []
    model_list = []
    for alpha in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=14,alpha=alpha/10,random_state=100, passes = 10)
        model_list.append(model)
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values_ALPHA(dictionary=id2word, corpus=corpus, num_topics=14, texts=data_lemmatized, start=1, limit=10, step=1)

# Plot graph of coherence values by varying alpha

limit=10; start=1; step=1;
x_axis = []
for x in range(start, limit, step):
    x_axis.append(x/10)
plt.plot(x_axis, coherence_values)
plt.xlabel("Alpha")
plt.ylabel("Coherence score")
plt.legend(("coherence"), loc='best')
plt.show()

# Print the coherence scores

for m, cv in zip(x_axis, coherence_values):
    print("Alpha =", m, " has Coherence Value of", round(cv, 4))

# go for alpha 0.4


### Coherence values for varying eta ###

def compute_coherence_values_ETA(corpus, dictionary, num_topics, alpha, texts, start, limit, step):
    coherence_values = []
    model_list = []
    for eta in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=14, random_state=100, alpha=0.5, eta=eta/100, passes=10)
        model_list.append(model)
        coherencemodel = gensim.models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values_ETA(corpus=corpus, dictionary=id2word, num_topics=14, alpha=0.4, texts=data_lemmatized, start=25, limit=50, step=5)

# Plot graph of coherence values by varying eta

limit=50; start=25; step=5;
x_axis = []
for x in range(start, limit, step):
    x_axis.append(x/100)
plt.plot(x_axis, coherence_values)
plt.xlabel("Eta")
plt.ylabel("Coherence score")
plt.legend(("coherence"), loc='best')
plt.show()

# Print the coherence scores

for m, cv in zip(x_axis, coherence_values):
   print("Eta =", m, " has Coherence Value of", round(cv, 4))
    
# go for eta = 0.25



######################### FINAL MODEL #########################

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=14, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=0.5,
                                           eta = 0.3,
                                           per_word_topics=True)

# Print the Keyword in the 10 topics

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda) # 0.44587



######################### OUTPUT INTERPRETATION #########################

### Finding dominant topic in each sentence ###

import collections

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=news1):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=news1)


# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)


### Finding the most representative document for each topic ###

# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet.head()


### Topic Distribution across documents ###

# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = (round(topic_counts/topic_counts.sum(), 4))*100

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([sent_topics_sorteddf_mallet[["Topic_Num","Keywords"]], topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Topic_Num', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics


### Pie Chart ###

import matplotlib.pyplot as plt

plt.pie(df_dominant_topics["Num_Documents"], labels = df_dominant_topics["Topic_Num"],autopct='%1.1f%%')
plt.axis('equal')
plt.show()