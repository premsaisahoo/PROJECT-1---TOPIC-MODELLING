# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 08:59:32 2021

@author: Ranjeet
"""

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd # not used in gnews
import urllib3
from urllib.request import Request,urlopen
import html5lib
import streamlit as st
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import spacy
nlp = spacy.load("en_core_web_sm")
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import collections
import plotly.express as px





#################### Web Scraping ####################


def news(link):
    req = Request(link,headers={'User-Agent':'Chrome/91.0.4472.114'})
    webpage = urlopen(req).read()
    with requests.Session() as c:
        soup = bs(webpage,'html5lib')
        # print(soup.encode("utf-8"))
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
            document = open("data.csv","a",encoding="utf-8") #encoding="utf-8"
            document.write("{},{},{},{} \n".format(title, time, descript, link))
            document.close()
        next = soup.find('a',attrs={'aria-label':'Next page'})
        next = (next['href'])
        link = root + next
        news(link)

#################### Data Cleaning ####################

# Drop duplicates

@st.cache
def drop_duplicate(news):
    news = news.drop_duplicates(subset = None, keep = 'first', inplace = False)
    return news

#### Lowercase, Removal - numbers,puctiations ####

EMAIL_REGEX_STR = '\S*@\S*'
MENTION_REGEX_STR = '@\S*'
HASHTAG_REGEX_STR = '#\S+'
URL_REGEX_STR = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'

remove_regex = re.compile(f'({EMAIL_REGEX_STR}|{MENTION_REGEX_STR}|{HASHTAG_REGEX_STR}|{URL_REGEX_STR})')

@st.cache
def clean_text(text):
 ''', , and '''
#Make text lowercase   
 text = text.lower()
#remove text in square brackets
 text = re.sub(r'\[.*?\]', '', text)
#remove punctuation   
 text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
 text = re.sub(r'�',"",text)
# Defined function
 text = re.sub(remove_regex, '', text)
#remove words containing numbers
 text = re.sub(r'\w*\d\w*', '', text)
 return text

@st.cache
def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

#### Removal - Short words ####

@st.cache
def remove_short_word(text):
    text = ' '.join([w for w in text.split() if len(w) > 3])
    return text


#### Removal - Stopwords ####

stop_words = stopwords.words('english')
@st.cache
def remove_stop_word(text):
    text = [word for word in text if not word in stop_words]
    return text

#### Lemmatization ####


def lemmatize(text):
    doc = nlp(' '.join(text))
    lemma_news = [token.lemma_ for token in doc]
    return lemma_news



#################### EDA ####################

#### Unigram ####

@st.cache
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


### Trigram ###

@st.cache
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3,3) , stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]



####### Model Building #####



def train_model(data_lemmatized,num_topics: int = 10):
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, chunksize=100, passes=10, per_word_topics=True)
    return lda_model



#################### Streamlit ####################

if __name__ == '__main__':
    
    
    link = None
    root = "https://www.google.com/"
    
    #with st.sidebar:
         #num_topics = st.number_input('Number of Topics', min_value=1,max_value=50,value=10)
    
    st.title("Excited to Get the Trending Topics?")
    
    link = st.text_input('')
    
    
    if link:
        try:
            news(link)
        except TypeError:
            print("Scraping complete")
        st.markdown("Scraping Done")
        
    else:
        st.markdown("ENTER LINK")
    
    data = None
    final = None
    
    # st.checkbox("Scraping Complete")
    
    num_topics = st.number_input('Number of Topics', min_value=1,max_value=50,value=10)
    
    check = st.checkbox("Get Topics")
    
    if check: 
        
        col_names = ['Title','Time','Description','Link','Unnec']
        
        
        data = pd.read_csv("C:\\Users\\Ranjeet\\myenv\\data.csv", names=col_names) #encoding="utf-8"
        data[0:10]
        data['news'] = data['Title'] + data['Description']
        
        final = data.loc[:,['news']]
        # final.news[0:10]
        
        # Drop duplicates
        
        drop_duplicate(final)
        # final[0:10]
        
        w = [news.strip() for news in final.news] # removes leading and trailing characters
        w = [news for news in w if news] # removes empty strings
        
        # Clean text
        news = map(clean_text, w) # map command used for lsit
        news = list(news)
        
        # Remove punctuation
        news_a = map(remove_punctuation,news)
        news_a = list(news_a)
        
        # Remove shortword
        news1 = map(remove_short_word, news_a) 
        news1 = list(news1)
        
        # Tokenize
        news2 = map(word_tokenize,news1)
        news2 = list(news2)
        
        # Remove stop word
        news3 = map(remove_stop_word,news2)
        news3 = list(news3)
        
        # Lemmatization
        news4 = map(lemmatize,news3)
        news4 = list(news4)
        
        # Remove stop word
        news4 = map(remove_stop_word,news4)
        news4 = list(news4)
        
        #################### EDA ####################
        
        ### feature extraction ###
        
        vectorizer = CountVectorizer()

        clean_news = [' '.join(ele) for ele in news4]
        clean_news = [news for news in clean_news if news]


        X = vectorizer.fit_transform(clean_news)
        m = vectorizer.vocabulary_ 
        
        #### News Length ####
        
        with st.beta_expander('Document Word Count Distribution', expanded = True):
            doc_lens = [len(d) for d in clean_news]
            fig, ax = plt.subplots()
            sns.histplot(pd.DataFrame(doc_lens,columns=['Words In Document']), ax=ax)
            st.pyplot(fig)
        
        ### WordCloud ###
        
        with st.beta_expander('WordCloud',expanded = True):
            stopwords = STOPWORDS
            wc = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(str(clean_news))
            st.image(wc.to_image(), caption='Dataset Wordcloud', use_column_width=True)
        
        ### Unigram ###
        
        with st.beta_expander('Unigram',expanded = True):
            common_words = get_top_n_words(clean_news, 10)
            unigram = pd.DataFrame(common_words, columns = ['unigram' , 'count'])
            st.write(unigram)
        
        ### Trigram ###
        
        with st.beta_expander('Trigram',expanded = True):
            common_words = get_top_n_trigram(clean_news, 10)
            trigram = pd.DataFrame(common_words, columns = ['trigram' , 'count'])
            st.write(trigram) 
    
        ### Model Building ###
        
        data_lemmatized = news4
        id2word = corpora.Dictionary(data_lemmatized)
        texts = data_lemmatized
        corpus = [id2word.doc2bow(text) for text in texts]
        
        lda_model = None
        
        
        lda_model = train_model(data_lemmatized,num_topics)
        
        st.header('Model')
        
        if lda_model:
            st.write(lda_model)
        
            
            ### Model Results ###
            
            st.header('Model Results')
            if lda_model:
                topics = lda_model.print_topics()
                st.subheader('Topic Word-Weighted Summaries')
                for topic in topics:
                    st.markdown(f'**Topic #{topic[0]}**: _{topic[1]}_')
            
            ### Finding Dominant Topics ###
            
            
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
            
            df_topic_sents_keywords= pd.DataFrame()
            try:
                df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=news1)
            except TypeError:
                print('complete')
            
            
            df_dominant_topic = None 
            df_dominant_topic = pd.DataFrame()
            df_dominant_topic = df_topic_sents_keywords.reset_index()
            df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
           
            
            with st.beta_expander('Dominant Topic',expanded = True):
                st.write(df_dominant_topic)
            
            ### Finding the most representative document for each topic ###
            
            sent_topics_sorteddf_mallet = None
            sent_topics_outdf_grpd = None
            
            sent_topics_sorteddf_mallet = pd.DataFrame()
            sent_topics_outdf_grpd = pd.DataFrame()
            
            sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

            for i, grp in sent_topics_outdf_grpd:
                sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                         grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                                        axis=0)
                                                        
            # Reset Index    
            sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
            
            # Format
            sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
            
            with st.beta_expander('Most Representative Documents', expanded= True):
                st.write(sent_topics_sorteddf_mallet)
            
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
            
            with st.beta_expander('Topic - Document Summary', expanded= True):
                st.write(df_dominant_topics)
            
            ### Pie Chart ###
            
            fig = px.pie(df_dominant_topics, values='Perc_Documents', names='Topic_Num',
                     title='Topic Distribution Across Documents',
                     hover_data=['Num_Documents'], labels={'Topic_Num':'Perc_Documents'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            # fig.show() # opens up in new tab
            st.write(fig)
        
        else:
            st.markdown('No model trained')
    else:
        st.markdown("No Data")