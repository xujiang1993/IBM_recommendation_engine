import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet', 'stopwords',
               'averaged_perceptron_tagger'])



class content_based_recommendation_engine(object):
    def __init__(self, df, df_content):
        '''
        initialise the class
        
        Input:
            df (pandas DataFrame) - data containing user item interaction information
            df_content (pandas DataFrame) - contains information regarding the items
        '''

        self.df = df
        self.df_content = df_content
        email_encoded = self.email_mapper()
        del self.df['email']
        self.df['user_id'] = email_encoded
        
        

    def email_mapper(self):
        coded_dict = dict()
        cter = 1
        email_encoded = []

        for val in self.df['email']:
            if val not in coded_dict:
                coded_dict[val] = cter
                cter+=1

            email_encoded.append(coded_dict[val])
        return email_encoded
    

    def get_article_names(self, article_ids, df):
        '''
        INPUT:
        article_ids - (list) a list of article ids
        df - (pandas dataframe) df as defined at the top of the notebook

        OUTPUT:
        article_names - (list) a list of article names associated with the list of article ids 
                        (this is identified by the title column)
        '''
        article_names = []
        for article_id in article_ids:
            article_idx = np.where(df['article_id'] == float(article_id))[0][0]
            article_names.append(df.iloc[article_idx]['title'])

        return article_names # Return the article names associated with list of article ids


    def tokenize(self,text):
        """
        This function is used to preprocess the text data into tokenized lists

        Input:
            text data (str)

        Output:
            tokens (list) a list of words

        """
        # Remove punctuation and lower case
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

        # lemmatize and remove stop words
        tokens = word_tokenize(text)
        stop_words = stopwords.words("english")
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens
                  if word not in stop_words]
        # remove short words
        tokens = [token for token in tokens if len(token) > 2]

        return tokens

    def make_content_recs(self,user_input, m=10,sim_threshold=0.5):
        '''
        This function is used to make recommendation based on the content similarity
        INPUT:
        user_id (float) - input user id
        m (int) - the number of the recommendations
        sim_threshold (float from 0 to 1) - this is the threhold for the content similarity
        OUTPUT:
        recs (list) - recommended article id
        rec_names (list) - the recommended article names
        '''
        corpus = []
        recs = []
        article_sim_df = []
        df_1 = self.df[['article_id', 'title']]
        df_2 = self.df_content[['article_id','doc_full_name']]
        df_2 = df_2.rename(columns={"doc_full_name": "title"})
        df_total = pd.concat([df_1, df_2], ignore_index=True)
        df_total.drop_duplicates(subset=['article_id'], inplace=True)
        df_total.sort_values(by='article_id', inplace=True)
        df_total.reset_index(drop=True, inplace=True)

        for i in range(len(df_total)):
            corpus_vector = df_total['title'][i]
            corpus.append(corpus_vector)
        vect = TfidfVectorizer(tokenizer=self.tokenize)
        Tfidf_vec = vect.fit_transform(corpus)


        user_Tfidf_vec = vect.transform(user_input)

        articles_sim = user_Tfidf_vec.dot(np.transpose(Tfidf_vec)).toarray()
        article_sim_df = pd.DataFrame(np.transpose(articles_sim),
                                      index=df_total['article_id'],
                                      columns=['article'])

        article_sim_df.sort_values(by=['article'],ascending=False, inplace=True)
        article_sim_df = article_sim_df[article_sim_df >= 0.2]
        recs = list(article_sim_df['article'].index)
        rec_names = self.get_article_names(recs, df=df_total)

        return recs[:m], rec_names[:m]

