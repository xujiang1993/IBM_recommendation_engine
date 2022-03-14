import json
import plotly
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Histogram
from APP import app




from model import content_based_recommendation_engine
nltk.download(['punkt', 'wordnet', 'stopwords',
               'averaged_perceptron_tagger'])



def tokenize(text):
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

# load data
df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']
print('Finished loading data...')
# load model
recommendation_eng = content_based_recommendation_engine.content_based_recommendation_engine(df, df_content)
print('Finished loading model')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    user_interaction_counts = df.groupby('user_id').count()['title']
    content_duplicate_counts = [df_content.duplicated(col).sum() for col in df_content.columns]
    article_counts = df['article_id'].value_counts()
    popular_articles = article_counts.sort_values(ascending=False) #op 30 articles
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Histogram(x=user_interaction_counts, xbins=dict(start=1, end=85, size=5))
            ],

            'layout': {
                'title': 'User Activity Distribution',
                'yaxis': {
                    'title': "<b>Occurences</b>"
                },
                'xaxis': {
                    'title': "<b># Interactions</b>"
                }
            }
        },
        {
            'data': [
                Histogram(x=article_counts)
            ],

            'layout': {
                'title': 'Distribution of Article Interactions',
                'yaxis': {
                    'title': "<b># of Articles</b>"
                },
                'xaxis': {
                    'title': "<b># Interactions</b>",
                    'tickangle': 30
                    
                }
                
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    
    user_id, arts_titles = recommendation_eng.make_content_recs(user_input = [query], m=10, sim_threshold=0.2)
    classification_results = dict(zip(user_id, arts_titles))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


#def main():
#    app.run(host='0.0.0.0', port=3000, debug=True)


#if __name__ == '__main__':
#    main()