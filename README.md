# IBM_recommendation_engine
This project has designed to make recommendations to users though analysing the IBM Watson Studio platform data.

This project contains five subsections:
* Exploratory Data Analysis: this section is to explore and preprocess the IBM Watson Studio platform data
* Rank Based Recommendations: This part build up a basic recommendation engine based the popularity of articles. This engine can be used for cold start problem.
* User-User Based Collaborative Filtering: this part develops a user based recommendation engine which articles with the most similar user's review list.
* Content Based Recommendations: this section develops a recommendation engine based on the article title and description. This engine uses Natural Language Processing (NLP) to characterise the articles then find the text-based similarity with the feature vectors.
* Matrix Factorization: This section has tested the effectiveness of SVD method on recommendation engine

## Data
This data of this project is sponsored by IBM including:
* user-item-interactions: this data contains messages of interactions between users and articles
* articles_community: this data contais the basic information of the articles, including title, description and content etc.

## Installation
This work uses python and the used python packages are given below:
* Pandas
* numpy
* matplotlib
* pickle
* re
* sklearn
* nltk

