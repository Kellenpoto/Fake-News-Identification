import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.dummy import DummyClassifier
import re
from sklearn.metrics import plot_roc_curve, confusion_matrix, classification_report
from nltk.tokenize import sent_tokenize

def clean_text(x):
    lst = sent_tokenize(x)
    if lst != [] and ' -' in lst[0]:
        lst[0] = lst[0].split(' - ')[1]
    return ''.join(lst)

def clean_titles(x):
    x = x.replace('Factbox: ', '')
    x = re.sub(r"\(.*\)","", x)
    x = x.replace('WATCH:', '')
    return x

def get_X_y_splits(df, X_col, y_col='truth'):
    ''' Takes a dataframe and returns a train test split for labeled data.
    Input: DataFrame, X column name (string), target column name (string).
    Ouput: X_train, X_test, y_train, y_test'''
    X = df[X_col].values
    y = df[y_col].values
    return train_test_split(X, y)

def baseline_model(X_train, y_train):
    dummy_clf = DummyClassifier()
    dummy_clf.fit(X_train, y_train)
    return dummy_clf

def naive_bayes_model(X_train, y_train):
    bayes_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
        ])
    grid = GridSearchCV(bayes_clf, param_grid = {'vect__ngram_range': [(1,1),(1,2)]},
                        cv=5,)
    grid.fit(X_train, y_train)
    return grid

def stochastic_gradient_descent_model(X_train, y_train):
    sgd_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier())
        ])
    grid = GridSearchCV(sgd_clf, param_grid = {
                        'vect__ngram_range': [(1,1), (1,2)],
                        'clf__alpha': [.001, .0005, .0001]
                        },
                        cv=5,)
    grid.fit(X_train,y_train)
    return grid

def passive_aggressive_model(X_train, y_train):
    pa_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', PassiveAggressiveClassifier())
        ])
    grid = GridSearchCV(pa_clf, param_grid = {
                        'vect__ngram_range': [(1,1), (1,2)],
                        'clf__C': [1.0, 2.0, 3.0]
                        },
                        cv=5,)
    grid.fit(X_train,y_train)
    return grid

def random_forest_model(X_train, y_train):
    rf_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', PassiveAggressiveClassifier())
        ])
    grid = GridSearchCV(rf_clf, param_grid = {
                        'vect__ngram_range': [(1,1), (1,2)],
                        'max_features': ['auto','log2']
                        },
                        cv=5,)
    grid.fit(X_train,y_train)
    return grid

if __name__ == '__main__':
    fake_df = pd.read_csv('data/Fake.csv')
    true_df = pd.read_csv('data/True.csv')
    fake_df['truth'] = 0
    true_df['truth'] = 1

    true_df['text'] = true_df['text'].apply(clean_text)
    all_news_df = pd.concat([fake_df, true_df])
    all_news_df['title'] = all_news_df['title'].apply(clean_titles)

    X_train, X_test, y_train, y_test = get_X_y_splits(all_news_df, 'title')
    nb_model = naive_bayes_model(X_train,y_train)
    sgd_model = stochastic_gradient_descent_model(X_train,y_train)
    pa_model = passive_aggressive_model(X_train,y_train)
    models = [nb_model, sgd_model, pa_model]
    for model in models:
        print(f'{model.best_estimator_.named_steps.clf} Accuracy: {model.score(X_test, y_test)}')