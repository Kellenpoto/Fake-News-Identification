import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import plot_roc_curve, confusion_matrix, classification_report

fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')
fake_df['truth'] = 0
true_df['truth'] = 1
all_news_df = pd.concat([fake_df, true_df])

def get_X_y_splits(df, X_col, y_col='truth'):
    ''' Takes a dataframe and returns a train test split for labeled data.
    Input: DataFrame, X column name (string), target column name (string).
    Ouput: X_train, X_test, y_train, y_test'''
    X = df[X_col].values
    y = df[y_col].values
    return train_test_split(X, y)

X_train, X_test, y_train, y_test = get_X_y_splits(all_news_df, 'title')

def baseline_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
    dummy_clf = DummyClassifier()
    dummy_clf.fit(X_train, X_test)
    return dummy_clf

def naive_bayes_model(X_train=X_train, y_train=y_train):
    bayes_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB())
        ])
    grid = GridSearchCV(bayes_clf, param_grid = {'vect__ngram_range': [(1,1),(1,2)]},
                        cv=5,
                        refit=True)
    grid.fit(X_train, y_train)
    return grid

def stochastic_gradient_descent_model(X_train=X_train, y_train=y_train):
    sgd_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', SGDClassifier())
        ])
    grid = GridSearchCV(sgd_clf, param_grid = {
                        'vect__ngram_range': [(1,1), (1,2)],
                        'clf__alpha': [.001, .0005, .0001]
                        },
                        cv=5,
                        refit=True)
    grid.fit(X_train,y_train)
    return grid

def passive_aggressive_model(X_train=X_train, y_train=y_train):
    pa_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', PassiveAggressiveClassifier())
        ])
    grid = GridSearchCV(pa_clf, param_grid = {
                        'vect__ngram_range': [(1,1), (1,2)],
                        'clf__C': [1.0, 1.5, 2.0]
                        },
                        cv=5,
                        refit=True)
    grid.fit(X_train,y_train)
    return grid

def random_forest_model(X_train=X_train, y_train=y_train):
    rf_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', PassiveAggressiveClassifier())
        ])
    grid = GridSearchCV(rf_clf, param_grid = {
                        'vect__ngram_range': [(1,1), (1,2)],
                        'max_features': ['auto','log2']
                        },
                        cv=5,
                        refit=True)
    grid.fit(X_train,y_train)
    return grid

if __name__ == '__main__':
    nb_model = naive_bayes_model()
    sgd_model = stochastic_gradient_descent_model()
    pa_model = passive_aggressive_model()
    models = [nb_model, sgd_model, pa_model]
    for model in models:
        print(f'{model} accuracy: {model.score(X_test, y_test)}')