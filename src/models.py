import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import re
from sklearn.metrics import plot_roc_curve, confusion_matrix, classification_report
from nltk.tokenize import sent_tokenize

def clean_text(x):
    term_filter = ['eatured', 'image', 'Image', 'Getty',
        'via', 'http', 'euters', 'Via', 'Read more:']
    lst = sent_tokenize(x)
    if lst != []:
        if ' -' in lst[0]:
            lst[0] = lst[0].split(' -')[1]
        for term in term_filter:
            if term in lst[-1]:
                lst[-1] = ''
        lst = ''.join(lst)
    x = str(lst)
    x = x.replace('Reuters','')
    x = x.replace('reuters', '')
    x = x.replace('video','')
    x = x.replace('Watch:','')
    x = x.replace('twitter.com', '')
    return x

def clean_titles(x):
    x = x.replace('Factbox: ', '')
    x = re.sub(r"\(.*\)","", x)
    x = re.sub(r"\[.*\]","", x)
    x = x.replace('WATCH:', '')
    x = x.replace('BREAKING:', '')
    x = x.replace(': White House', '')
    return x

def get_X_y_splits(df, X_col, y_col='truth'):
    ''' Takes a dataframe and returns a train test split for labeled data.
    Input: DataFrame, X column name (string), target column name (string).
    Ouput: X_train, X_test, y_train, y_test'''
    X = df[X_col].values
    y = df[y_col].values
    return train_test_split(X, y)

def baseline_model(X_train, y_train):
    dummy_clf = DummyClassifier(strategy='stratified')
    dummy_clf.fit(X_train, y_train)
    return dummy_clf

def naive_bayes_model(X_train, y_train):
    bayes_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
        ])
    grid = GridSearchCV(bayes_clf, param_grid = {'vect__ngram_range': [(1,1),(1,2)]},
                        cv=5,
                        scoring='recall')
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
                        cv=5,
                        scoring='recall')
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
                        cv=5,
                        scoring='recall')
    grid.fit(X_train,y_train)
    return grid

def random_forest_model(X_train, y_train):
    rf_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
        ])
    grid = GridSearchCV(rf_clf, param_grid = {
                        'vect__ngram_range': [(1,1), (1,2)],
                        'clf__max_features': ['auto','log2']
                        },
                        cv=5,
                        scoring='recall')
    grid.fit(X_train,y_train)
    return grid
    
def plot_word_counts(X_train, fig, ax, col=None):
    cv = CountVectorizer(stop_words='english', max_features=1000, ngram_range=(1,2))
    counts = cv.fit_transform(X_train).sum(axis=0)
    features = cv.get_feature_names()
    counts_df = pd.DataFrame(index=features, data=counts.T, columns=['counts'])
    counts_df.sort_values('counts', ascending=False, inplace=True)
    ax.bar(counts_df.index[:20], counts_df['counts'][:20].values, color='blue')
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(f'Word Counts: {col}')
    ax.set_ylabel('Counts')
    fig.tight_layout()
    fig.savefig(f'images/word_counts_{col}')

def plot_word_freq_diff(df, fig, ax, col):
    fake_news = df[df['truth']==0]
    real_news = df[df['truth']==1]
    fake_cv = CountVectorizer(stop_words='english', max_features=1000, ngram_range=(1,2))
    real_cv = CountVectorizer(stop_words='english', max_features=1000, ngram_range=(1,2))
    fake_counts = fake_cv.fit_transform(fake_news[col]).sum(axis=0)
    real_counts = real_cv.fit_transform(real_news[col]).sum(axis=0)
    fake_features = fake_cv.get_feature_names()
    real_features = real_cv.get_feature_names()
    fake_counts_df = pd.DataFrame(index=fake_features, data=fake_counts.T, columns=['fake_counts'])
    real_counts_df = pd.DataFrame(index=real_features, data=real_counts.T, columns=['real_counts'])
    all_counts_df = pd.concat([real_counts_df, fake_counts_df], axis=1).fillna(0)
    all_counts_df['real_freq'] = all_counts_df['real_counts']/len(real_news)
    all_counts_df['fake_freq'] = all_counts_df['fake_counts']/len(fake_news)
    all_counts_df['freq_dif'] = all_counts_df['real_freq']-all_counts_df['fake_freq']
    all_counts_df.sort_values('freq_dif', key=abs, ascending=False, inplace=True)
    ax.bar(all_counts_df.index[:20], all_counts_df['freq_dif'][:20].values, color=(all_counts_df['freq_dif'][:20] > 0).map({True:'green',False:'red'}))
    ax.tick_params(axis='x', labelrotation=45)
    if col == 'title':
        ax.set_title(f'Term Frequency: Titles')
    if col == 'text':
        ax.set_title(f'Term Frequency: Text')
    ax.set_ylabel('Difference in Term Frequency')
    fig.tight_layout()
    fig.savefig(f'images/word_frequency_{col}')

def plot_feature_significance(model, fig, ax, col=None):
    bag = model.best_estimator_.named_steps.vect.get_feature_names()
    model_coefs = model.best_estimator_.named_steps.clf.coef_
    freq_df = pd.DataFrame(index=bag, data={'coefs': model_coefs[0]})
    freq_df = freq_df.iloc[(-freq_df['coefs'].abs()).argsort()]
    ax.bar(freq_df.index[:20], freq_df['coefs'][:20].values, color=(freq_df['coefs'][:20] > 0).map({True:'green',False:'red'}))
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_title(f'Feature Significance: {col}')
    ax.set_ylabel('Coefficients')
    fig.tight_layout()
    fig.savefig(f'images/feature_correlation_{col}')

def plot_all_roc_curves(X_test, y_test, models, titles, fig, ax, zoom=True, col=None):
    for model, title in zip(models,titles):
        plot_roc_curve(model, X_test, y_test, name=f'{title}', ax=ax)
    if zoom:
        ax.set_ylim(.8,1.01)
        ax.set_xlim(0,.2)
    ax.set_title(f'ROC Curves: {col}')
    fig.tight_layout()
    if zoom:
        fig.savefig(f'images/zoomed_roc_curves_{col}')
    else:
        fig.savefig(f'images/roc_curves_{col}')

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
    rf_model = random_forest_model(X_train, y_train)
    models = [nb_model, sgd_model, pa_model, rf_model]
    for model in models:
        print(f'{model.best_estimator_.named_steps.clf} Recall: {model.score(X_test, y_test)}')