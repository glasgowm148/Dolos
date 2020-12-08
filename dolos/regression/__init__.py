from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import os
import numpy as np

porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def main(keywords):
    path = os.path.dirname(__file__)
    path = os.path.join(path, '../data/corona_fake.csv')

    df = pd.read_csv(path)

    df.loc[df['label'] == 'Fake', ['label']] = 'FAKE'
    df.loc[df['label'] == 'fake', ['label']] = 'FAKE'
    df.loc[df['source'] == 'facebook', ['source']] = 'Facebook'
    df.text.fillna(df.title, inplace=True)

    df.loc[5]['label'] = 'FAKE'
    df.loc[15]['label'] = 'TRUE'
    df.loc[43]['label'] = 'FAKE'
    df.loc[131]['label'] = 'TRUE'
    df.loc[242]['label'] = 'FAKE'

    df = df.sample(frac=1).reset_index(drop=True)
    df.title.fillna('missing', inplace=True)
    df.source.fillna('missing', inplace=True)

    df['title_text'] = df['title'] + ' ' + df['text']

    print(df['label'].value_counts())

    
    tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None,
                        #tokenizer=tokenizer_porter,
                        use_idf=True,
                        norm='l2',
                         smooth_idf=True)
    X = tfidf.fit_transform(df['title_text'])
    y = df.label.values

    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.5, shuffle=False)

    #clf = LogisticRegressionCV(cv=5, scoring='accuracy', random_state=0, n_jobs=-1, verbose=3, max_iter=300).fit(X_train, y_train)

    #fake_news_model = open('/Users/pseudo/Documents/GitHub/hci-ae/cogs/dolos/regression/fake_news_model.sav', 'wb')
    #pickle.dump(clf, fake_news_model)
    #fake_news_model.close()
    path = os.path.dirname(__file__)
    path = os.path.join(path, '../data/fake_news_model.sav')
    saved_clf = pickle.load(open(path, 'rb'))

    #saved_clf.score(X_test, y_test)
    #saved_clf.score(X_test, keywords)

    #y_pred = clf.predict(X_test)
    #print("---Test Set Results---")
    #print("Accuracy with logreg: {}".format(accuracy_score(y_test, y_pred)))
    #print(classification_report(y_test, y_pred))
    
    vectorized_text = tfidf.transform(keywords)#.toarray()
    
    pred = saved_clf.predict(vectorized_text)

    #print(accuracy_score(y_test.transform(pred)))

    unique, counts = np.unique(pred, return_counts=True)

    t = dict(zip(unique, counts))
    print(t)
    return t['TRUE'] / t['FAKE'] 