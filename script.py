import pandas as pd
import pickle
import string


def preprocess(text):
    text = text.lower()
    text = "".join(c if c not in string.punctuation else f" {c} " for c in text)
    return " ".join(w.strip() for w in text.split())


test = pd.read_parquet('data/task1_test_for_user.parquet')
test.item_name = test.item_name.apply(preprocess)

tfidf = pickle.load(open('count_vec_baseline', 'rb'))
clf = pickle.load(open('clf_baseline', 'rb'))

X_test = tfidf.transform(test.item_name)

pred = clf.predict(X_test)

res = pd.DataFrame(pred, columns=['pred'])
res['id'] = test['id']

res[['id', 'pred']].to_csv('answers.csv', index=None)
