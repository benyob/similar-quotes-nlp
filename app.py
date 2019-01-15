import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('./input/mini-quotes.csv', sep=';')
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df['quote'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
results = {}

for idx, row in df.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-20:-1]
    similar_items =[(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices]
    results[row['id']] = similar_items[1:]

def getQuote(id):
    return df.loc[df['id'] == id]['quote'].tolist()[0]

def recommend(quote_id, num):
    print( str(num) + " Similar quotes to : '"  + getQuote(quote_id) + "'...")
    print("###################################################################")
    recs = results[quote_id][:num]
    for rec in recs:
        print("*) " + getQuote(rec[1]) + " (score:" + str(rec[0]) + ")")


recommend(10,4)
