import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def search_in_corpus(query, corpus):
    # 1. create create_tfidf_index

    tfiidf_index = {}
    documents = list(corpus.values())
    corpus = list(map(lambda doc: doc.title, documents))

    for doc in corpus:
        for term in doc:
            tfiidf_index[term] = tfiidf(term, doc, corpus)

    # 2. create query vector
    #query_vector = {}
    #for term in query:
    #    query_vector[term] = tfiidf(term, query, corpus)
#
    ## 3. calculate cosine similarity
    #results = []
    #for doc in corpus:
    #    score = 0.0
    #    for term in doc:
    #        if term in query_vector:
    #            score += tfiidf_index[term] * query_vector[term]
    #    results.append(score)
#
#
    ## 4. sort by similarity
    #results.sort(reverse=True)
#
#
    ## 5. return top 10 results
    #return results[:10]


def termFequency(term, document):
    return document.count(term) / len(document)

def inverseDocumentFrequency(term, documents):
    n = 0
    for doc in documents:
        if term.lower() in doc:
            n += 1
    return 1.0 + np.log(float(len(documents)) / n) if (n > 0) else 1.0


def tfiidf(term, document, documents):
    tf = termFequency(term, document)
    idf = inverseDocumentFrequency(term, documents)
    return tf * idf


#This function will clean our text from data that is not important so that has no weight 
def clean_text(tweet):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
   
    tweet = tweet.lower() # Transform in lowercase

    tweet = re.sub(r'@[a-zA-Z]+', '', tweet) # Here we remove the mentions in the tweet ex: @canodep
    tweet = re.sub(r"\B#([a-z0-9]{2,})(?![~!@#$%^&*()=+_`\-\|\/'\[\]\{\}]|[?.,]*\w)", '', tweet) # Here we remove the hashtags, because we will treat it later
    tweet = re.sub(r'[^\w\s]', '', tweet) # Here we remove punctuation marks
    tweet = re.sub(r'http\S+', '',tweet) # Remove http and https
    tweet = tweet.split() # Tokenize the text to get a list of terms

    tweet = [word for word in tweet if word not in stop_words]  # eliminate the stopwords
    tweet = [stemmer.stem(word) for word in tweet] # Perform stemming 
    return tweet