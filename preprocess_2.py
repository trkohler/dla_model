import nltk
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
import en_core_web_sm

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import warnings

# Enable logging for gensim - optional
import logging
from nltk.corpus import stopwords

nltk.download("stopwords")


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.ERROR
)
warnings.filterwarnings("ignore", category=DeprecationWarning)
stop_words = stopwords.words("english")
stop_words.extend(["from", "subject", "re", "edu", "use"])

df = pd.read_json(
    "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"
)
data = df.content.values.tolist()

data = [re.sub("\S*@\S*\s?", "", sent) for sent in data]
data = [re.sub("\s+", " ", sent) for sent in data]
data = [re.sub("'", "", sent) for sent in data]

data_words = list(sent_to_words(data))

# print(data_words[:1])

bigram = gensim.models.Phrases(
    data_words, min_count=5, threshold=100
)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
# print(trigram_mod[bigram_mod[data_words[0]]])

data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)

nlp = en_core_web_sm.load()

data_lemmatized = lemmatization(
    data_words_bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
)

# print(data_lemmatized[:1])

id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized

corpus = [id2word.doc2bow(text) for text in texts]

print(corpus[:1])

lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=20,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
    per_word_topics=True,
)
lda_model.save("lda.model")
