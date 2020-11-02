from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
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
from gensim.test.utils import get_tmpfile
from gensim.corpora import MalletCorpus

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
    stop_words = stopwords.words("english")
    stop_words.extend(["from", "subject", "re", "edu", "use"])
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = en_core_web_sm.load()
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.ERROR
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    stop_words = stopwords.words("english")
    stop_words.extend(["from", "subject", "re", "edu", "use", "http"])

    # df = pd.read_json(
    #     "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"
    # )
    df = pd.read_csv("./reddit_dataset.csv")
    data = df["text"].tolist()

    data = [re.sub("\S*@\S*\s?", "", sent) for sent in data]
    data = [re.sub("\s+", " ", sent) for sent in data]
    data = [re.sub("'", "", sent) for sent in data]

    data_words = list(sent_to_words(data))

    print(data_words[:1])

    bigram = gensim.models.Phrases(
        data_words, min_count=5, threshold=100
    )  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    print(trigram_mod[bigram_mod[data_words[0]]])

    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostops)

    nlp = en_core_web_sm.load()

    data_lemmatized = lemmatization(
        data_words_bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
    )

    print(data_lemmatized[:1])

    dictionary = corpora.Dictionary(data_lemmatized)
    dictionary.filter_extremes(no_below=10, no_above=0.1)
    texts = data_lemmatized

    corpus = [dictionary.doc2bow(text) for text in texts]

    print(corpus[:1])

    mallet_path = "/Users/troykohler/Downloads/mallet-2.0.8/bin/mallet"
    ldamallet = gensim.models.wrappers.LdaMallet(
        mallet_path, corpus=corpus, num_topics=20, id2word=dictionary
    )
    lda_native = malletmodel2ldamodel(ldamallet)
    lda_native.save("model/mallet_to_native.lda")

    corpora.Dictionary.save(dictionary, "model/dictionary.dict")
    corpora.BleiCorpus.save_corpus(fname="model/corpus.lda-c", corpus=corpus)
    bigram.save("model/bigram.phs")
    trigram.save("model/trigram.phs")
