from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
import nltk
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from pandarallel.pandarallel import NB_WORKERS
import pandas as pd
from pprint import pprint
from pandarallel import pandarallel
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import get_tmpfile
from gensim.corpora import MalletCorpus

# spacy for lemmatization
import en_core_web_sm

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import warnings

# Enable logging for gensim - optional
import logging
from nltk.corpus import stopwords


# CONSTANTS
NB_OF_WORKERS = 3
USE_MEMORY_FS = None # change this to True if you're on linux!
TOPICS_NUMBER = 100
mallet_path = "/Users/troykohler/Downloads/mallet-2.0.8/bin/mallet"

nltk.download("stopwords")
nlp = en_core_web_sm.load()
pandarallel.initialize(nb_workers=NB_OF_WORKERS, progress_bar=True, use_memory_fs=USE_MEMORY_FS)
stop_words = stopwords.words("english")
stop_words.extend(["from", "subject", "re", "edu", "use"])
lemmatizer = WordNetLemmatizer() 
stop_words = stopwords.words("english")
stop_words.extend(
    [
        "from", "subject", "re", "edu", "use", "http", 
        "shit", "fuck", "fucking", "gt", "nt", "na", "op", "ocd", 
        "lol", "ca", "pm", "meme", "dick"
    ]
)


def sent_to_words(sentence):
    return gensim.utils.simple_preprocess(str(sentence), deacc=True)


def remove_stopwords(text):
    return [word for word in simple_preprocess(str(text)) if word not in stop_words]


def lemmatization(text):
    return [lemmatizer.lemmatize(word) for word in text]

def make_bigrams(comment_tokenized):
    bigram = gensim.models.Phrases(comment_tokenized, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod

def find_bigrams(input_list):
    lst = list(zip(*[input_list[i:] for i in range(2)]))
    lst_str = [" ".join(tpl) for tpl in lst]
    [lst_str.append(word) for word in input_list]
    return lst_str

def find_trigrams(input_list):
    return list(zip(*[input_list[i:] for i in range(3)]))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.ERROR
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    df = pd.read_csv("./reddit_dataset.csv")

    
    df['text_new'] = df['text'].str.replace("\S*@\S*\s?\s+\'", "", regex=True)
    df['text_new'] = df['text_new'].parallel_apply(sent_to_words)
    print(df['text_new'].head(3))


    df['no_stop_words'] = df['text_new'].parallel_apply(remove_stopwords)
    print(df['no_stop_words'].head(5))

    # with alive_bar(1, spinner="pointer") as bar:
    #     df['bigrams'] = df['no_stop_words'].parallel_apply(find_bigrams)
    #     df['bigrams'] = gensim.models.Phrases(df['bigrams'], min_count=5, threshold=100)  # higher threshold fewer phrases.
    #     df['bigrams'] = gensim.models.phrases.Phraser(df['bigrams'])
    #     bar()

    # print(df['bigrams'].head(5))

   
    df['lemmatized'] = df['no_stop_words'].parallel_apply(lemmatization)
    print(df['lemmatized'].head(15))

    dictionary = corpora.Dictionary(df['lemmatized'])
    dictionary.filter_extremes(no_below=10, no_above=0.1)

    df['doc_to_bowed'] = df['lemmatized'].parallel_apply(dictionary.doc2bow)
    print(df['doc_to_bowed'].head(5))

    corpus = df['doc_to_bowed']

    ldamallet = gensim.models.wrappers.LdaMallet(
        mallet_path, corpus=corpus, num_topics=TOPICS_NUMBER, id2word=dictionary
    )
    lda_native = malletmodel2ldamodel(ldamallet)
    lda_native.save("model/mallet_to_native.lda")

    corpora.Dictionary.save(dictionary, "model/dictionary.dict")
    corpora.BleiCorpus.save_corpus(fname="model/corpus.lda-c", corpus=corpus)

    pprint(lda_native.show_topics(20))

    # MIGHT BE USEFUL, SO DONT REMOVE IT
    # bigram = gensim.models.Phrases(
    #     df['bigrams'], min_count=5, threshold=100
    # )  # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[df['text_new']], threshold=100)

    # # Faster way to get a sentence clubbed as a trigram/bigram
    # bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)

    # # See trigram example
    # print(trigram_mod[bigram_mod[df['text_new'][0]]])

    # data_words_nostops = remove_stopwords(data_words)
    # data_words_bigrams = make_bigrams(data_words_nostops)


    # data_lemmatized = lemmatization(
    #     data_words_bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
    # )

    # print(data_lemmatized[:1])

    # dictionary = corpora.Dictionary(data_lemmatized)
    # dictionary.filter_extremes(no_below=10, no_above=0.1)
    # texts = data_lemmatized

    # corpus = [dictionary.doc2bow(text) for text in texts]

    # print(corpus[:1])

    # mallet_path = "/Users/troykohler/Downloads/mallet-2.0.8/bin/mallet"
    # ldamallet = gensim.models.wrappers.LdaMallet(
    #     mallet_path, corpus=corpus, num_topics=20, id2word=dictionary
    # )
    # lda_native = malletmodel2ldamodel(ldamallet)
    # lda_native.save("model/mallet_to_native.lda")

    # corpora.Dictionary.save(dictionary, "model/dictionary.dict")
    # corpora.BleiCorpus.save_corpus(fname="model/corpus.lda-c", corpus=corpus)
    # bigram.save("model/bigram.phs")
    # trigram.save("model/trigram.phs")
