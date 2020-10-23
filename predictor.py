from gensim.models import LdaMulticore
from gensim import corpora
from gensim.models.phrases import Phrases
from nltk.corpus import stopwords
import re, string
from nltk.stem.snowball import SnowballStemmer
import operator
from pprint import pprint

stemmer = SnowballStemmer("english", ignore_stopwords=True)
regex = re.compile("[%s]" % re.escape(string.punctuation))


class LdaPredictor:
    def __init__(self, lda_path, dict_path, bigram_path, trigram_path):
        """
        lda_path - path to lda model
        dict_path - path to dict
        bigram_path - path to bigrams
        trigram_path - path to trigrams

        param: lda_path str
        param: dict_path str
        param: bigram_path str
        param: trigram_path str
        """
        self.dictionary = corpora.Dictionary.load(dict_path)
        self.lda = LdaMulticore.load(lda_path)
        self.bigram_path = bigram_path
        self.trigram_path = trigram_path

    def clean(self, text):
        text = regex.sub("", text)
        text = [token for token in text.split()]
        text = [stemmer.stem(token) for token in text]
        text = [token for token in text if token]
        return " ".join(text)

    def show_topics(self):
        pprint(self.lda.print_topics(num_words=20))

    def bigram(self, text):
        bigram = Phrases.load(self.bigram_path)
        trigram = Phrases.load(self.trigram_path)
        text_clean = text
        for idx in range(len(text_clean)):
            for token in bigram[text_clean[idx]]:
                if "_" in token:
                    text_clean[idx].append(token)
            for token in trigram[text_clean[idx]]:
                if "_" in token:
                    text_clean[idx].append(token)
        return text_clean

    def predict(self, text):
        clean_text = self.clean(text).split()
        bigram = self.bigram([clean_text])
        new_review_bow = self.dictionary.doc2bow(bigram[0])
        new_review_lda = self.lda[new_review_bow]
        return sorted(new_review_lda, reverse=True, key=operator.itemgetter(1))


lda_path = "./model/mallet_model.lda"
dict_path = "./model/dictionary.dict"
bigram_path = "./model/bigram.phs"
trigram_path = "./model/trigram.phs"
lda = LdaPredictor(lda_path, dict_path, bigram_path, trigram_path)
text = "A church building or church house, often simply called a church, is a building used for Christian religious activities, particularly for Christian worship services. The term is often used by Christians to refer to the physical buildings where they worship, but it is sometimes used as an analogy to refer to buildings of other religions.[1] In traditional Christian architecture, a church interior is often structured in the shape of a Christian cross. When viewed from plan view the vertical beam of the cross is represented by the center aisle and seating while the horizontal beam and junction of the cross is formed by the bema and altar."
predict = lda.predict(text)
print(predict)

lda.show_topics()
