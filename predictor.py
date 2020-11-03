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
    def __init__(self, lda_path, dict_path):
        """
        lda_path - path to lda model
        dict_path - path to dict

        param: lda_path str
        param: dict_path str
        """
        self.dictionary = corpora.Dictionary.load(dict_path)
        self.lda = LdaMulticore.load(lda_path)

    def clean(self, text):
        text = regex.sub("", text)
        text = [token for token in text.split()]
        text = [stemmer.stem(token) for token in text]
        text = [token for token in text if token]
        return " ".join(text)

    def show_topics(self):
        pprint(self.lda.print_topics(num_words=30))


    def predict(self, text):
        clean_text = self.clean(text).split()
        new_review_bow = self.dictionary.doc2bow(clean_text)
        new_review_lda = self.lda[new_review_bow]
        return sorted(new_review_lda, reverse=True, key=operator.itemgetter(1))


lda_path = "model/mallet_to_native.lda"
dict_path = "model/dictionary.dict"
lda = LdaPredictor(lda_path, dict_path)
text = "I didn't have particular aspirations"
predict = lda.predict(text)
print(predict)

# lda.show_topics()
