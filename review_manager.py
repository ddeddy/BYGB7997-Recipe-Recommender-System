import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from math import log
from collections import Counter
import warnings


class ReviewManager:
    def __init__(self):
        self.data = pd.DataFrame()

    def read_csv(self, csv):
        self.data = pd.read_csv(csv)
        self.data.dropna(inplace=True)

    def read_dataframe(self, df):
        self.data = df

    def tokenize(self, text):
        tokens = nltk.word_tokenize(text.lower())
        nonstop = [i for i in tokens if i not in self.stopwords_]
        stems = [PorterStemmer().stem(x) for x in nonstop if x.isalpha() is True]
        return stems

    def tokenize_review(self):
        self.stopwords_ = stopwords.words('english')
        self.stopwords_.extend(['!', ',', '.', '?', '-s', '-ly', '</s>', "'s"])
        self.data["tokens"] = self.data["review"].map(self.tokenize)

    def calculate_document_frequency(self, inverse=True):
        warnings.warn("Note! Long Running Time")
        tokens_list = self.data["tokens"].to_list()[0].copy()
        for i in self.data["tokens"].to_list()[1:]:
            tokens_list.extend(i)
        tokens_list = list(set(tokens_list))
        df_dict = dict(zip(tokens_list, [0 for i in range(len(tokens_list))]))
        for sent in self.data["tokens"].to_list():
            for token in [i for i in tokens_list if i in sent]:
                df_dict[token] += 1
        if inverse == False: return df_dict
        total = self.data.shape[0]
        idf_dict = dict(zip(tokens_list, [log(total / (i + 1)) for i in df_dict.values()]))
        return idf_dict

    def calculate_tfidf(self, sentence):
        tfidf_dict = {k: v * self.idf_dict[k]
                      for k, v in dict(Counter(sentence)).items()
                      if k in self.idf_dict}
        return tfidf_dict

    def get_keywords_sentence(self, tokens, threhold=5, low_bound=3):
        tfidf_dict = self.calculate_tfidf(tokens)
        keywords = [token for token in tfidf_dict.keys() if tfidf_dict[token] > threhold]
        while len(keywords) == 0:
            if threhold == low_bound: break
            threhold += (-0.5)
            keywords = [token for token in tfidf_dict.keys() if tfidf_dict[token] > threhold]
        return keywords

    def get_keywords_total(self, verbose=False):
        self.idf_dict = self.calculate_document_frequency()
        self.data["keywords"] = self.data["tokens"].map(self.get_keywords_sentence)
        keyword_df = pd.DataFrame(self.data.set_index("recipe_id")["keywords"].groupby("recipe_id").sum())
        if verbose == True: return keyword_df
        keywords_distinct = keyword_df["keywords"].to_list()[0].copy()
        for i in keyword_df["keywords"].to_list()[1:]:
            keywords_distinct.extend(i)
        return list(set(keywords_distinct))

    def fast_keywords(self, source, source_type="csv"):
        if source_type == "csv":
            self.read_csv(source)
        if source_type == "dataframe":
            self.read_dataframe(source)
        self.tokenize_review()
        return self.get_keywords_total()