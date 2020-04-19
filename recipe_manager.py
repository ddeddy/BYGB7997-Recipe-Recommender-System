import pandas as pd
import nltk
from nltk.corpus import stopwords


class RecipeManager:
    def __init__(self):
        self.data = pd.DataFrame()

    def read_csv(self, csv):
        self.data = pd.read_csv(csv)
        self.data.drop(self.data.columns[[2, 3, 4, 6, 7, 8, 9, 11]], axis=1)
        self.data['ingredients'] = self.data['ingredients'].apply(lambda x: x.replace('[', '').replace(']', ''))
        self.data['tags'] = self.data['tags'].apply(lambda x: x.replace('[', '').replace(']', ''))
        self.data.set_index("id", inplace=True)
        self.data.dropna(inplace=True)

    def name_str(self):
        n = self.data['name'].to_list()
        name = ",".join(n)
        return name

    def tag_str(self):
        t = self.data['tags'].to_list()
        tag = "".join(t)
        return tag

    def ingre_str(self):
        i = self.data['ingredients'].to_list()
        ingre = "".join(i)
        return ingre

    def name_keyword(self):
        name_tokens = nltk.word_tokenize(self.name_str())
        sw = stopwords.words('english')
        sw.extend(['!', ',', '.', '?', '-s', '-ly', '</s>', "'s", '', ", "])
        name_words = [w.lower() for w in name_tokens if w.isalpha() if w.lower() not in sw]
        return name_words

    def tag_keyword(self):
        tag = self.tag_str().replace("''", "','")
        lt = list(tag.split("'"))
        lt = list(set(lt))
        sw = stopwords.words('english')
        sw.extend(['!', ',', '.', '?', '-s', '-ly', '</s>', "'s", '', ", "])
        tag_words = [w.lower() for w in lt if w.lower() not in sw]
        return tag_words

    def ingre_keyword(self):
        ingre = self.ingre_str().replace(', "', "''")
        lin = list(ingre.split("\'"))
        lin = list(set(lin))
        sw = stopwords.words('english')
        sw.extend(['!', ',', '.', '?', '-s', '-ly', '</s>', "'s", '', ", "])
        ingre_words = [w.lower() for w in lin if w.lower() not in sw]
        return ingre_words

    def get_name_list(self):
        wnl = nltk.WordNetLemmatizer()
        name_stem = [wnl.lemmatize(w) for w in self.name_keyword()]
        name_list = list(set(name_stem))
        return name_list

    def get_tag_list(self):
        wnl = nltk.WordNetLemmatizer()
        tag_stem = [wnl.lemmatize(w) for w in self.tag_keyword()]
        tag_list = list(set(tag_stem))
        return tag_list

    def get_ingre_list(self):
        wnl = nltk.WordNetLemmatizer()
        ingre_stem = [wnl.lemmatize(w) for w in self.ingre_keyword()]
        ingre_list = list(set(ingre_stem))
        return ingre_list

    def get_keyword_df(self):
        wnl = nltk.WordNetLemmatizer()
        self.data["recipe_name"] = self.data["name"].map(lambda x: [wnl.lemmatize(w) for w in x.split() if w in self.name_keyword()])
        self.data["recipe_tag"] = self.data["tags"].map(lambda x: [wnl.lemmatize(w) for w in x.split("'") if w in self.tag_keyword()])
        self.data["recipe_ingre"] = self.data["ingredients"].map(
            lambda x: [wnl.lemmatize(w) for w in x.split("\'") if w in self.ingre_keyword()])
        return self.data



