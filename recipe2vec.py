from recipe_manager import RecipeManager

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class Recipe2Vec:
    def __init__(self):
        self.re = RecipeManager()
        self.keywords_name = list()
        self.keywords_tag = list()
        self.keywords_ingre = list()
        self.keywords_df = pd.DataFrame()

    def fit_raw_data(self, source):
        self.re.read_csv(source)
        self.keywords_name = self.re.get_name_list()
        self.keywords_tag = self.re.get_tag_list()
        self.keywords_ingre = self.re.get_ingre_list()
        self.keywords_df = self.re.get_keyword_df()

    def fit_scrubed_local(self, csv):
        self.keywords_df = pd.read_csv(csv)
        self.keywords_df.set_index("recipe_id", inplace=True)
        name_keywords_distinct = self.keywords_df["recipe_name"].to_list()[0].copy()
        for i in self.keywords_df["recipe_name"].to_list()[1:]:
            name_keywords_distinct.extend(i)
        self.keywords_name = list(set(name_keywords_distinct))
        tag_keywords_distinct = self.keywords_df["recipe_tag"].to_list()[0].copy()
        for i in self.keywords_df["recipe_tag"].to_list()[1:]:
            tag_keywords_distinct.extend(i)
        self.keywords_tag = list(set(tag_keywords_distinct))
        ingre_keywords_distinct = self.keywords_df["recipe_ingre"].to_list()[0].copy()
        for i in self.keywords_df["recipe_ingre"].to_list()[1:]:
            ingre_keywords_distinct.extend(i)
        self.keywords_ingre = list(set(ingre_keywords_distinct))

    def keywords_vec_generator(self, all_keywords):
        keywords_array = np.array(all_keywords)
        label_encoder, onehot_encoder = LabelEncoder(), OneHotEncoder(sparse=False)
        label_encoded = label_encoder.fit_transform(keywords_array)
        onehot_encoder.fit(label_encoded.reshape(len(label_encoded), 1))
        return label_encoder, onehot_encoder

    def keywords2vec(self, keywords, label_encoder, onehot_encoder):
        label_encoded = label_encoder.transform(keywords)
        label_encoded = label_encoded.reshape(len(label_encoded), 1)
        if label_encoded.shape == (0, 1):
            return np.zeros(len(onehot_encoder.get_feature_names()))
        onehot_encoded = onehot_encoder.transform(label_encoded)
        vector = np.sum(onehot_encoded, axis=0)
        return vector

    def transform_name(self):
        encoders = dict(zip(("label_encoder", "onehot_encoder"), self.keywords_vec_generator(self.keywords_name)))
        self.keywords_df["recipe_name"] = self.keywords_df["recipe_name"].map(lambda x: self.keywords2vec(x, **encoders))
        return self.keywords_df

    def transform_tag(self):
        encoders = dict(zip(("label_encoder", "onehot_encoder"), self.keywords_vec_generator(self.keywords_tag)))
        self.keywords_df["recipe_tag"] = self.keywords_df["recipe_tag"].map(lambda x: self.keywords2vec(x, **encoders))
        return self.keywords_df

    def transform_ingre(self):
        encoders = dict(zip(("label_encoder", "onehot_encoder"), self.keywords_vec_generator(self.keywords_ingre)))
        self.keywords_df["recipe_ingre"] = self.keywords_df["recipe_ingre"].map(lambda x: self.keywords2vec(x, **encoders))
        return self.keywords_df

    def calculate_name_distance(self, recipe_id):
        user_select_vec = self.keywords_df["recipe_name"][recipe_id]
        user_select_vec_dist = self.keywords_df['recipe_name'].apply(
            lambda y: np.sqrt(np.square(user_select_vec - y).sum()))
        return user_select_vec_dist[user_select_vec_dist.index != recipe_id].sort_values()

    def calculate_tag_distance(self, recipe_id):
        user_select_vec = self.keywords_df["recipe_tag"][recipe_id]
        user_select_vec_dist = self.keywords_df['recipe_tag'].apply(
            lambda y: np.sqrt(np.square(user_select_vec - y).sum()))
        return user_select_vec_dist[user_select_vec_dist.index != recipe_id].sort_values()

    def calculate_ingre_distance(self, recipe_id):
        user_select_vec = self.keywords_df["recipe_ingre"][recipe_id]
        user_select_vec_dist = self.keywords_df['recipe_ingre'].apply(
            lambda y: np.sqrt(np.square(user_select_vec - y).sum()))
        return user_select_vec_dist[user_select_vec_dist.index != recipe_id].sort_values()