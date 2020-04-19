#!/usr/bin/env python
# coding: utf-8

from review_manager import ReviewManager

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class Review2Vec:
    def __init__(self):
        self.rm = ReviewManager()
        self.keywords = list()
        self.keywords_df = pd.DataFrame()
    
    def fit_raw_data(self, source, source_type):
        if source_type == "csv":
            self.rm.read_csv(source)
        if source_type == "dataframe":
            self.rm.read_dataframe(source)
        self.rm.tokenize_review()
        self.keywords = self.rm.get_keywords_total()
        self.keywords_df = self.rm.get_keywords_total(verbose = True)
        
    def fit_scrubed_local(self, csv):
        self.keywords_df = pd.read_csv(csv)
        keywords_distinct = self.keywords_df["keywords"].to_list()[0].copy()
        for i in self.keywords_df["keywords"].to_list()[1:]:
            keywords_distinct.extend(i)
        self.keywords = list(set(keywords_distinct))
        
    def keywords_vec_generator(self, all_keywords):
        keywords_array = np.array(all_keywords)
        label_encoder, onehot_encoder = LabelEncoder(), OneHotEncoder(sparse=False)
        label_encoded = label_encoder.fit_transform(np.array(all_keywords))
        onehot_encoder.fit(label_encoded.reshape(len(label_encoded), 1))
        return label_encoder, onehot_encoder
    
    def keywords2vec(self, keywords, label_encoder, onehot_encoder, **kwargs):
        label_encoded = label_encoder.transform(keywords)
        label_encoded = label_encoded.reshape(len(label_encoded), 1)
        if label_encoded.shape == (0, 1): 
            return np.zeros(len(onehot_encoder.get_feature_names()))
        onehot_encoded = onehot_encoder.transform(label_encoded)
        vector = np.sum(onehot_encoded, axis=0)
        return vector

    def transform(self):
        encoders = dict(zip(("label_encoder", "onehot_encoder"), self.keywords_vec_generator(self.keywords)))
        self.keywords_df["keywords"] = self.keywords_df["keywords"].map(lambda x:self.keywords2vec(x, **encoders))
        return self.keywords_df
    
    def calculate_distance(self, recipe_id):
        user_select_vec = self.keywords_df["keywords"][recipe_id]
        user_select_vec_dist = self.keywords_df['keywords'].apply(lambda y:np.sqrt(np.square(user_select_vec - y).sum()))
        return user_select_vec_dist[user_select_vec_dist.index != recipe_id].sort_values()
        
