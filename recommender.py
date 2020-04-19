#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from review2vec import Review2Vec
from recipe2vec import Recipe2Vec

class Recommender:
    def __init__(self):
        self.transformer_review = Review2Vec()
        self.transformer_recipe = Recipe2Vec()
    
    def fit_transform(self, file):
        self.transformer_review.fit_raw_data()
        self.transformer_recipe.fit_raw_data()
        self.transformer_review.transform()
        self.transformer_recipe.transform_name()
        self.transformer_recipe.transform_tag()
        self.transformer_recipe.transform_ingre()
    
    def generator(self, recipe_id):
        recipe_df = transformer_review.calculate_distance(recipe_id)
        name_df = transformer_recipe.calculate_name_distance(recipe_id)
        tag_df = transformer_recipe.calculate_tag_distance(recipe_id)
        ingre_df = transformer_recipe.calculate_ingre_distance(recipe_id)
        for df in [recipe_df, name_df, tag_df, ingre_df]:
            try:
                ranking_df = recipe_df.merge(df, left_index=True, right_index=True)
            except:
                ranking_df = df
        ranking_df["score"] = ranking_df["keywords"]*ranking_df["recipe_name"]*ranking_df["recipe_tag"]*ranking_df["recipe_ingre"]
        ranking_df.sort_values("score",inplace = True)
        for id_ in ranking_df.index:
            yield id_
