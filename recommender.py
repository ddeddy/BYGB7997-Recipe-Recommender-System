import pandas as pd
from review2vec import Review2Vec
from recipe2vec import Recipe2Vec


class Recommender:
    def __init__(self):
        self.transformer_review = Review2Vec()
        self.transformer_recipe = Recipe2Vec()

    def fit_transform(self, review_file, recipe_file):
        self.transformer_review.fit_raw_data(review_file, "csv")
        self.transformer_recipe.fit_raw_data(recipe_file)
        self.transformer_review.transform()
        self.transformer_recipe.transform_name()
        self.transformer_recipe.transform_tag()
        self.transformer_recipe.transform_ingre()

    def generator(self, recipe_id):
        recipe_df = self.transformer_review.calculate_distance(recipe_id).to_frame()
        name_df = self.transformer_recipe.calculate_name_distance(recipe_id).to_frame()
        tag_df = self.transformer_recipe.calculate_tag_distance(recipe_id).to_frame()
        ingre_df = self.transformer_recipe.calculate_ingre_distance(recipe_id).to_frame()

        dfs = [recipe_df, name_df, tag_df, ingre_df]
        ranking_df = pd.concat(dfs, join='outer', axis=1)

        ranking_df["score"] = ranking_df["keywords"] * ranking_df["recipe_name"] * ranking_df["recipe_tag"] * ranking_df["recipe_ingre"]
        ranking_df.sort_values("score", inplace=True)
        for id_ in ranking_df.index:
            yield id_
