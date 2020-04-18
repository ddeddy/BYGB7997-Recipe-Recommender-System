This is our group project for BYGB 7997 Text Analytics at Fordham University. 

Our Team contains BAI, Jiaqi; LIANG, Chenqi; OUYANG, Zhiyu; REN, Yifan; SHEN, Hengda

ReviewManager
-------
To fast extract keywords from reviews:
```python
  >>> from review_manager import ReviewManager
  >>> rm = ReviewManager()
  >>> keyword_list = rm.fast_keywords(source, source_type = "{}")  # source can be a csv file or a dataframe
  ```
To get a dataframe of keywords with the recipe id
```python
  >>> from review_manager import ReviewManager
  >>> rm = ReviewManager()
  >>> rm.read_csv(source)  # if source is dataframe, use self.read_dataframe
  >>> rm.tokenize_review()
  >>> df = rm.get_keywords_total(verbose = True)
  ```
RecipeManager
-------
To get name/tag/ingredients keyword list from according columns:
```python
  >>> from recipe_manager import RecipeManager
  >>> re = RecipeManager()
  >>> re.read_csv(".csv")
  >>> name_list = re.get_name_list()
  >>> tag_list = re.get_tag_list()
  >>> ingre_list = re.get_ingre_list()
  ```
To get a dataframe of keywords with the recipe id
```python
  >>> from recipe_manager import RecipeManager
  >>> re = RecipeManager()
  >>> re.read_csv(".csv")
  >>> df = re.get_keyword_df()
  ```
  
Review2Vec
-------
To generate a dataframe only containg vectorized keywords:
```python
  >>> from review2vec import Review2Vec
  >>> transformer = Review2Vec()
  >>> transformer.fit_raw_data(source, source_type = "{}")  # source can be a csv file or a dataframe
  >>> vectorized_df = transformer.transform()
  ```
Calculate the distance between selected recipe and the others:
```python
  # must after transform
  >>> distance_df = transformer.calculate_distance(<recipe_id>)
  ```
Recipe2vec
------
To generate a dataframe only containg vectorized keywords:
```python
  >>> from recipe2vec import Recipe2Vec
  >>> transformer = Recipe2Vec()
  >>> transformer.fit_raw_data(".csv")
  >>> transformer.transform_name()
  >>> transformer.transform_tag()
  >>> vectorized_df = transformer.transform_ingre()
  ```
Calculate the distance between selected recipe and the others:
```python
  # must after transform
  >>> distance_df = transformer.calculate_distance(<recipe_id>)
  ```
