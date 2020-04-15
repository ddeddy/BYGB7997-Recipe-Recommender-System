This is our group project for BYGB 7997 Text Analytics at Fordham University. 

Our Team contains BAI, Jiaqi; LIANG, Chenqi; OUYANG, Zhiyu; REN, Yifan; SHEN, Hengda

ReviewManager
-------
To fast extract keywords from reviews:
```python
  >>> from ReviewManager import ReviewManager
  >>> rm = ReviewManager()
  >>> keyword_list = rm.fast_keywords(source)  # source can be a csv file or a dataframe
  ```
To get a dataframe of keywords with the recipe id
```python
  >>> from ReviewManager import ReviewManager
  >>> rm = ReviewManager()
  >>> rm.read_csv(source)  # if source is dataframe, use self.read_dataframe
  >>> rm.tokenize_review(source)
  >>> df = rm.get_keywords_total(verbose = True)
  ```
