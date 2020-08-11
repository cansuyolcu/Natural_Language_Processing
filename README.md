# Natural_Language_Processing
In this NLP project I will be attempting to classify Yelp Reviews into 1 star or 5 star categories based off the text content in the reviews.


I will be using [Yelp Review Data Set from Kaggle](https://www.kaggle.com/c/yelp-recsys-2013).

Each observation in this dataset is a review of a particular business by a particular user.

The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.

The "cool" column is the number of "cool" votes this review received from other Yelp users.

All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.

The "useful" and "funny" columns are similar to the "cool" column.

## Imports

```python

import numpy as np
import pandas as pd

```

## The Data

```python

yelp = pd.read_csv('yelp.csv')
yelp.head()

```

<img src= "https://user-images.githubusercontent.com/66487971/89876402-47ee5d00-dbc7-11ea-8f3b-4869021c1a41.png" width = 1000>

```python

yelp.info()

```

<img src= "https://user-images.githubusercontent.com/66487971/89876494-68b6b280-dbc7-11ea-92be-e33efa73f6c0.png" width = 500>

```python

yelp.describe()

```

<img src= "https://user-images.githubusercontent.com/66487971/89876585-884ddb00-dbc7-11ea-8bb6-a8f9d2792789.png" width = 500>

**I create a new column called "text length" which is the number of words in the text column.**

```python

yelp['text length'] = yelp['text'].apply(len)

```

## EDA















