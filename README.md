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

<img src= "https://user-images.githubusercontent.com/66487971/89876494-68b6b280-dbc7-11ea-92be-e33efa73f6c0.png" width = 350>

```python

yelp.describe()

```

<img src= "https://user-images.githubusercontent.com/66487971/89876585-884ddb00-dbc7-11ea-8bb6-a8f9d2792789.png" width = 500>

**I create a new column called "text length" which is the number of words in the text column.**

```python

yelp['text length'] = yelp['text'].apply(len)

```

# EDA


## Imports

```python

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline

```

```python

g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')

```

<img src= "https://user-images.githubusercontent.com/66487971/89876894-fabebb00-dbc7-11ea-9e00-980ecf19f6cf.png" width = 1000>


```python

sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')

```

<img src= "https://user-images.githubusercontent.com/66487971/89877001-20e45b00-dbc8-11ea-9d72-8fff5749c725.png" width = 500>

```python

sns.countplot(x='stars',data=yelp,palette='rainbow')

```

<img src= "https://user-images.githubusercontent.com/66487971/89877075-3eb1c000-dbc8-11ea-816b-557ff417db42.png" width = 500>



```python
stars = yelp.groupby('stars').mean()
stars
```

<img src= "https://user-images.githubusercontent.com/66487971/89877201-6d2f9b00-dbc8-11ea-8c5f-0a1f5dbecf25.png" width = 350>

```python

stars.corr()

```

<img src= "https://user-images.githubusercontent.com/66487971/89877277-8a646980-dbc8-11ea-8b10-5089d8252db4.png" width = 350>

**Then I use seaborn to create a heatmap based off that .corr() dataframe:**

```python
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)

```
<img src= "https://user-images.githubusercontent.com/66487971/89877388-b122a000-dbc8-11ea-90ef-28170a130c7e.png" width = 350>

# NLP Classification 

** I create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.**

```python

yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]
```

```python

X = yelp_class['text']
y = yelp_class['stars']

```

**I import CountVectorizer and create a CountVectorizer object.**

```python 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```

```python 
X = cv.fit_transform(X)
```

# Train Test Split

```python


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

```

# Training a Model

I use multinomial Naive Bayes.

```python
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
```










































