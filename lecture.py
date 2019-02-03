import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.listdir('data')

# books
bk = pd.read_csv('data/books.csv')
bk.iloc[:,16:21].head()

# book tags
bk_tags = pd.read_csv('data/book_tags.csv')
bk_tags.head()

# ratings
rats = pd.read_csv('data/ratings.csv')
rats.head()

# tags
tags = pd.read_csv('data/tags.csv')
tags.head()

tg_freq = bk_tags.groupby('tag_id').aggregate({'count' : 'sum'})
tg_freq = tg_freq.sort_values('count', ascending = False).iloc[:5,:].index.values

tags.query('tag_id in @tg_freq')

# to read
to_read = pd.read_csv('data/to_read.csv')
to_read.head()
