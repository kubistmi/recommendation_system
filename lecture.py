import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.listdir('data')

# books
book = pd.read_csv('data/books.csv')
book.head()

# book tags
bk_tags = pd.read_csv('data/book_tags.csv')
bk_tags.head()

# ratings
rats = pd.read_csv('data/ratings.csv')
rats.head()

# tags
tags = pd.read_csv('data/tags.csv')
tags.head()

# to read
to_read = pd.read_csv('data/to_read.csv')
to_read.head()

# are goodreads_id and book_id the same?
sum(
    book.book_id.isin(
        bk_tags.goodreads_book_id.drop_duplicates()
        )
    )

# find frequent tags
tg_freq = (
    bk_tags
    .groupby('tag_id')
    .aggregate(
        {'count' : 'sum',
        'goodreads_book_id': 'size'
        }
    )
)

most_freq = (
    tg_freq
    .sort_values('count', ascending = False)
    .iloc[:100,:]
)

tags100 = (
    tags    
    .merge(most_freq, left_index= True, right_index = True)
    .sort_values('count', ascending = False)
)

del(tg_freq, most_freq)

# remove nonsense
tags100.query('goodreads_book_id < 1000')
tags100 = tags100[~tags100.tag_name.str.contains('read|own|buy|default|ya')]
tags100.loc[:,['count', 'goodreads_book_id']].describe()

# fancy plots
_ = plt.hist(tags100.goodreads_book_id, bins = 20)
_ = plt.xlabel('# books with given tag')
_ = plt.ylabel('Frequency')
plt.show()

_ = plt.scatter(tags100['count'], tags100.goodreads_book_id)
plt.show()

# define table of tag dummies
bk_tags100 = (
    bk_tags
    .merge(tags100, on = 'tag_id')
    .drop({'count_x', 'count_y', 'goodreads_book_id_y'}, axis = 1)
    .rename({'goodreads_book_id_x': 'goodreads_book_id'}, axis = 1)
)

bk_tag_mat = (
    bk_tags100
    .drop('tag_name', axis = 1)
    .assign(help = 1)
    .pivot_table(
        values = 'help',
        index = 'goodreads_book_id',
        columns = 'tag_id',
        fill_value = 0)
)
bk_tag_mat.iloc[:10, :10]
bk_tag_mat.shape

del(tags100)

# check the tags per book
tg_per_book = bk_tag_mat.apply(sum, axis = 1)

tg_per_book.describe()

idx = tg_per_book[tg_per_book == 47].index.values
book[book.book_id.isin(idx)].iloc[:,:10]
bk_tags100.query('goodreads_book_id == @idx[0]')

del(idx, tg_per_book)

# ratings frequency
usr_rat = (
    rats
    .groupby('user_id')
    .size()
    .sort_values()
    .reset_index()
    .rename({0 : 'ratings'}, axis = 1)
)

usr_rat.ratings.describe()

_ = plt.scatter(x = usr_rat.index, y = usr_rat.ratings)
_ = plt.xlabel('Users')
_ = plt.ylabel('# ratings per user')
_ = plt.hlines(22, 0, 55000)
plt.show()

del(usr_rat)

# one user one book, more ratings?
rat_dup = (
        rats
        .groupby(['user_id', 'book_id'])
        .size()
)

rat_dup = (
    rat_dup
    [rat_dup>1]
    .reset_index()
    .merge(rats, on = ['user_id', 'book_id'])
    .rename({0:'dups'}, axis = 1)
)

max_rat = rats.groupby(['book_id', 'user_id']).rating.transform(max)
rats = rats.loc[rats.rating == max_rat].drop_duplicates()

sum(
    rats
    .groupby(['user_id', 'book_id'])
    .size() 
    > 1
)