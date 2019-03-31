import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# API
import requests
# SQL
from sqlalchemy import create_engine, MetaData, Table, select
# SCIPY
import scipy.spatial.distance as dst
from scipy import sparse
#SKLEARN
from sklearn.model_selection import train_test_split as tts, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as knn_cls
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression as ols, Lasso
# STATMODELS
import statsmodels.api as sm


os.listdir('data')

# books
"""a simple csv approach - depreciated in favor of API
book = pd.read_csv('data/books.csv')
book.head()
"""

engine = create_engine(
    'postgresql://{user}:{password}@{host}:{port}/{database}'.format(
        host= '',
        port= '5432',
        database= '',
        user= '',
        password= '' 
        )
    )

# PYTHON OOP
conn = engine.connect()
metadata = MetaData()

books = Table('books', metadata, autoload_with=engine)

query = select([books]).where(books.columns.id == 1)
print(query)

sql_res = conn.execute(query).fetchmany(5)
pd.DataFrame(sql_res[:15], columns= sql_res[0].keys())
conn.close()

# DIRECTLY
sql_res = engine.execute("SELECT * FROM books").fetchall()
book = pd.DataFrame(sql_res, columns= sql_res[0].keys())

# book tags
"""a simple csv approach - depreciated in favor of API
bk_tags = pd.read_csv('data/book_tags.csv')
bk_tags.head()
"""

req = requests.get('http://{host}:8000/tags-all'.format(host = 'localhost'))

req
req.headers
req.encoding
req.json()[:10]
bk_tags = pd.DataFrame(req.json())

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

tags_pat = 'read|own|buy|default|favou?rit|book|library|wish'
tags100 = tags100[~tags100.tag_name.str.contains(tags_pat)]
tags100.loc[:,['count', 'goodreads_book_id']].describe()

# fancy plots
_ = plt.hist(tags100.goodreads_book_id, bins = 20)
_ = plt.xlabel('# books with given tag')
_ = plt.ylabel('Frequency')
#plt.show()

_ = plt.scatter(tags100['count'], tags100.goodreads_book_id)
#plt.show()

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

idx = tg_per_book[tg_per_book == max(tg_per_book)].index.values
book[book.book_id.isin(idx)].iloc[:,:10]
bk_tags100.query('goodreads_book_id == @idx[0]')

del(idx, tg_per_book)

# distance test
a = bk_tag_mat.iloc[:1000,]
b = a.dot(a.T)
(b.apply(np.mean)).describe()

one = np.zeros(69)
one[:17] = 1

two = np.zeros(69)
two[9:26] = 1

np.linalg.norm(one - two)
np.sqrt(np.linalg.norm(one)**2 + np.linalg.norm(two)**2)

dst.cosine(one, two)

del(a, b, one, two)

# ratings frequency
rats.user_id.drop_duplicates().count()

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
_ = plt.hlines(usr_rat.ratings.mean(), 0, 55000)
#plt.show()

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
rat_dup.head()

max_rat = rats.groupby(['book_id', 'user_id']).rating.transform(max)
rats = rats.loc[rats.rating == max_rat].drop_duplicates()

sum(
    rats
    .groupby(['user_id', 'book_id'])
    .size() 
    > 1
)

del(max_rat, rat_dup)

# rating distribution
rats.rating.describe()

_ = plt.hist(rats.rating)
#plt.show()

rats.groupby('rating').size()

# ratings regression
rats_reg = rats.loc[:,['book_id', 'rating']].merge(book.iloc[:,:2], left_on = 'book_id', right_on = 'id', suffixes =('_x', '')).drop({'book_id_x', 'id'}, axis = 1).merge(bk_tag_mat, left_on = 'book_id', right_index = True)
rats_reg.set_index('book_id', inplace = True)

reg_x = rats_reg.drop('rating', axis = 1)
reg_y = rats_reg.rating
del(rats_reg)

# ML approach
sklr_estimate = ols().fit(reg_x, reg_y)
resid = sklr_estimate.predict(reg_x) - reg_y
sklr_estimate.coef_

_ = plt.scatter(x = range(reg_x.shape[0]), y = resid, marker = 'o', s = 0.002)
_ = plt.hlines(y = 0, xmin = 0, xmax = 1000000)
#plt.show()

# STATS approach
lm_estimate = sm.OLS(reg_y, reg_x).fit()
lm_estimate.summary()

# LASSO approach
""" RUN AT OWN RISK!
lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)
n_folds = 5

tuned_parameters = [{'alpha': alphas}]

cvl = GridSearchCV(lasso, tuned_parameters, cv = n_folds, n_jobs = -1)
cvl.fit(reg_x, reg_y)
scores = cvl.cv_results_['mean_test_score']
scores_std = cvl.cv_results_['std_test_score']
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores, color = 'red')
plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

plt.show()

best_lasso = cvl.best_estimator_.fit(reg_x, reg_y).coef_
"""

best_lasso = Lasso(alpha = 0.00017433288221999874).fit(reg_x, reg_y).coef_
best_lasso
del(resid, reg_x, reg_y, best_lasso)

# good and bad ratings
rats = rats.assign(good = rats.rating == 5).astype(int)

# choose an user
np.random.seed(1234)
user = (
    usr_rat
    .query('ratings > 50')
    .sample()
    .reset_index()
)
user

user = rats[rats.user_id == user.user_id[0]]

sum(user.good)

del(usr_rat)

# prepare tags of books read by user
user_rat = (
    user[['book_id', 'good']]
    .merge(book[['id', 'book_id']], left_on = 'book_id', right_on = 'id')
    .drop('id', axis = 1)
    .rename({'book_id_x' : 'book_id', 'book_id_y':'goodreads_book_id'}, axis = 1)
    .merge(bk_tag_mat, how = 'left', left_on = 'goodreads_book_id', right_index = True)
    .drop('goodreads_book_id', axis = 1)
    .set_index('book_id')
)
user_rat.iloc[:10,:10]

# prepare tags of books not-read by user
bk_top = (
    rats
    [~rats.book_id.isin(user_rat.index.values)]
    .groupby('book_id')
    .good
    .mean()
    .sort_values(ascending = False)
    [:1000]
)
bk_top.head()

other_rat = (
    book[['id','book_id']]
    [book.id.isin(bk_top.index)]
    .merge(bk_tag_mat, how = 'left', left_on = 'book_id', right_index = True)
    .drop('book_id', axis = 1)
    .rename({'id':'book_id'}, axis = 1)
    .set_index('book_id')
)
other_rat.iloc[:10,:10]

# KNN recommendations

# validated k
tr_x, te_x, tr_y, te_y = tts(
    user_rat.drop('good', axis = 1),
    user_rat.good,
    test_size = 0.2,
    random_state = 12345
    )

def knn_train(k):
    global tr_x, tr_y, te_x, te_y
    pred = (
        knn_cls(n_neighbors = k, n_jobs= -1)
        .fit(tr_x, tr_y)
        .score(te_x, te_y) 
    )
    return(pred)

# show accuracy
k_arr = range(1,20)
acc = [knn_train(i) for i in k_arr]
acc

_ = plt.plot(k_arr, acc)
_ = plt.xticks(k_arr, k_arr)
#plt.show()

knn = knn_cls(n_neighbors = 3, weights= 'distance', n_jobs= -1)
knn_trained = knn.fit(user_rat.drop('good', axis = 1), user_rat.good)
knn_meta = knn_trained.kneighbors(other_rat)

other_rat['pred'] = knn_trained.predict(other_rat)
other_rat['dist'] = np.apply_along_axis(np.mean, 1, knn_meta[0])

del(knn, knn_trained, knn_meta)

knn_rec = (
    other_rat
    .query('pred == 1')
    .sort_values('dist')
    .iloc[:10, -2:]
)
knn_rec

# validate results - tags?
pred = (
    book
    [book.id.isin(knn_rec.index)]
    [['book_id']]
    .merge(bk_tags100, left_on = 'book_id', right_on = 'goodreads_book_id')
    .groupby('tag_name')
    [['tag_id']]
    .count()
    .sort_values('tag_id', ascending = False)
    [:10]
    /knn_rec.shape[0]
)

act = (
    book
    [book.id.isin(user_rat.index)]
    [['book_id']]
    .merge(bk_tags100, left_on = 'book_id', right_on = 'goodreads_book_id')
    .groupby('tag_name')
    [['tag_id']]
    .count()
    .sort_values('tag_id', ascending = False)
    .iloc[:10]
    /user_rat.shape[0]
)

act.merge(
    pred, how = 'outer',
    left_index = True, right_index = True,
    suffixes = ('_act', '_pred')
    )

del(act, pred)

# validate results - to read?
user_to_read = to_read[to_read.user_id == user.user_id.iloc[0]]
user_to_read.book_id.isin(knn_rec.index).values

# validate results - most frequent?
knn_rec.index.isin(bk_top.index[:20])
del(user_rat, other_rat)

# PCA
pca = PCA().fit(bk_tag_mat)

_ = plt.plot(pca.explained_variance_ratio_, marker = 'x')
#plt.show()

imp_comps = [i for i in pca.explained_variance_ratio_ if i > 0.015]
comps = pd.DataFrame(pca.components_).T

comps.set_index(bk_tag_mat.columns, inplace = True)
comps.iloc[:10, :10]

comps = comps.iloc[:, :len(imp_comps)]

comps = (
    tags
    .merge(comps, left_on = 'tag_id', right_index = True)
    .drop('tag_id', axis = 1)
    .set_index('tag_name')
)
 # PCA interpret
for i in range(len(imp_comps)):
    print('PCA' + str(i+1) + '  ' + 20*'#')
    x = comps[i]
    x = x.reindex(x.abs().sort_values(ascending = False).index)[:5]
    print(x)

# PCA transform
bk_pca = (
    PCA(n_components = len(imp_comps))
    .fit(bk_tag_mat)
    .transform(bk_tag_mat)
)

bk_pca = pd.DataFrame(bk_pca, index = bk_tag_mat.index)
del(comps, imp_comps, i)

# PCA user and other ratings
user_pca = (
    user[['book_id', 'good']]
    .merge(book[['id', 'book_id']], left_on = 'book_id', right_on = 'id')
    .drop('id', axis = 1)
    .rename({'book_id_x' : 'book_id', 'book_id_y':'goodreads_book_id'}, axis = 1)
    .merge(bk_pca, how = 'left', left_on = 'goodreads_book_id', right_index = True)
    .drop('goodreads_book_id', axis = 1)
    .set_index('book_id')
)
user_pca.head()

other_pca = (
    book[['id','book_id']]
    [book.id.isin(bk_top.index)]
    .merge(bk_pca, how = 'left', left_on = 'book_id', right_index = True)
    .drop('book_id', axis = 1)
    .rename({'id':'book_id'}, axis = 1)
    .set_index('book_id')
)
other_pca.head()

# PCA KNN
knn = knn_cls(weights = 'distance', n_jobs = -1)
knn_trained = knn.fit(user_pca.drop('good', axis = 1), user_pca.good)
knn_meta = knn_trained.kneighbors(other_pca)

other_pca['pred'] = knn_trained.predict(other_pca)
other_pca['dist'] = np.apply_along_axis(np.mean, 1, knn_meta[0])

del(knn, knn_trained, knn_meta)

knn_rec_pca = (
    other_pca
    .query('pred == 1')
    .sort_values('dist')
    .iloc[:10, -2:]
)
knn_rec_pca

# PCA validate results - tags?
pred = (
    book
    [book.id.isin(knn_rec_pca.index)]
    [['book_id']]
    .merge(bk_tags100, left_on = 'book_id', right_on = 'goodreads_book_id')
    .groupby('tag_name')
    [['tag_id']]
    .count()
    .sort_values('tag_id', ascending = False)
    [:10]
    /knn_rec_pca.shape[0]
)

act = (
    book
    [book.id.isin(user_pca.index)]
    [['book_id']]
    .merge(bk_tags100, left_on = 'book_id', right_on = 'goodreads_book_id')
    .groupby('tag_name')
    [['tag_id']]
    .count()
    .sort_values('tag_id', ascending = False)
    .iloc[:10]
    /user_pca.shape[0]
)

act.merge(
    pred, how = 'outer',
    left_index = True, right_index = True,
    suffixes = ('_act', '_pred')
    )

del(act, pred)

# PCA validate results - to read?
user_to_read = to_read[to_read.user_id == user.user_id.iloc[0]]
user_to_read.book_id.isin(knn_rec_pca.index).values

del(bk_pca, user_pca, other_pca, knn_rec_pca)

# Collaborative Filtering - item

# rat_mat = (
#     rats
#     [['book_id', 'user_id', 'good']]
#     .pivot_table(
#         values = 'good',
#         index = 'user_id',
#         columns = 'book_id',
#         fill_value = np.nan)
# )

# check IDs
rats.book_id.sort_values().drop_duplicates()
rats.user_id.sort_values().drop_duplicates()

rat_mat = sparse.csr_matrix((rats.good, (rats.user_id-1, rats.book_id-1)))

items = (rat_mat.transpose() @ rat_mat).toarray()

items = np.divide(items, np.diag(items).reshape((-1,1)))

np.fill_diagonal(items, 0)

ibcf = pd.DataFrame(items, index = book.id, columns = book.id)

ibcf.iloc[:10,:10]

ibcf = ibcf.assign(closest = ibcf.idxmax(axis = 1), sim = ibcf.max(axis = 1))

# size estimation - mixed data
def estimate_size(nrow, ncol, out = 'Gb'):
    out = out.lower()
    out_dict = {'tb':10**12, 'gb':10**9, 'mb':10**6}
    return(nrow * ncol * 11 / out_dict[out])

ibcf.__sizeof__() / 2**20 # megabytes


ibcf.iloc[ibcf.index.isin(user.book_id), -2:].sort_values('sim')

# Collaborative Filtering - user
g = MiniBatchKMeans(n_clusters=20).fit(rat_mat)
_, counts = np.unique(g.labels_, return_counts  = True)
counts
# ! nah - not really

# SVD decomposition

rat_mat = sparse.csr_matrix((rats.rating, (rats.user_id-1, rats.book_id-1)))

u, s, v = svds(rat_mat.asfptype(), 10)

ubcf = pd.DataFrame(u, index = range(1, max(rats.user_id)+1))

svd_knn = knn_cls(n_neighbors=5).fit(ubcf[ubcf.index != user.user_id.iloc[0]], ubcf[ubcf.index != user.user_id.iloc[0]].index)
svd_knn.predict(ubcf[ubcf.index == user.user_id.iloc[0]])

ubcf.loc[[13794, 6634]]

rats6634 = rats[rats.user_id == 6634]
rats13794 = rats[rats.user_id == 13794]

rats6634 = rats6634[~rats6634.book_id.isin(rats13794.book_id)]

rats6634[rats6634.good == 1].merge(book.loc[:,['id', 'title']], left_on = 'book_id', right_on = 'id')
rats13794[rats13794.good == 1].merge(book.loc[:,['id', 'title']], left_on = 'book_id', right_on = 'id')
