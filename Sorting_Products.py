import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%s' % x)

df = pd.read_csv("movies_metadata.csv", low_memory=False)

df = df[["title", "vote_average", "vote_count"]]
df.head()
df.shape

# VOTE AVERAGE A GÖRE SIRALAMA:

df.sort_values("vote_average", ascending=False).head(20)

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head()

from sklearn.preprocessing import MinMaxScaler

df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)).fit(df[["vote_count"]]).transform(df[["vote_count"]])

# vote_average * vote_count:

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score", ascending=False).head(20)


# IMDB Weigthed Rating

# Weighted_rating = (vote count / (vote count + min votes required to be listed in the top 250) * vote average + (min votes required to be listed in the top 250 / (vote count + min votes required to be listed in the top 250) * the mean vote across the whole report(currently 7.0)

# Weighted_rating = (v / (v+M) * r) + (M / (v+M) * C

# Film 1:
# r = 8
# M = 500
# v = 1000
# (1000 / (1000 + 500)) * 8 = 5.33





# Film 2:
# r = 8
# M = 500
# v = 3000

# birinci bölüm: (3000 / (3000+500)) * 8 = 6.85

# ikinci bölüm: 500 / (3000+500) * 7 = 1

# toplam = 7.85