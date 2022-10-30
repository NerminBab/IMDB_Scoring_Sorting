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


# IMDB Weigthed Rating Calculating

# Weighted_rating = (v / (v+M) * r) + (M / (v+M) * C

# Weighted_rating = (vote count / (vote count + min votes required to be listed in the top 250) * vote average + (min votes required to be listed in the top 250 / (vote count + min votes required to be listed in the top 250) * the mean vote across the whole reporT

# Film 1:
# r = 8
# M = 500
# v = 1000

# birinci hesap: (1000 / (1000+500)) * 8 = 5.33
# ikinci hesap: 500 / (1000+500) * 7 = 2.33
# Toplam = 5.33 + 2.33 = 7.66

# Film 2:
# r = 8
# M = 500
# v = 3000

# birinci hesap: (3000 / (3000+500)) * 8 = 6.85
# ikinci hesap: 500 / (3000+500) * 7 = 1
# Toplam = 6.85 + 1 = 7.85

M = 2500
C = df["vote_average"].mean()

def weighted_rating(r, v, M, C):
    return (v/(v+M)*r) + (M/(v+M)*C)

# df.sort_values("average_count_score", ascending=False).head(20)
# Deadpool filmi için hesap yapalım (7.4/11444.0)
weighted_rating(7.40000, 11444.00, M, C)

# Inception filmi için hesap yapalım (8.1/14075.0)
weighted_rating(8.10, 14075.0, M, C)

# The Shawshank Redemption için hesap yapalım (8.5/8358.00)
weighted_rating(8.50, 8358.00, M, C)

df["weigthted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)

df.sort_values("weigthted_rating", ascending=False).head(10)


# BAYESIAN AVERAGE RATING SCORE İLE IMDB SIRALAMASI
# IMDB nin 2015 yılına kadar yaptığı sıralamaya göre ilk 5 filmi şöyle bulmuştuk:
# The Dark Knight - The Shawshank Redemption - Fight Club - Inception - Pulp Fiction
# Günümüzdeki listesine baktığımızda "Baba" filmlerini görüyotuz. IMDB sıralama yöntemini değiştirmiş. Bar skor yöntemini uygulayıp IMDB nin yeni listesine yaklaşıp yaklaşmadığımıza bakalım:
import math

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1-(1-confidence)/2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k+1)*(n[k]+1) / (N+K)
        second_part += (k+1)*(k+1)*(n[k]+1)/(N+K)
    score = first_part-z*math.sqrt((second_part-first_part*first_part)/(N+K+1))
    return score
# esaretin bedeli filmi için yıldız alma puanlarını yazarak bayes ortalama derecelendirme puanı hesaplarsak:
bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

df = pd.read_csv("imdb_ratings.csv") # yıldız puanlarına eriştik
df = df.iloc[0:, 1:]

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]]),axis=1)

df.sort_values("bar_score", ascending=False).head(20)

# IMDB nin günümüzdeki sıralaması ile neredeyse aynı.
