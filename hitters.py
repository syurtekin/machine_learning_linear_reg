# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör
#
# putouts ile isabetli vuruş

from helpers.eda import *
from helpers.data_prep import *
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve

df = pd.read_csv("hitters.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df.shape
df.head()
check_df(df)

# aykırı değer
cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))

# aykırı değerleri baskıla

for col in num_cols:
    replace_with_thresholds(df, col)

# baskıladıktan sonra kontrol
for col in num_cols:
    print(col, check_outlier(df, col))

# eksik değer kontrolü
df.isnull().values.any()
df.dropna(inplace = True)
df.head()

df.isnull().values.any()

# feature engineering

df["NEW_HITS"] = df["Hits"] / df["CHits"]
df["NEW_RBI"] = df["RBI"] / df["CRBI"]
df["NEW_WALKS"] = df["Walks"] / df["CWalks"]
df["NEW_CAT_BAT"] = df["CAtBat"] / df["Years"]
df["NEW_CRUNS"] = df["CRuns"] / df["Years"]
df["NEW_CHITS"] = df["CHits"] / df["Years"]
df["NEW_CHMRUN"] = df["CHmRun"] / df["Years"]
df["NEW_CRBI"] = df["CRBI"] / df["Years"]
df["NEW_CWALKS"] = df["CWalks"] / df["Years"]

# kariyeri boyunca topa vurma sayısı ile isabetli vuruş arasındaki ilişki
df["NEW_SUCCESS"] = df["NEW_HITS"] * df["NEW_CAT_BAT"]

#OYUNCUNUN TAKIM ARKADAŞINLA YARDIMLAŞMASI VE İSABETLİ VURUŞ SAYISI
df["NEW_PUT_CHITS"] = df["PutOuts"] * df["CHits"]

# asist ve takım arkadaşı

df["NEW_ASIST_PUT"] = df["Assists"] / df["PutOuts"]
df.dropna(inplace = True)

# hits- error

df["NEW_RUN_ERR"] = df["Hits"] - df["Errors"]

check_df(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in df.columns:
    label_encoder(df, col)

df.head()
check_df(df)
# scale

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

# model

X = df.drop('Salary', axis=1)
y = df[["Salary"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

#ahmin Başarısını Değerlendirme

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
#0.25

# TRAIN RKARE
reg_model.score(X_train, y_train)
#0.811
# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#0.307
# Test RKARE
reg_model.score(X_test, y_test)
#0.69

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
# 0.30
