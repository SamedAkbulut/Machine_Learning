# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:49:19 2022

@author: bjk_a
"""

#burada   bir hata aldım sebebini bulamadım
#eski kodlarda olduğu gibi ohe yapsam bence çözülür

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
#pd.read_csv("veriler.csv")


#burada yes no olan sütunları labelencoder ile 0-1 yapıp
#outlook u ise ohe (onehat encoder) yapıp 3 yeni sütunda gösterio sonra onları ayrı numpylar yapıp
#en sonda birleştirebilirdik
#onun yerine başka bir şey denedik

#veri on isleme

#encoder:  Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder
veriler2 = veriler.apply(LabelEncoder().fit_transform)
#bütün verilere label encoder yaptık
#numeric değerlerin encode edilmesini aslında istemiyoruz

c = veriler2.iloc[:,:1]
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)
#ilk sütun yani outlook'a onehat encoder uyguladık
#çünkü o veriler birbirinden bağımsız ve 3 sütun olması daha iyi olur 001 100 vb...

#hepsini tek bir df de birleştiriyoruz
#range 14, veri sayısı (Satır) overcast rainy ve sunny sütunlarının da başlıkla ohe ye uygun yazıldı
havadurumu = pd.DataFrame(data = c, index = range(14), columns=['overcast','rainy','sunny'])
#verilerde yer alan tempature ve hunidity numeric olduğu için direkt olarak alıp concat ile birleştirdik
#aynı df ye aldık
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
#label encoder yapılandan (veriler2) label encoder (yani 0-1) olan sütunları aldık ve aynı df ye attık
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)
#sonveriler = pd.concat([sonveriler,veriler2.iloc[:,-2:],sonveriler], axis = 1)
#bu şekilde  de ifade edilebilirdi, aynı şey

#Humidity yi tahmin ettirmeye çalışalım

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)
#başlangıçtan sona kadar olan kolonlar bağımsız değişken
#son kolon ise bağımlı değiken (humudity)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)

print(y_pred)
#buradan sonra backward elimination başladı



import statsmodels.formula.api as sm 
#14 (14 satır), tüm satırları al, son kolona kadar al (humudity almıyor)
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
#tüm kolonları al (x te humudity yok, yukarda zaten almadık)
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
#humidity bağımlı değişken (hedefimiz)
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())

#pi  değeri yüksek olan sütun ilk sütundu, onu verisetinden çıkarıyoruz
sonveriler = sonveriler.iloc[:,1:]

import statsmodels.formula.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
print(r.summary())


#attığım kolon (windy) yi x train ve x testten de çıkarıyoruz

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]


#sistemi yeni değerlerle eğitip yeniden tahmin yaptırıyoruz

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)


