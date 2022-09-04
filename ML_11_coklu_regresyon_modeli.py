# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 18:56:20 2022

@author: bjk_a
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
Yas = veriler.iloc[:,1:4].values
print(Yas)


#ülkevecinsiyet için sayısal değerler oluşturuyoruz
#encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#encoder: Kategorik -> Numeric
c = veriler.iloc[:,-1:].values
print(c)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

print(c)


#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

#dummy variabledan kurtulmak adına tek satır olarak cinsiyetler gelecek (0dan -1e seçildi)
cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)


#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

#simple linear regressiondan farklı birden çok inputumuz olması
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#x traine bak ve y traini öğren (aralarında model kur)
regressor.fit(x_train,y_train)

#x test e göre predict yapmasını istedik
y_pred = regressor.predict(x_test)


#bu sefer de boyu tahmin ettirmeye çalışalım
#bu yüzden boy kolonunun sağı ve solunu 2 ayrı df yapalım
#kendisini ise ayrı bir df yapalım
boy = s2.iloc[:,3:4].values
print(boy)
#bütün satırlar, 3. kolona kadar al
sol = s2.iloc[:,:3]
#bütün satırlar, 4.. kolondan sonra
sag = s2.iloc[:,4:]

#sol ve sağda yer alan verileri birleştirelim
veri = pd.concat([sol,sag],axis=1)

#verimizi train ve testlere böldük
x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

#artık fit edip tahmin ettirebiliriz
r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)





