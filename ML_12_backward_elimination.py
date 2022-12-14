# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 18:51:01 2022

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


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
print(boy)
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)


r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)

#modelin ve modelin ba??ar??s??n?? hesaplamak i??in statsmodel ??a??r??ld??

import statsmodels.api as sm

#hangi de??i??kenin sistemi ne kadar etkiled??ini anlamak i??in bunlar?? i??eren bir dizi ??zerinden ??al????aca????z
#bir dizi olu??turup t??m de??i??kenleri i??ine ekliyoruz
#sonras??nda s??ras??yla de??i??kenleri eleyerek (p value y??ksek olanlar??) devam edece??iz 

#bir sabitt de??i??ken olmas?? i??in bir dzi ekleyece??iz (Beta 0)
#np.ones diyerek birlerden olu??an bir dizi olu??turuyoruz ekliyoruz (22 tane 1 ekledik)
#astype ile tipini belirledik
#valuelar?? veriden al dedik ve veri dizisine ekledik
#axis 1 dedik ve kolon olarak ekledik -1 desek sat??r eklerdi


X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1)

#t??m sat??rlar?? ve t??m s??tunlar?? i??eren bir liste olu??turuyoruz

X_l = veri.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)

#istatistiksel de??erleri ????karacak olan kodumuz
#boy kolonunu tahmin ettirmek istedi??imiz i??in onu yazd??k
#x l i??inde yer alan her kolonun boy kolonu ??zerine olan etkisini bulma gayesindeyiz


model = sm.OLS(boy,X_l).fit()
print(model.summary())

#ols result a g??re p de??eri ne kadar d??????k olursa o kadar iyidir
#x5 fazla geldi, 4. eleman?? eleyece??iz
#x5 yani 4. eleman?? (5.kolon) almad??k

#i??lemi tekrar ediyoruz

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

#eleye eleye bize g??re  kabul edilebilir olana kadar devam ediyoru<

X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()
print(model.summary())

#elde edilen verinin daha g??venilir oldu??u d??????n??lmekte oldu??u i??in 
#kalan kolonlar?? regresyona sokup sonu?? elde dilebilir


