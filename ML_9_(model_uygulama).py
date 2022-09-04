# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:49:08 2021

@author: bjk_a
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme

#aylar ve satışları ayrı df yaptık
aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

#bu işlemi(kolonları bölme) iloc ile de yapabiliriz

satislar2 = veriler.iloc[:,:1].values
print(satislar2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

#satışlar bağlı değişken (aylara bağlı old. düşünüyoruz
#yani önce aylar sonra satışlar yazılır)

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)
'''
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

#veriler standartlaştırıldı

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''
#anlamlı tahminler gelsin diye yukarıyı comment line yaptık
#fakat ilerikiçalışmalarda standartlaştırma gerekebilir

#model insası(linear regression)
#model inşa etmek için sklearn den linear regression u import ettik
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#fit ile modeli inşa etmeye çalışıyor
lr.fit(x_train,y_train)

#neyden tahmin edileceğini predictin içine yazdık
#yani x teste göre ona karşılık gelecek y leri buluyor
tahmin = lr.predict(x_test)



