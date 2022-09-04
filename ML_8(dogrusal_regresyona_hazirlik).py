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

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

#veriler standartklaştırıldı

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)








