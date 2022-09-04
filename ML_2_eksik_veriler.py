# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:49:08 2021

@author: bjk_a
"""
#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yukleme
veriler = pd.read_csv("eksikveriler.csv")
print("veriler")
#veri on isleme
boykilo=veriler[["boy","kilo"]]
print(boykilo)

class insan:
    boy=180
    def kosmak(self,b):
        return b+10

ali= insan() 
#alinin insan sınıfında olduğunu gösterdim
print(ali.boy)
print(ali.kosmak(90))

l=[1,3,4] #liste

#eksik veriler

#sci-kit learn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
#strategy, ne ile impute edeceğimizdir (ne ile yerine değiştireceği)
#yukarıda amacımız boş verileri doldurmak
Yas= veriler.iloc[:,1:4].values
#tüm satırlar geldi, 1-2-3. sütunlar geldi
print(Yas)
#imputer ile sayısal kolonlar üzerinde fit fonk. kullanarak
#eğitme yapılır. Yani 1ve 4 e kadar olan kolonlarda öğrenme yaıyorum
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
#fit ile öğrendiklerimizi transform ile değiştiriyoruz (öğrendiğini uygulasın)
#yani öğrendiği ortalama değerleri yerine yazdırıyoruz
print(Yas)

