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
Yas= veriler.iloc[:,1:4].values
print(Yas)
#imputer ile sayısal kolonlar üzerinde fit fonk. kullanarak
#eğitme yapılır. Yani 1ve 4 e kadar olan kolonlarda öğrenme yaıyorum
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
#fit ile öğrendiklerimizi transform ile değiştiriyoruz (öğrendiğini uygulasın)
#yani öğrendiği ortalama değerleri yerine yazdırıyoruz
print(Yas)

#amacımız ülke ismi gibi sayısal anlamı olmayan verilere sayısal bir anlam yüklemek
ulke = veriler.iloc[:,0:1].values
#tüm satırlar alınacağı için :, 0 ve 1. kolonlarda çalışacağız
print(ulke)
#sadece ülkeler yazdırıldı

#kategorik kolonun dönüşmesi için encoderlar çağrılacak
from sklearn import preprocessing
le= preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)
#0.kolondaki değerlere label encoding uygulamasını istedik
#her ülke için sayısal değer verdi (1,2,0 vb)

ohe = preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

#ülke kolonundan (az önce sayıya çevirdiğimiz) öğrenip daha sonra bunu transform edecek
#böylece hangi ülkedeyse  ona göre bir numaralandırma yapcak
#bulunduğu ülke 1 diğerleri 0