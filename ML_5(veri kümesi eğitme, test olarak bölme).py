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


ulke = veriler.iloc[:,0:1].values
#tüm satırlar alınacağı için :, 0 ve 1. kolonlarda çalışacağız
print(ulke)

#kategorik kolonun dönüşmesi için encoderlar çağrılacak
from sklearn import preprocessing
le= preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)
#her ülke için sayısal değer verdi (1,2,0 vb)

ohe = preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

#ülke kolonundan (az önce sayıya çevirdiğimiz) öğrenip daha sonra bunu transform edecek
#böylece hangi ülkedeyse  ona göre bir numaralandırma yapcak
#bulunduğu ülke 1 diğerleri 0

sonuc= pd.DataFrame(data=ulke,index= range(22), columns=["fr","tr","us"]) 
#ilk 22 satırı al, sütun başlıklarını yaz
print(sonuc)

sonuc2=pd.DataFrame(data=Yas, index=range(22),columns=["boy","kilo","yas"])
print(sonuc2)
#bu verileri bir dataframe haline getirelim
cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3= pd.DataFrame(data=cinsiyet, index=range(22),columns=["cinsiyet"])
print(sonuc3)

#ilk başta normal sıralı halde olan cinsiyet, artık dataframe oldu

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
# axis =1 diyince ilk kolonu ilk df den alıp 2. df de ilk kolonla eşliyor

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#tüm hepsini tek bir df oluşturduk

#EĞİTİM VE TEST KÜMESİ OLARAK DF İ BÖLME
#ülke boy kilo ve yaştan cinseyeti tahmin etmeye çalışalım
#train ve test verinin satır bazlı bölümü (belli bir yere kadarı train belli bir yerden sonrası test)
# x ve y verinin bağımlı ve bağımsız değişken olarak bölünmesi
# x bizim bağımsız değişkeninimiz olan veriler
#bağımlı yani hedef olan değişken ise sonuç df sini yani bu örnekte cinsiyeti verir
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(s,sonuc3,test_size = 0.33,random_state=0)
#random state rastsal bir değere göre bölmeyi ifade ediyor
#%33 test, 67 si train edilecek
#veriyi dikey eksende bağımlı bağımsız olarak ayırıyoruz
#yatay eksende de train ve test olarak ayırıyoruz