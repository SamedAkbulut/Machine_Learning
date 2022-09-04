# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:06:12 2022

@author: bjk_a
"""
#5.2
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

#eğitim seviyesini x, maaşları y olarak veri setimizi ayırıyoruz
# [:,1:2] satırların hepsini al, sütunlardan 1. yi al
# [:,2:] satırların hepsini al, sütunlardan 2.yi al
X = veriler.iloc[:,1:2].values
Y = veriler.iloc[:,2:].values
#DF de hata verebileceği için numpy array gibi kaydettik, .values yazmasak error alıyoruz 


#verimiz lineer değil ama denemek için (öylesine) bir lineer model oluşturalım
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#oluşturulan modeli görselleştirelim
plt.scatter(X,Y, color='red')
#x in karşılığı olan değeri linreg den tahmin etsin
plt.plot(X,lin_reg.predict(X), color='blue')
plt.show()



#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
#üstteki, herhangi bir sayıyı polinomal olarak ifade etmeye yarar
#2.dereceden bir polinomal regresyon objesi oluşturalım (degree isteğe göre ayarlanabilir)
poly_reg = PolynomialFeatures(degree = 2)
#önce polinom olarak ifade edelim
x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)
#x poly de oluşturduğumuz olduğumuz değişkenleri kullanarak y değerini öğrenmesini istedik
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
#(poly_reg.fit_transform(X) --- lin reg için çizilecek her x değerini önce polinomal hale getir dedik
plt.show()


#tahminler

#önce normal lineer regresyon için
print(lin_reg.predict([[6.6]]))
print(lin_reg.predict([[11]]))


#sonra da polinomal regresyon için tahmin yaptırdık
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


