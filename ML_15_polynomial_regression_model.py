# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:41:46 2022

@author: bjk_a
"""

#5.3
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

X = veriler.iloc[:,1:2].values
Y= veriler.iloc[:,2:].values



#linear regression
#doğrusal model oluşturuyoruz
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


#polynomial regression
#doğrusal olmayan (nonlinear model) oluşturuyoruz

# 2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)

# 4. dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,Y)



#görselleştirmeler

#linear regression için
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X), color = 'blue')
plt.show()

#2.dereceden polinom için
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

#4.dereceden polinom için
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()


#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))



