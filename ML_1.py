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
veriler = pd.read_csv("veriler.csv")
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


