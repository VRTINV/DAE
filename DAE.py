# -*- coding: utf-8 -*-
"""
Licensed @C.TR
"""

import random
import pandas
    
class fun:
    def f(x):
        if x <= -mparam.A/2:
            return float(0)
        if x > -mparam.A/2 and x < mparam.A/2:
            return float(x/mparam.A)
        if x >= mparam.A/2:
            return float(1)
    def g(x):
        if x <= -mparam.A/2:
            return float(0)
        if x > -mparam.A/2 and x < -mparam.A/2 + mparam.epsilon:
            return float(1/(mparam.epsilon*mparam.A)*x+1/(2*mparam.epsilon))
        if x >= -mparam.A/2 + mparam.epsilon and x <= mparam.A/2 - mparam.epsilon:
            return float(1/mparam.A)
        if x > mparam.A/2 - mparam.epsilon and x < mparam.A/2:
            return float(-1/(mparam.epsilon*mparam.A)*x-1/(2*mparam.epsilon))
        if x >= mparam.A/2:
            return float(0)

class model():
    def __init__(self,N,M,D):
        self.entries=N
        self.numberfeatures1=M
        self.numberfeatures2=D
        self.T1=pandas.DataFrame([[random.uniform(-param.h,param.h) for i in range(self.numberfeatures2)] for j in range(self.numberfeatures1)])
        self.T2=pandas.DataFrame([[random.uniform(-param.h,param.h) for i in range(self.numberfeatures1)] for j in range(self.numberfeatures2)])
        self.activate=fun.f
        self.dactivate=fun.g
    def update(self,X,S,alpha):
        dE1,dE2=model.dE(self,X,S)
        T1temp,T2temp=self.T1-alpha*dE1,self.T2-alpha*dE2
        self.T1,self.T2=T1temp,T2temp      
    def encode(MODEL,X):
        X1=X.dot(MODEL.T1)
        Y1=X1.map(MODEL.activate)
        return Y1         
    def encode_decode(MODEL,X):
        X1=X.dot(MODEL.T1)
        Y1=X1.map(MODEL.activate)
        X2=Y1.dot(MODEL.T2)
        Y2=X2.map(MODEL.activate)
        return Y2
    def forward_compute(MODEL,X):
        X1=X.dot(MODEL.T1)
        Y1=X1.map(MODEL.activate)
        H1=X1.map(MODEL.dactivate)
        X2=Y1.dot(MODEL.T2)
        Y2=X2.map(MODEL.activate)
        H2=X2.map(MODEL.dactivate)
        return Y1,Y2,H1,H2
    def dE(MODEL,X,S):
        Y1,Y2,H1,H2=model.forward_compute(MODEL,X)
        E2=Y2-S
        dE2=pandas.DataFrame.transpose(Y1).dot(E2)
        E1=pandas.DataFrame.transpose(MODEL.T2.dot(pandas.DataFrame.transpose(E2))).mul(H1)
        dE1=pandas.DataFrame.transpose(X).dot(E1)
        return dE1,dE2
    
    def error(MODEL,X,S):
        Y=model.encode_decode(MODEL,X)
        return pow(Y-S,2).sum().sum()
    
class param:
    N=int(20)
    D=int(50)
    alpha=float(0.01)
    h=float(0.1)
    
class mparam:
    epsilon=float(0.2)
    A=float(10)
    
