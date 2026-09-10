# -*- coding: utf-8 -*-
"""
Licensed @C.TR
"""

import numpy
    
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
            return 0

class model():
    def __init__(self,N,M,D):
        self.entries=N
        self.numberfeatures1=M
        self.numberfeatures2=D
        self.T1=numpy.random.uniform(-param.h,param.h,[self.numberfeatures1,self.numberfeatures2])
        self.T2=numpy.random.uniform(-param.h,param.h,[self.numberfeatures2,self.numberfeatures1])
        self.activate=numpy.vectorize(fun.f)
        self.dactivate=numpy.vectorize(fun.g)
    def update(self,X,S,alpha):
        dE1,dE2=model.dE(self,X,S)
        T1temp,T2temp=self.T1-alpha*dE1,self.T2-alpha*dE2
        self.T1,self.T2=T1temp,T2temp      
    def encode(MODEL,X):
        X1=numpy.matmul(X,MODEL.T1)
        Y1=MODEL.activate(X1)
        return Y1         
    def encode_decode(MODEL,X):
        X1=numpy.matmul(X,MODEL.T1)
        Y1=MODEL.activate(X1)
        X2=numpy.matmul(Y1,MODEL.T2)
        Y2=MODEL.activate(X2)
        return Y2
    def forward_compute(MODEL,X):
        X1=numpy.matmul(X,MODEL.T1)
        Y1=MODEL.activate(X1)
        H1=MODEL.dactivate(X1)
        X2=numpy.matmul(Y1,MODEL.T2)
        Y2=MODEL.activate(X2)
        H2=MODEL.dactivate(X2)
        return Y1,Y2,H1,H2
    def dE(MODEL,X,S):
        Y1,Y2,H1,H2=model.forward_compute(MODEL,X)
        E2=numpy.multiply(Y2-S,H2)
        dE2=numpy.matmul(numpy.transpose(Y1),E2)
        E1=numpy.multiply(numpy.transpose(numpy.matmul(MODEL.T2,numpy.transpose(E2))),H1)
        dE1=numpy.matmul(numpy.transpose(X),E1)
        return dE1,dE2
    
    def error(MODEL,X,S):
        Y=model.encode_decode(MODEL,X)
        return float(sum(sum(pow(Y-S,2))))/len(S)
    
class param:
    N=int(20)
    D=int(50)
    alpha=float(0.01)
    h=float(0.1)
    
class mparam:
    epsilon=float(0.2)
    A=float(10)
    