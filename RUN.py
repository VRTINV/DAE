# -*- coding: utf-8 -*-
"""
Licensed @C.TR
"""
import numpy

import norm
import DAE
import noise

class generate_data:
    def generate(N,M):
        return numpy.random.normal(0,1,[N,M])

N=25
M=100

i=10

RAW_DATA=generate_data.generate(N,M)

#RUN1
[X_current,[mu0,sigma0]]=norm.normalize.global_z(RAW_DATA)
[X_current,[minimum,maximum]]=norm.normalize.global_m(X_current)

#RUN2 
Xnoise_current=noise.noise.make_thresh(X_current)

#INIT_MODEL
MODEL=DAE.model(N,M,2*M)

#RUN_MODEL
j=0
while j<i:
    MODEL.update(Xnoise_current,X_current,DAE.param.alpha)
    j=j+1
    print(DAE.model.error(MODEL,Xnoise_current,X_current))

#TEST
XTEST=RAW_DATA
XTEST=(XTEST-mu0)/sigma0
XTEST=(XTEST-minimum)/(maximum-minimum)
    
XR=MODEL.encode(XTEST)

[XR,[mu,sigma]]=norm.normalize.global_z(XR)
    