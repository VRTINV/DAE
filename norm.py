# -*- coding: utf-8 -*-
"""
Licensed @C.TR
"""

import numpy
    
class normalize:
    def global_z(X):
        mu=X.mean()
        sigma=X.std()
        return numpy.array((X-mu)/sigma),[mu,sigma]
    def line_max(x):
        m=x.min()
        M=x.max()
        return (x-m)/(M-m),[m,M]
    def global_m(X):
        U=numpy.array([])
        mvec=numpy.array([])
        Mvec=numpy.array([])
        for x in X:
            h,[m,M]=normalize.line_max(x)
            if len(U)==0:
                U=numpy.array([h])
            if len(U)>1:
                U=numpy.concatenate([U,[h]],axis=0)
            mvec=numpy.concatenate([mvec,[m]],axis=0)
            Mvec=numpy.concatenate([Mvec,[M]],axis=0)
        return numpy.array(U),[mvec.mean(),Mvec.mean()]
            

        
