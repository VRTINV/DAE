# -*- coding: utf-8 -*-
"""
Licensed @C.TR
"""

import pandas
    
class normalize:
    def global_z(X):
        mu=X.mean().pop(0)
        sigma=X.std().pop(0)
        return (X-mu)/sigma,[mu,sigma]
    def line_max(x):
        m=x.min()
        M=x.max()
        return (x-m)/(M-m),[m,M]
    def global_m(X):
        U=pandas.DataFrame([])
        mvec=pandas.DataFrame([])
        Mvec=pandas.DataFrame([])
        for k in X.index:
            h,[m,M]=normalize.line_max(X.loc[k])
            U=pandas.concat([U,pandas.DataFrame([h])],axis=0,ignore_index=True)
            mvec=pandas.concat([mvec,pandas.DataFrame([m])],axis=0,ignore_index=True)
            Mvec=pandas.concat([Mvec,pandas.DataFrame([M])],axis=0,ignore_index=True)
        return U,[mvec.mean(),Mvec.mean()]
            

        
