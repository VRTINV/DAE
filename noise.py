# -*- coding: utf-8 -*-
"""
Licensed @C.TR
"""

import random
import pandas

class noise:
    def make_thresh(X):
        XNEW=pandas.DataFrame([])
        for k in X.index:
            xNEWline=pandas.DataFrame([])
            for y in X.loc[k]:
                if y != 0 and y!=1:
                    p=random.random()
                    if p < 0.1:
                        value=0
                    if p >= 0.1 and p<= 0.9:
                        value=y
                    if p > 0.9:
                        value=1
                if y==0 or y==1:
                    value=y
                  
                xNEWline=pandas.concat([xNEWline,pandas.DataFrame([value])],axis=1,ignore_index=True)
            XNEW=pandas.concat([XNEW,xNEWline],axis=0,ignore_index=True)
        return XNEW
        

