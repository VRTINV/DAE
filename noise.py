# -*- coding: utf-8 -*-
"""
Licensed @C.TR
"""

import numpy

class noise:
    def make_thresh(X):
        XNEW=numpy.array([])
        for x in X:
            xNEWline=numpy.array([])
            for y in x:
                if y != 0 and y!=1:
                    p=numpy.random.random()
                    if p < 0.1:
                        value=0
                    if p >= 0.1 and p<= 0.9:
                        value=y
                    if p > 0.9:
                        value=1
                if y==0 or y==1:
                    value=y
                xNEWline=numpy.concatenate([xNEWline,[value]],axis=0)
            if len(XNEW)==0:
                XNEW=numpy.array([xNEWline])
            if len(XNEW)>1:
                XNEW=numpy.concatenate([XNEW,[xNEWline]],axis=0)
        return XNEW
        

