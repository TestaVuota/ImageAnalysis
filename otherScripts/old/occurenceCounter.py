# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:12:31 2022

@author: nicol
"""

import numpy
a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
unique, counts = numpy.unique(a, return_counts=True)

res = dict(zip(unique, counts))
res = numpy.array(counts)
print('res',res)
# {0: 7, 1: 4, 2: 1, 3: 2, 4: 1}