"""
@ File Name     :   regression.py
@ Time          :   2022/10/17
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   func1() -- func desc1
@ Class List    :   Class1 -- class1 desc1
@ Details       :   None
"""


import numpy as np


class LogisticRegression:
    def __init__(self):

        self.beta = np.array([0, 0, 0])

    def update_paras(self, new_beta):

        self.beta = new_beta

    def reset(self):

        self.beta = np.array([0, 0, 0])

    def forward(self, data):

        return 1 / (1 + np.exp(np.array([(-self.beta * data).sum(axis=1)]).T))
