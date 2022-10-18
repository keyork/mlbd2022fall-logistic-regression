"""
@ File Name     :   dataset.py
@ Time          :   2022/10/17
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   func1() -- func desc1
@ Class List    :   Class1 -- class1 desc1
@ Details       :   None
"""


import pandas as pd
import numpy as np


class DataSet:
    def __init__(self, data_path, fold_num, shuffle, norm):

        raw_df = pd.read_csv(data_path)
        if norm:
            raw_df = raw_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        raw_df = raw_df.drop(columns=["x2"])
        raw_data = raw_df.values

        raw_data = np.insert(raw_data, 1, values=1, axis=1)
        self.data_num = raw_data.shape[0]
        if shuffle:
            data_idx = np.random.permutation(self.data_num)
            full_data = raw_data[data_idx]
        else:
            full_data = raw_data
        self.data = full_data[:, 1:]
        self.label = full_data[:, :1]
        # print(self.data)

        self.fold_num = fold_num
        self.fold_size = self.label.shape[0] // self.fold_num
