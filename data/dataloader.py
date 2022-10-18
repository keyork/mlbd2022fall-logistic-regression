"""
@ File Name     :   dataloader.py
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


class DataLoader:
    def __init__(self, dataset, is_valid, fold_id):

        if is_valid:
            self.data = dataset.data[
                fold_id * dataset.fold_size : (fold_id + 1) * dataset.fold_size, :
            ]
            self.label = dataset.label[
                fold_id * dataset.fold_size : (fold_id + 1) * dataset.fold_size, :
            ]
        else:
            self.data = np.vstack(
                (
                    dataset.data[: fold_id * dataset.fold_size, :],
                    dataset.data[(fold_id + 1) * dataset.fold_size :, :],
                )
            )
            self.label = np.vstack(
                (
                    dataset.label[: fold_id * dataset.fold_size, :],
                    dataset.label[(fold_id + 1) * dataset.fold_size :, :],
                )
            )

    def __iter__(self):
        return self

    def __next__(self):
        return self.data, self.label
