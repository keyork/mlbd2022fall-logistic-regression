"""
@ File Name     :   sgdtoolbox.py
@ Time          :   2022/10/04
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   loss_func() -- cal loss
                    bl_search() -- back line search
                    optimizer() -- update beta
@ Details       :   None
"""


import copy
import numpy as np


def loss_func(pred: np.array, label: np.array):

    loss = ((label - pred) ** 2).sum() / pred.shape[0]
    loss_list = label - pred
    return loss, loss_list


def bl_search(
    label: np.array,
    data: np.array,
    learning_rate,
    alpha,
    eta,
    update_lr,
    model,
    loss,
    loss_list,
):
    learning_rate_1st = learning_rate
    model_bls = copy.deepcopy(model)
    base_beta = model_bls.beta
    grad = (loss_list * data).sum(axis=0)

    while True:
        model_bls.beta = base_beta + learning_rate_1st * grad
        pred_bls = model_bls.forward(data)
        bls_loss, _ = loss_func(pred_bls, label)
        judge_flag_1st = (
            bls_loss <= loss - (alpha * learning_rate_1st * (grad**2)).sum()
        )
        if judge_flag_1st:
            break
        learning_rate_1st = learning_rate_1st * update_lr

    while True:
        model_bls.beta = base_beta + learning_rate_1st * grad
        pred_bls = model_bls.forward(data)
        bls_loss, _ = loss_func(pred_bls, label)
        judge_flag_1st = (
            bls_loss > loss - (alpha * learning_rate_1st * (grad**2)).sum()
        )
        if judge_flag_1st:
            learning_rate_1st = learning_rate_1st * update_lr
            break
        learning_rate_1st = learning_rate_1st / update_lr

    learning_rate_2nd = learning_rate_1st

    while True:
        model_bls.beta = base_beta + learning_rate_2nd * grad
        pred_bls = model_bls.forward(data)
        _, bls_loss_list = loss_func(pred_bls, label)
        bls_grad = -(bls_loss_list * data).sum(axis=0)
        judge_flag_2nd = (bls_grad * grad).sum() < eta * (grad**2).sum()
        if judge_flag_2nd:
            learning_rate_2nd /= update_lr
            break
        learning_rate_2nd = learning_rate_2nd * update_lr
    learning_rate = np.sqrt(learning_rate_1st * learning_rate_2nd)
    return learning_rate


def optimizer(model, loss_list, train_data, learning_rate):

    para_update = (learning_rate * loss_list * train_data).sum(axis=0)
    new_para = model.beta + para_update
    model.update_paras(new_para)
