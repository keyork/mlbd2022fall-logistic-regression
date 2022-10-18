"""
@ File Name     :   train.py
@ Time          :   2022/10/17
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   None
@ Function List :   func1() -- func desc1
@ Class List    :   Class1 -- class1 desc1
@ Details       :   None
"""


import copy
import argparse

from data.dataloader import DataLoader
from data.dataset import DataSet
from model.regression import LogisticRegression
from utils.toolbox import LOGGER, str2bool
from utils.sgdtoolbox import loss_func, optimizer, bl_search
from utils.drawtoolbox import draw_3d, draw_loss, draw_loss_single
from tqdm import tqdm


def train(config, dataset):
    LOGGER.info("Start Train")
    model = LogisticRegression()
    learning_rate = config.lr
    min_loss = 1e5
    best_model = None
    for fold_id in range(5):
        LOGGER.info("Valid Fold ID: {}".format(fold_id))
        train_loader = DataLoader(dataset, False, fold_id)
        valid_loader = DataLoader(dataset, True, fold_id)
        train_data, train_label = train_loader.__next__()
        valid_data, valid_label = valid_loader.__next__()
        model.reset()
        train_loss_log = []
        for iter in tqdm(range(config.iter), ncols=70, colour="blue"):
            # Train
            y_pred = model.forward(train_data)
            loss, loss_list = loss_func(y_pred, train_label)

            train_loss_log.append(loss)
            if config.bls:
                learning_rate = bl_search(
                    train_label,
                    train_data,
                    learning_rate,
                    5e-3,
                    0.1,
                    0.8,
                    model,
                    loss,
                    loss_list,
                )
            else:
                learning_rate /= 1.001
            optimizer(model, loss_list, train_data, learning_rate)

            # Valid
            val_pred = model.forward(valid_data)
            val_loss, val_loss_list = loss_func(val_pred, valid_label)

            if val_loss < 1e-2:
                break

        # print(val_loss)
        # print(model.beta)
        if val_loss < min_loss:
            min_loss = val_loss
            best_model = copy.deepcopy(model)
            draw_data = [
                train_data,
                train_label,
                valid_data,
                valid_label,
                best_model,
            ]
            draw_loss_log = train_loss_log
    LOGGER.info("Result")
    print("Valid Pred : ", end="")
    print(1.0 * (val_pred.T >= 0.5))
    print("Valid Label: ", end="")
    print(valid_label.T)
    print(
        "Valid Acc = {}%".format(
            (
                float((1.0 * (1.0 * (val_pred >= 0.5) == valid_label)).sum())
                / len(val_pred)
            )
            * 100
        )
    )
    print("beta = ", end="")
    print(best_model.beta)
    return draw_data, draw_loss_log


def main(config):

    LOGGER.info("config list")
    print("\tfold: {}".format(config.fold))
    print("\tlearning rate: {}".format(config.lr))
    print("\titer: {}".format(config.iter))
    print("\tback line search: {}".format(config.bls))
    print("\tnormlize: {}".format(config.norm))
    print("\tcompare normalize: {}".format(config.compnorm))
    print("\tcompare back line search: {}".format(config.compbls))
    print("\timg dir: {}".format(config.img_path))
    print("\tdata path: {}".format(config.data_path))

    dataset = DataSet(config.data_path, config.fold, True, config.norm)

    if config.compnorm:
        loss_path = (
            config.img_path
            + "loss-fold_{}-iter_{}-bls_{}-lr_{}-norm_comp.png".format(
                config.fold, config.iter, config.bls, config.lr
            )
        )
        loss_list = []
        x_list = []
        for config.norm in [True, False]:
            dataset = DataSet(config.data_path, config.fold, True, config.norm)
            img_path = (
                config.img_path
                + "img-fold_{}-iter_{}-bls_{}-lr_{}-norm_{}-compnorm.png".format(
                    config.fold, config.iter, config.bls, config.lr, config.norm
                )
            )
            draw_data, draw_loss_log = train(config, dataset)
            x_log = [i for i in range(len(draw_loss_log))]
            loss_list.append(draw_loss_log)
            x_list.append(x_log)
            LOGGER.info("Draw 3D IMG")
            draw_3d(draw_data, config.norm, img_path)
            print("Path -> {}".format(img_path))
        LOGGER.info("Draw Loss Curve")
        draw_loss(
            x_list, loss_list, ["Norm", "no-Norm"], loss_path, "loss curve & normalize"
        )
        print("Path -> {}".format(loss_path))
    elif config.compbls:
        loss_path = (
            config.img_path
            + "loss-fold_{}-iter_{}-bls_comp-lr_{}-norm_{}.png".format(
                config.fold, config.iter, config.lr, config.norm
            )
        )
        loss_list = []
        x_list = []
        for config.bls in [True, False]:
            img_path = (
                config.img_path
                + "img-fold_{}-iter_{}-bls_{}-lr_{}-norm_{}-compbls.png".format(
                    config.fold, config.iter, config.bls, config.lr, config.norm
                )
            )
            draw_data, draw_loss_log = train(config, dataset)
            x_log = [i for i in range(len(draw_loss_log))]
            loss_list.append(draw_loss_log)
            x_list.append(x_log)
            LOGGER.info("Draw 3D IMG")
            draw_3d(draw_data, config.norm, img_path)
            print("Path -> {}".format(img_path))
        LOGGER.info("Draw Loss Curve")
        draw_loss(
            x_list,
            loss_list,
            ["Back Line Search", "no-Back Line Search"],
            loss_path,
            "loss curve & back line search",
        )
        print("Path -> {}".format(loss_path))
    else:
        img_path = (
            config.img_path
            + "img-fold_{}-iter_{}-bls_{}-lr_{}-norm_{}.png".format(
                config.fold, config.iter, config.bls, config.lr, config.norm
            )
        )
        loss_path = (
            config.img_path
            + "loss-fold_{}-iter_{}-bls_{}-lr_{}-norm_{}.png".format(
                config.fold, config.iter, config.bls, config.lr, config.norm
            )
        )
        draw_data, draw_loss_log = train(config, dataset)
        LOGGER.info("Draw 3D IMG")
        draw_3d(draw_data, config.norm, img_path)
        print("Path -> {}".format(img_path))
        x_log = [i for i in range(len(draw_loss_log))]
        LOGGER.info("Draw Loss Curve")
        draw_loss_single(x_log, draw_loss_log, loss_path)
        print("Path -> {}".format(loss_path))


if __name__ == "__main__":
    LOGGER.warning("Start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=5, help="fold number")
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--iter", type=int, default=500, help="iter num")
    parser.add_argument("--bls", type=str2bool, default=True, help="back line search")
    parser.add_argument("--norm", type=str2bool, default=True, help="normlize")
    parser.add_argument(
        "--compnorm", type=str2bool, default=False, help="compare normlize"
    )
    parser.add_argument(
        "--compbls", type=str2bool, default=False, help="compare back line search"
    )
    parser.add_argument("--img_path", type=str, default="./img/", help="img path")
    parser.add_argument(
        "--data_path", type=str, default="./data/LG_data.CSV", help="data path"
    )
    args = parser.parse_args()

    main(args)
    LOGGER.warning("Done")
