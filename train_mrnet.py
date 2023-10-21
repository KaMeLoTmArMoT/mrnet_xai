import argparse
import os
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from dotmap import DotMap
from sklearn import metrics
from tensorboardX import SummaryWriter
from torchsample.transforms import Compose, RandomFlip, RandomRotate, RandomTranslate
from torchvision import models, transforms


def train_model(
    model, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every=100
):
    _ = model.train()

    if torch.cuda.is_available():
        model.cuda()

    y_preds = []
    y_trues = []
    losses = []

    for i, (image, label, weight) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float())

        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0][1]))
        y_preds.append(probas[0][1].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar("Train/Loss", loss_value, epoch * len(train_loader) + i)
        writer.add_scalar("Train/AUC", auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print(
                """[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| avg train loss {4} | train auc : {5} | lr : {6}""".format(
                    epoch + 1,
                    num_epochs,
                    i,
                    len(train_loader),
                    np.round(np.mean(losses), 4),
                    np.round(auc, 4),
                    current_lr,
                )
            )

    writer.add_scalar("Train/AUC_epoch", auc, epoch + i)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    return train_loss_epoch, train_auc_epoch


class MRNet(nn.Module):
    def __init__(self, backbone="alexnet"):
        super().__init__()
        # self.pretrained_model = models.alexnet()
        self.pretrained_model = models.alexnet(
            weights=models.AlexNet_Weights.IMAGENET1K_V1
        )
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(256, 2)

    def forward(self, x):
        if len(x.shape) == 6 or len(x.shape) == 5:
            # Batch processing
            if len(x.shape) == 6:
                batch_size, _, num_slices, c, h, w = x.size()
            else:
                batch_size, num_slices, c, h, w = x.size()
            x = x.view(-1, c, h, w)
            features = self.pretrained_model.features(x)
            pooled_features = self.pooling_layer(features)
            pooled_features = pooled_features.view(batch_size, num_slices, -1)
            flattened_features, _ = torch.max(pooled_features, dim=1)
        else:
            # Original single-sample behavior
            x = torch.squeeze(x, dim=0)
            features = self.pretrained_model.features(x)
            pooled_features = self.pooling_layer(features)
            pooled_features = pooled_features.view(pooled_features.size(0), -1)
            flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]

        output = self.classifer(flattened_features)  # Classifier
        return output


class MRDataset(data.Dataset):
    def __init__(self, root_dir, task, plane, train=True, transform=None, weights=None):
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.train = train
        if self.train:
            self.folder_path = self.root_dir + "train/{0}/".format(plane)
            self.records = pd.read_csv(
                self.root_dir + "train-{0}.csv".format(task),
                header=None,
                names=["id", "label"],
            )
        else:
            transform = None
            self.folder_path = self.root_dir + "valid/{0}/".format(plane)
            self.records = pd.read_csv(
                self.root_dir + "valid-{0}.csv".format(task),
                header=None,
                names=["id", "label"],
            )

        self.records["id"] = self.records["id"].map(
            lambda i: "0" * (4 - len(str(i))) + str(i)
        )
        self.paths = [
            self.folder_path + filename + ".npy"
            for filename in self.records["id"].tolist()
        ]
        self.labels = self.records["label"].tolist()

        self.transform = transform
        if weights is None:
            pos = np.sum(self.labels)
            neg = len(self.labels) - pos
            self.weights = torch.FloatTensor([1, neg / pos])
        else:
            self.weights = torch.FloatTensor(weights)
        # print(self.weights)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        label = self.labels[index]
        if label == 1:
            label = torch.FloatTensor([[0, 1]])
        elif label == 0:
            label = torch.FloatTensor([[1, 0]])

        if self.transform:
            array = self.transform(array)
        else:
            array = np.stack((array,) * 3, axis=1)
            array = torch.FloatTensor(array)

        # if label.item() == 1:
        #     weight = np.array([self.weights[1]])
        #     weight = torch.FloatTensor(weight)
        # else:
        #     weight = np.array([self.weights[0]])
        #     weight = torch.FloatTensor(weight)

        return array, label, self.weights


def evaluate_model(
    model, val_loader, epoch, num_epochs, writer, current_lr, log_every=20
):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []

    for i, (image, label, weight) in enumerate(val_loader):
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float())

        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0][1]))
        y_preds.append(probas[0][1].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar("Val/Loss", loss_value, epoch * len(val_loader) + i)
        writer.add_scalar("Val/AUC", auc, epoch * len(val_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print(
                """[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | avg val loss {4} | val auc : {5} | lr : {6}""".format(
                    epoch + 1,
                    num_epochs,
                    i,
                    len(val_loader),
                    np.round(np.mean(losses), 4),
                    np.round(auc, 4),
                    current_lr,
                )
            )

    writer.add_scalar("Val/AUC_epoch", auc, epoch + i)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)
    return val_loss_epoch, val_auc_epoch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def run(args):
    log_root_folder = "./logs/{0}/{1}/".format(args.task, args.plane)
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    augmentor = Compose(
        [
            transforms.Lambda(lambda x: torch.Tensor(x)),
            RandomRotate(25),
            RandomTranslate([0.11, 0.11]),
            RandomFlip(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
        ]
    )

    train_dataset = MRDataset(
        "./data/", args.task, args.plane, transform=augmentor, train=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, drop_last=False
    )

    validation_dataset = MRDataset("./data/", args.task, args.plane, train=False)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=-True, drop_last=False
    )

    mrnet = MRNet()
    # print("load dict start")
    # mrnet = torch.load("models/model_prefix_acl_axial_val_auc_0.8173_train_auc_0.8291_epoch_11.pth")
    # print("load dict done")

    if torch.cuda.is_available():
        mrnet = mrnet.cuda()

    optimizer = optim.Adam(mrnet.parameters(), lr=args.lr, weight_decay=0.1)

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.3, threshold=1e-4, verbose=True
        )
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=args.gamma
        )

    best_val_loss = float("inf")
    best_val_auc = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience
    log_every = args.log_every

    t_start_training = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)

        t_start = time.time()

        train_loss, train_auc = train_model(
            mrnet,
            train_loader,
            epoch,
            num_epochs,
            optimizer,
            writer,
            current_lr,
            log_every,
        )
        val_loss, val_auc = evaluate_model(
            mrnet, validation_loader, epoch, num_epochs, writer, current_lr
        )

        if args.lr_scheduler == "plateau":
            scheduler.step(val_loss)
        elif args.lr_scheduler == "step":
            scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        print(
            "train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
                train_loss, train_auc, val_loss, val_auc, delta
            )
        )

        iteration_change_loss += 1
        print("-" * 30)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            if bool(args.save_model):
                file_name = f"model_{args.prefix_name}_{args.task}_{args.plane}_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_epoch_{epoch + 1}.pth"

                os.makedirs("./models/", exist_ok=True)
                for f in os.listdir("./models/"):
                    if (
                        (args.task in f)
                        and (args.plane in f)
                        and (args.prefix_name in f)
                    ):
                        os.remove(f"./models/{f}")

                torch.save(mrnet, f"./models/{file_name}")
                print(f"Saving new best epoch with {val_auc=}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print(
                "Early stopping after {0} iterations without the decrease of the val loss".format(
                    iteration_change_loss
                )
            )
            break

    t_end_training = time.time()
    print(f"training took {t_end_training - t_start_training} s")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", type=str, required=True, choices=["abnormal", "acl", "meniscus"]
    )

    parser.add_argument(
        "-p",
        "--plane",
        type=str,
        required=True,
        choices=["sagittal", "coronal", "axial"],
    )

    parser.add_argument("--prefix_name", type=str, required=True)
    parser.add_argument("--augment", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--lr_scheduler", type=str, default="plateau", choices=["plateau", "step"]
    )

    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--flush_history", type=int, choices=[0, 1], default=0)
    parser.add_argument("--save_model", type=int, choices=[0, 1], default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=100)

    args = parser.parse_args()
    return args


def get_args(**kwargs):
    args = {}

    args["task"] = "abnormal"
    args["plane"] = "sagittal"
    args["prefix_name"] = "prefix"
    args["augment"] = 1
    args["lr_scheduler"] = "plateau"
    args["gamma"] = 0.5
    args["epochs"] = 50
    args["lr"] = 1e-5
    args["flush_history"] = 0
    args["save_model"] = 1
    args["patience"] = 5
    args["log_every"] = 100

    for key, val in kwargs.items():
        print(f"override {key} from {args[key]} -> {val}")
        args[key] = val

    return args


if __name__ == "__main__":
    # %load_ext tensorboard
    # %tensorboard --logdir logs

    # task 'abnormal', 'acl', 'meniscus'
    # plane 'sagittal', 'coronal', 'axial'

    epochs = 35
    task = "acl"

    # args = get_args(epochs=epochs, task=task, plane="sagittal", patience=7)
    # args = get_args(epochs=epochs, task=task, plane="coronal", patience=7)
    args = get_args(epochs=epochs, task=task, plane="axial", patience=7)
    mapped = DotMap(args)
    run(mapped)
