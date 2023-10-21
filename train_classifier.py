import os

import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from train_mrnet import MRDataset


def extract_predictions(task, plane, train=True):
    assert task in ["acl", "meniscus", "abnormal"]
    assert plane in ["axial", "coronal", "sagittal"]

    models = os.listdir("models/")

    model_name = list(filter(lambda name: task in name and plane in name, models))[0]
    model_path = f"models/{model_name}"

    mrnet = torch.load(model_path)
    _ = mrnet.eval()

    train_dataset = MRDataset("data/", task, plane, transform=None, train=train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, drop_last=False
    )
    predictions = []
    labels = []
    with torch.no_grad():
        # for image, label, _ in tqdm_notebook(train_loader):
        for image, label, _ in tqdm(train_loader):
            logit = mrnet(image.cuda())
            print(logit)
            prediction = torch.sigmoid(logit)
            print(prediction)
            predictions.append(prediction[0][1].item())
            print(prediction[0][1].item())
            print("-----------------")
            labels.append(label[0][0][1].item())

    return predictions, labels


if __name__ == "__main__":
    predictions, labels = extract_predictions("acl", "axial", train=False)

    task = "acl"
    results = {}

    for plane in ["axial", "coronal", "sagittal"]:
        predictions, labels = extract_predictions(task, plane)
        results["labels"] = labels
        results[plane] = predictions

    X = np.zeros((len(predictions), 3))
    X[:, 0] = results["axial"]
    X[:, 1] = results["coronal"]
    X[:, 2] = results["sagittal"]

    y = np.array(labels)

    logreg = LogisticRegression(solver="lbfgs")
    logreg.fit(X, y)

    task = "acl"
    results_val = {}

    for plane in ["axial", "coronal", "sagittal"]:
        predictions, labels = extract_predictions(task, plane, train=False)
        results_val["labels"] = labels
        results_val[plane] = predictions

    X_val = np.zeros((len(results_val["axial"]), 3))
    X_val[:, 0] = results_val["axial"]
    X_val[:, 1] = results_val["coronal"]
    X_val[:, 2] = results_val["sagittal"]

    y_val = np.array(results_val["labels"])

    y_pred = logreg.predict_proba(X_val)[:, 1]
    metrics.roc_auc_score(y_val, y_pred)
