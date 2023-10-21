import os
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
from torchvision import models, transforms
from tqdm import tqdm

from train_mrnet import MRDataset, MRNet


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    slice_cams = []
    for s in range(bz):
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv[s].reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            slice_cams.append(cv2.resize(cam_img, size_upsample))
    return slice_cams


def find_last_conv_layer(model):
    last_conv_layer = None
    last_conv_layer_name = None

    # Traverse through all modules
    for name, module in model.named_modules():
        print(f"{name=}, {module}")
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module
            last_conv_layer_name = name

    return last_conv_layer, last_conv_layer_name


def hook_feature(module, input, output):
    feature_blobs.append(output.data.cpu().numpy())


def create_patiens_cam(case, plane):
    patient_id = case["id"]
    mri = case["mri"]

    folder_path = f"./CAMS/{plane}/{patient_id}/"
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    os.makedirs(folder_path + "slices/")
    os.makedirs(folder_path + "cams/")

    params = list(mrnet.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    num_slices = mri.shape[1]
    global feature_blobs
    feature_blobs = []
    mri = mri.to(device)
    logit = mrnet(mri)
    size_upsample = (256, 256)
    feature_conv = feature_blobs[0]

    h_x = F.softmax(logit, dim=1).data.squeeze(0)
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()
    slice_cams = returnCAM(feature_blobs[-1], weight_softmax, idx[:1])

    # for s in tqdm.tqdm(range(num_slices), leave=False):
    for s in range(num_slices):
        slice_pil = transforms.ToPILImage()(mri.cpu()[0][s] / 255)
        slice_pil.save(folder_path + f"slices/{s}.png", dpi=(300, 300))

        img = mri[0][s].cpu().numpy()
        img = img.transpose(1, 2, 0)
        heatmap = cv2.cvtColor(
            cv2.applyColorMap(cv2.resize(slice_cams[s], (256, 256)), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )
        result = heatmap * 0.3 + img * 0.6

        pil_img_cam = Image.fromarray(np.uint8(result))
        pil_img_cam.save(folder_path + f"cams/{s}.png", dpi=(300, 300))


if __name__ == "__main__":
    task = "acl"
    plane = "sagittal"  # 'axial', 'coronal', 'sagittal'
    prefix = ""

    model_name = [
        name
        for name in os.listdir("models/")
        if (task in name) and (plane in name) and (prefix in name)
    ][0]
    print(f"loading model {model_name}")

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    mrnet = torch.load(f"models/{model_name}")
    mrnet = mrnet.to(device)

    _ = mrnet.eval()

    dataset = MRDataset("data/", task, plane, transform=None, train=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )

    res = find_last_conv_layer(mrnet)
    res

    # finalconv_name = "pretrained_model"
    #

    #
    # mrnet._modules.get(finalconv_name).register_forward_hook(hook_feature);

    feature_blobs = []
    mrnet._modules.get("pretrained_model")._modules.get("features")._modules.get(
        "10"
    ).register_forward_hook(hook_feature)

    patients = []

    for i, (image, label, _) in tqdm(enumerate(loader), total=len(loader)):
        patient_data = {}
        patient_data["mri"] = image
        patient_data["label"] = label[0][0][1].item()
        patient_data["id"] = "0" * (4 - len(str(i))) + str(i)
        patients.append(patient_data)

    acl = list(filter(lambda d: d["label"] == 1, patients))
    no_acl = list(filter(lambda d: d["label"] == 0, patients))

    len(acl), len(no_acl)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mrnet = MRNet()
    mrnet = mrnet.to(device)

    _ = mrnet.eval()

    case = patients[94]
    mri = case["mri"]
    mri = mri.to(device)

    case = patients[102]
    mri2 = case["mri"]
    mri2 = mri2.to(device)

    batched_mri = torch.stack([mri, mri2], dim=0)

    mrnet(batched_mri)

    for i in range(len(patients)):
        print(np.shape(patients[i]["mri"]), i)

    np.shape(patients[0]["mri"])

    for person in tqdm(patients):
        create_patiens_cam(person, plane)
