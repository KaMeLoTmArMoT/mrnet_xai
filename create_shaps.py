import gc
import os

import cv2
import numpy as np
import shap
import torch
from tqdm import tqdm

from train_mrnet import MRDataset


def clear(additional: list = None):
    variables_to_delete = [
        "background_samples",
        "background_tensor",
        "shap_values",
        "indexes",
        "e",
    ]
    if additional:
        variables_to_delete.extend(additional)
    for var in variables_to_delete:
        try:
            del globals()[var]
        except KeyError:
            pass  # Variable was not defined, so just pass
    gc.collect()


def create_patiens_shaps(task="acl"):
    planes = ["sagittal", "coronal", "axial"]
    pbar = tqdm(total=len(planes) * 120, position=0, leave=True)
    for plane in planes:
        model_name = [
            name
            for name in os.listdir("models/")
            if (task in name) and (plane in name) and (prefix in name)
        ][0]
        print(f"Loading model for {plane}:{model_name}")

        dataset = MRDataset("data/", task, plane, transform=None, train=False)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
        )

        patients = []

        # for i, (image, label, _) in tqdm.tqdm(enumerate(loader), total=len(loader)):
        for i, (image, label, _) in enumerate(loader):
            patient_data = {}
            patient_data["mri"] = image
            patient_data["label"] = label[0][0][1].item()
            patient_data["id"] = "0" * (4 - len(str(i))) + str(i)
            patients.append(patient_data)

        model = torch.load(f"models/{model_name}").eval()
        model = model.cuda()

        slices = {}
        for i in range(len(patients)):
            slice = np.shape(patients[i]["mri"])[1]
            if slice in slices:
                slices[slice] += 1
            else:
                slices[slice] = 1

        slices = sorted(slices.items(), key=lambda x: x[1], reverse=True)

        for images_in_set, sets_count in slices:
            all_images_exist_for_all_patients = all(
                os.path.exists(
                    os.path.join(
                        f"SHAPS_V2/{plane}/{tensor_to_id_map[set_idx]}/",
                        f"{img_idx}.png",
                    )
                )
                for set_idx in range(sets_count)
                for img_idx in range(images_in_set)
            )
            if all_images_exist_for_all_patients:
                pbar.update(
                    sets_count
                )  # Update progress bar by the number of patients in the current set
                continue

            background_samples = []
            tensor_to_id_map, n = {}, 0
            for patient_idx in range(len(patients)):
                if np.shape(patients[patient_idx]["mri"])[1] == images_in_set:
                    tensor_to_id_map[n] = patient_idx
                    background_samples.append(patients[patient_idx]["mri"])
                    n += 1

            background_tensor = torch.cat(background_samples, axis=0)
            background_tensor = background_tensor.to(device)
            e = shap.GradientExplainer(model, background_tensor)

            # for set_idx in tqdm.tqdm(range(sets_count)):
            for set_idx in range(sets_count):
                real_set_id = tensor_to_id_map[set_idx]

                # Check if all images for this set already exist
                all_images_exist = all(
                    os.path.exists(
                        os.path.join(
                            f"SHAPS_V2/{plane}/{real_set_id}/", f"{img_idx}.png"
                        )
                    )
                    for img_idx in range(images_in_set)
                )
                if all_images_exist:
                    pbar.update(1)
                    continue

                to_explain = background_tensor[set_idx : set_idx + 1]
                # shap_values, indexes = e.shap_values(to_explain, nsamples=1)
                shap_values, indexes = e.shap_values(to_explain, nsamples=images_in_set)

                data = shap_values[0]
                for img_idx, img in enumerate(data):
                    a = np.sum(img, axis=0)
                    summed_data = np.zeros(
                        (a.shape[0] // region_size, a.shape[1] // region_size)
                    )

                    summed_img = np.sum(img, axis=0)
                    for x in range(0, summed_img.shape[0], region_size):
                        for y in range(0, summed_img.shape[1], region_size):
                            region_sum = np.sum(
                                summed_img[x : x + region_size, y : y + region_size]
                            )
                            summed_data[x // region_size, y // region_size] = region_sum

                    alpha = 0.3

                    v = to_explain.cpu().numpy()[0][img_idx]
                    v_transposed = np.transpose(v, (1, 2, 0))
                    v_transposed = v_transposed.astype(np.uint8)

                    resized_mask = cv2.resize(
                        summed_data, (v_transposed.shape[1], v_transposed.shape[0])
                    )
                    norm_mask = cv2.normalize(
                        resized_mask,
                        None,
                        alpha=0,
                        beta=255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U,
                    )
                    colored_mask = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)
                    blended = cv2.addWeighted(
                        v_transposed, 1 - alpha, colored_mask, alpha, 0
                    )

                    path = f"SHAPS_V3/{plane}/{real_set_id}/"
                    os.makedirs(path, exist_ok=True)
                    img_path = os.path.join(path, f"{img_idx}.png")
                    cv2.imwrite(img_path, blended)
                pbar.update(1)

            # clear memory
            clear()
    pbar.close()


if __name__ == "__main__":
    prefix = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    region_size = 32
    create_patiens_shaps()
