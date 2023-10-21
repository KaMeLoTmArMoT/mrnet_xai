import ctypes
import os

import cv2
import numpy as np
import pandas as pd

font = cv2.FONT_HERSHEY_SIMPLEX
black = (0, 0, 0)
red = (0, 0, 255)
green = (0, 255, 0)


def center_window(w_, h_):
    user32 = ctypes.windll.user32
    w_s, h_s = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    cv2.moveWindow(win_name, int((w_s - w_) / 2), int((h_s - h_) / 2) - 30)


def get_images(source_pth, patient):
    pth_cams = os.path.join(source_pth, patient, "cams")
    pth_slices = os.path.join(source_pth, patient, "slices")

    imgs_cam, imgs_slice = [], []

    names_cam = os.listdir(pth_cams)
    names_slise = os.listdir(pth_slices)

    names_cam.sort(key=lambda x: int(x[:-4]))
    names_slise.sort(key=lambda x: int(x[:-4]))

    for cam_, slice_ in zip(names_cam, names_slise):
        cam_pth = os.path.join(pth_cams, cam_)
        imgs_cam.append(cv2.imread(cam_pth, cv2.IMREAD_COLOR))

        slice_pth = os.path.join(pth_slices, slice_)
        imgs_slice.append(cv2.imread(slice_pth, cv2.IMREAD_COLOR))

    return imgs_cam, imgs_slice


def get_shap_images(source_pth, patient):
    patient = str(int(patient))
    pth_shap = os.path.join(source_pth, patient)
    imgs_shap = []

    names_shap = os.listdir(pth_shap)
    names_shap.sort(key=lambda x: int(x[:-4]))

    for shap_ in names_shap:
        cam_pth = os.path.join(pth_shap, shap_)
        imgs_shap.append(cv2.imread(cam_pth, cv2.IMREAD_COLOR))

    return imgs_shap


def paste(bg_, img_, row, col):
    y1 = pad * row + 256 * (row - 1)
    y2 = pad * row + 256 * row
    x1 = pad * col + 256 * (col - 1)
    x2 = pad * col + 256 * col

    bg_[y1:y2, x1:x2] = img_


def add_text(header_, txt, row, show_plane: str = None):
    pos_x = pad * row + 90 + 256 * (row - 1)
    pox_y = 55
    cv2.putText(header_, txt, (pos_x, pox_y), font, 1, black, 1)

    if show_plane:
        val = df_results.iloc[int(case)][show_plane]
        pred = str(round(val, 2))
        size, _ = cv2.getTextSize(f"{show_plane}:{pred}", font, 1, 1)
        text_width, text_height = size
        cv2.putText(
            header_,
            f"{show_plane}:{pred}",
            (int(pos_x + 40 - text_width / 2), pox_y + 30),
            font,
            1,
            black,
            1,
        )


def get_header():
    header_ = np.full((100, win_w, 3), 255, dtype=np.uint8)

    add_text(header_, f"{show_a_id}/{len(images_cam_a) - 1}", row=1, show_plane="axial")
    add_text(
        header_, f"{show_c_id}/{len(images_cam_c) - 1}", row=2, show_plane="coronal"
    )
    add_text(
        header_, f"{show_s_id}/{len(images_cam_s) - 1}", row=3, show_plane="sagittal"
    )

    pos_y1 = 25

    size, _ = cv2.getTextSize(f"case:{case}", font, 1, 1)
    text_width, text_height = size
    cv2.putText(
        header_,
        f"case:{case}",
        (int(win_w / 2 - text_width / 2), pos_y1),
        font,
        1,
        black,
        1,
    )

    ground_truth = str(df.iloc[int(case)]["label"])
    size, _ = cv2.getTextSize(f"label:{ground_truth}", font, 1, 1)
    text_width, text_height = size
    cv2.putText(
        header_,
        f"label:{ground_truth}",
        (int(win_w / 2 - text_width / 2 - 256), pos_y1),
        font,
        1,
        black,
        1,
    )

    val = df_results.iloc[int(case)]["preds"]
    color = red if val > 0.5 else green
    pred = str(round(val, 2))
    size, _ = cv2.getTextSize(f"pred:{pred}", font, 1, 1)
    text_width, text_height = size
    cv2.putText(
        header_,
        f"pred:{pred}",
        (int(win_w / 2 - text_width / 2 + 256 + 10), pos_y1),
        font,
        1,
        color,
        1,
    )

    status = np.full((30, win_w, 3), 255, dtype=np.uint8)
    pred_lbl = 1 if val > 0.5 else 0
    ground_truth = int(ground_truth)
    if ground_truth == 0:
        if ground_truth == pred_lbl:
            txt = "True Negative"
            color = green
        else:
            txt = "False Positive"
            color = red
    else:  # == 1
        if ground_truth == pred_lbl:
            txt = "True Positive"
            color = green
        else:
            txt = "False Negative"
            color = red

    size, _ = cv2.getTextSize(f"{txt}", font, 1, 1)
    text_width, text_height = size
    cv2.putText(
        status, f"{txt}", (int(win_w / 2 - text_width / 2), 25), font, 1, color, 1
    )
    merged = np.vstack([status, header_])

    return merged


def add_sidebar(img_merge):
    sidebar = np.full((img_merge.shape[0], 110, 3), 255, dtype=np.uint8)

    cv2.putText(sidebar, f"CAMs", (int(15), int(270)), font, 1, black, 2)
    cv2.putText(sidebar, f"IMG", (int(25), int(530)), font, 1, black, 2)
    cv2.putText(sidebar, f"SHAPs", (int(10), int(790)), font, 1, black, 2)
    cv2.putText(sidebar, f"V:{source_shap}", (int(10), int(830)), font, 0.4, red, 1)

    return np.hstack([sidebar, img_merge])


def fill_str(txt):
    if type(txt) != str:
        txt = str(txt)

    while len(txt) < 4:
        txt = "0" + txt
    return txt


if __name__ == "__main__":
    df = pd.read_csv("data/valid-acl.csv", header=None, names=["id", "label"])
    df["id"] = df["id"] - 1130

    df_results = pd.read_csv(
        "data/results.csv",
        header=0,
        names=["id", "labels", "axial", "coronal", "sagittal", "preds"],
    )

    win_name = "preview"
    cv2.namedWindow(win_name)

    pad = 10
    img_h, img_w = 256, 256
    n_rows, n_cols = 3, 3
    win_h = img_h * n_rows + (n_rows + 1) * pad
    win_w = img_w * n_cols + (n_cols + 1) * pad

    # bg = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    bg = np.full((win_h, win_w, 3), 255, dtype=np.uint8)

    # -------------------------------- CAMS
    source = "CAMS"
    source_a = os.path.join(source, "axial")
    source_c = os.path.join(source, "coronal")
    source_s = os.path.join(source, "sagittal")

    cases_a = os.listdir(source_a)
    cases_c = os.listdir(source_c)
    cases_s = os.listdir(source_s)

    cases_a.sort(key=lambda x: int(x))
    cases_c.sort(key=lambda x: int(x))
    cases_s.sort(key=lambda x: int(x))

    scale = 1.0
    case = "0010"
    new_case = case

    images_cam_a, images_slice_a = get_images(source_a, case)
    images_cam_c, images_slice_c = get_images(source_c, case)
    images_cam_s, images_slice_s = get_images(source_s, case)
    show_a_id, show_c_id, show_s_id = (
        int(len(images_cam_a) / 2),
        int(len(images_cam_c) / 2),
        int(len(images_cam_s) / 2),
    )

    # -------------------------------- SHAPS

    shap_sources = ["SHAPS", "SHAPS_V2"]
    shap_sources_id = 0
    source_shap = shap_sources[shap_sources_id]
    source_shap_a = os.path.join(source_shap, "axial")
    source_shap_c = os.path.join(source_shap, "coronal")
    source_shap_s = os.path.join(source_shap, "sagittal")

    cases_shap_a = os.listdir(source_shap_a)
    cases_shap_c = os.listdir(source_shap_c)
    cases_shap_s = os.listdir(source_shap_s)

    cases_shap_a.sort(key=lambda x: int(x))
    cases_shap_c.sort(key=lambda x: int(x))
    cases_shap_s.sort(key=lambda x: int(x))

    images_shap_a = get_shap_images(source_shap_a, case)
    images_shap_c = get_shap_images(source_shap_c, case)
    images_shap_s = get_shap_images(source_shap_s, case)

    # -------------------------------- SHAPS

    header = get_header()

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord("q") or key == 27:
            break

        if key == ord("d"):  # d = 100
            print("use debug mode")

        if key == ord("v") and len(shap_sources) > 1:  # v = 118
            if shap_sources_id + 1 < len(shap_sources):
                shap_sources_id += 1
                print("next", shap_sources_id, len(shap_sources))
            else:
                shap_sources_id = 0
                print("first")

            source_shap = shap_sources[shap_sources_id]
            source_shap_a = os.path.join(source_shap, "axial")
            source_shap_c = os.path.join(source_shap, "coronal")
            source_shap_s = os.path.join(source_shap, "sagittal")

            cases_shap_a = os.listdir(source_shap_a)
            cases_shap_c = os.listdir(source_shap_c)
            cases_shap_s = os.listdir(source_shap_s)

            cases_shap_a.sort(key=lambda x: int(x))
            cases_shap_c.sort(key=lambda x: int(x))
            cases_shap_s.sort(key=lambda x: int(x))

            images_shap_a = get_shap_images(source_shap_a, case)
            images_shap_c = get_shap_images(source_shap_c, case)
            images_shap_s = get_shap_images(source_shap_s, case)

        # down
        if key == 49 and show_a_id - 1 >= 0:
            show_a_id -= 1
        if key == 50 and show_c_id - 1 >= 0:
            show_c_id -= 1
        if key == 51 and show_s_id - 1 >= 0:
            show_s_id -= 1

        # up
        if key == 55 and show_a_id + 1 < len(images_cam_a):
            show_a_id += 1
        if key == 56 and show_c_id + 1 < len(images_cam_c):
            show_c_id += 1
        if key == 57 and show_s_id + 1 < len(images_cam_s):
            show_s_id += 1

        # case switch key:[ENTER]
        if key == 13:
            min_case_id, max_case_id = "0000", fill_str(len(cases_a) - 1)
            case_tmp = input(f"enter case id in range [{min_case_id}]-[{max_case_id}]")
            if case_tmp and case_tmp.isnumeric():
                if 0 <= int(case_tmp) < len(cases_a):
                    new_case = fill_str(case_tmp)

        # left right
        if key == 52 and int(case) - 1 >= 0:
            new_case = str(int(case) - 1)
            new_case = fill_str(new_case)
        if key == 54 and int(case) + 1 < len(cases_a):
            new_case = str(int(case) + 1)
            new_case = fill_str(new_case)

        if case != new_case:
            case = new_case

            images_cam_a, images_slice_a = get_images(source_a, case)
            images_cam_c, images_slice_c = get_images(source_c, case)
            images_cam_s, images_slice_s = get_images(source_s, case)

            images_shap_a = get_shap_images(source_shap_a, case)
            images_shap_c = get_shap_images(source_shap_c, case)
            images_shap_s = get_shap_images(source_shap_s, case)

            show_a_id, show_c_id, show_s_id = (
                int(len(images_cam_a) / 2),
                int(len(images_cam_c) / 2),
                int(len(images_cam_s) / 2),
            )

        paste(bg, images_cam_a[show_a_id], 1, 1)
        paste(bg, images_slice_a[show_a_id], 2, 1)
        paste(bg, images_shap_a[show_a_id], 3, 1)

        paste(bg, images_cam_c[show_c_id], 1, 2)
        paste(bg, images_slice_c[show_c_id], 2, 2)
        paste(bg, images_shap_c[show_c_id], 3, 2)

        paste(bg, images_cam_s[show_s_id], 1, 3)
        paste(bg, images_slice_s[show_s_id], 2, 3)
        paste(bg, images_shap_s[show_s_id], 3, 3)

        header = get_header()
        img_merge = np.vstack([header, bg])
        img_merge = add_sidebar(img_merge)

        # scale
        if key == 61 and scale + 0.1 < 3:
            scale += 0.1
        if key == 45 and scale - 0.1 > 0.5:
            scale -= 0.1
        if key == 48:
            scale = 1.0

        if scale != 1.0:
            width = int(img_merge.shape[1] * scale)
            height = int(img_merge.shape[0] * scale)
            img_merge = cv2.resize(
                img_merge, (width, height), interpolation=cv2.INTER_CUBIC
            )
            center_window(width, height)

        cv2.imshow(win_name, img_merge)

    cv2.destroyAllWindows()
