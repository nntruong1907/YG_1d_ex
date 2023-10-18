# python test_img.py --model fcnn1d --data test/Warrior3.jpg

import os
import tempfile
import pickle
import argparse
import tensorflow as tf
import numpy as np
import cv2
from def_lib import (
    get_skeleton,
    predict_pose,
    draw_prediction_on_image,
    resizew800
)
from def_lib import find_newest_model_with_prefix

from datetime import datetime
from keras.models import model_from_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="fcnn1d",
        help="model name: svm / fcnn1d / conv1d",
    )

    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="test/Cobra.jpg",
        help="image to be detected",
    )
    args = parser.parse_args()

    list_dir = [
        "./results",
    ]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    class_names = [
        "Chair",
        "Cobra",
        "Dolphin Plank",
        "Downward-Facing Dog",
        "Plank",
        "Side Plank",
        "Tree",
        "Warrior III",
        "Warrior II",
        "Warrior I",
    ]

    newest_model = find_newest_model_with_prefix("save_models", args.model)
    # model = tf.keras.models.load_model("save_models/" + newest_model + ".h5")
    if args.model not in ['svm']:
        # load model da duoc huan luyen
        json_file = open("save_models/" + newest_model + ".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("save_models/" + newest_model + ".h5")
    else:
        with open(f"save_models/{newest_model}.pkl", "rb") as model_file:
            model = pickle.load(model_file)

    img_arr = cv2.imread(args.data)
    if img_arr.shape[1] > 800:
        # Tạo một thư mục tạm để lưu ảnh
        with tempfile.TemporaryDirectory() as temp_dir:
            img_resized = resizew800(img_arr)
            temp_image_path = os.path.join(temp_dir, 'resized_image.jpg')
            cv2.imwrite(temp_image_path, img_resized)

            image = tf.io.read_file(temp_image_path)
            image = tf.image.decode_image(image)
    else: 
        image = tf.io.read_file(args.data)
        image = tf.image.decode_image(image)

    person, lm_pose = get_skeleton(image)

    class_name_pred, acc_pred, predict = predict_pose(args.model, model, lm_pose, class_names)
    acc = round(acc_pred * 100, 2)

    print("="*100)
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {predict[0][i]:f}")
    print("="*100)
    print("Input shape:", lm_pose.shape)
    print("Output shape:", predict.shape)
    print("="*100)

    """Draw on image"""
    
    font = cv2.FONT_HERSHEY_DUPLEX
    org = (10, 40)
    fontScale = 1
    color = (19, 255, 30)
    thickness = 1

    image = np.array(image)
    cv2.putText(
        image,
        class_name_pred + " | " + str(acc_pred),
        org,
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    image = draw_prediction_on_image(
        image, person, crop_region=None, close_figure=False, keep_input_size=True
    )

    curr_datetime = datetime.now().strftime("%Hh%Mm%Ss %d_%m_%Y ")
    r = f'{args.model}_{acc_pred}_{curr_datetime}'
    image_pred_path = "./results/draw_skeleton %s.png" % r
    image_result_path = "./results/result %s.png" % r

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_pred_path, image)

    """--------------------------------------- SHOW IMAGE -------------------------------------------"""
    # Read First Image
    img1 = cv2.imread(args.data)
    if img1.shape[1] > 800:
        # Tạo một thư mục tạm để lưu ảnh
        with tempfile.TemporaryDirectory() as temp_dir:
            img_resized = resizew800(img_arr)
            temp_image_path = os.path.join(temp_dir, 'resized_image.jpg')
            cv2.imwrite(temp_image_path, img_resized)
            img1 = cv2.imread(temp_image_path)
    # Read Second Image
    img2 = cv2.imread(image_pred_path)
    # concatenate image Horizontally
    Hori = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(image_result_path, Hori)

    # # concatenate image Vertically
    # Verti = np.concatenate((img1, img2), axis=0)

    cv2.imshow("CLASSIFICATION OF YOGA POSE", Hori)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
