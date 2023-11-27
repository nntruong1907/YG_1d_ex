# python test_cam.py --model fcnn1d

import pickle

import argparse
import tensorflow as tf
import numpy as np
import cv2
from def_lib import (
    draw_prediction_on_image,
    get_skeleton,
)
from def_lib import find_newest_model_with_prefix, predict_pose, draw_class_name_on_image

from keras.models import model_from_json


def gen_video(model_name, model, class_names):
    cap = cv2.VideoCapture(0)
    label = "waiting"
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # Reshape Image
        if ret == True:
            img = frame.copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # img = cv2.resize(img, (854, 480))
            # img = cv2.resize(img, (640, 360))
            img = tf.convert_to_tensor(img, dtype=tf.uint8)
            i = i + 1

            print(f"Start detect: frame {i}")
            person, lm_pose = get_skeleton(img)

            label, acc_pred, predict = predict_pose(model_name, model, lm_pose, class_names)

            img = np.array(img)
            img = draw_prediction_on_image(
                img, person, crop_region=None, close_figure=False, keep_input_size=True
            )

            img = draw_class_name_on_image(label, img)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Yoga Detection", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="fcnn1d",
        help="model name: svm / fcnn1d / conv1d",
    )

    args = parser.parse_args()

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

    if args.model not in ["svm"]:
        # load model da duoc huan luyen
        json_file = open("save_models/" + newest_model + ".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("save_models/" + newest_model + ".h5")
    else:
        with open(f"save_models/{newest_model}.pkl", "rb") as model_file:
            model = pickle.load(model_file)

    gen_video(args.model, model, class_names)
