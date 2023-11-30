import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from def_lib import (
    load_csv,
    preprocess_data,
    plot_confusion_matrix,
    find_newest_model_with_prefix,
)
from keras.models import model_from_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="fnn",
        help="model name: svm / fnn / conv1d",
    )
    parser.add_argument(
        "--folder_model", type=str, default="save_models", help="the folder save models"
    )
    args = parser.parse_args()

    # Lấy danh sách tất cả các tham số đã đăng ký
    registered_args = parser._actions
    # Hiển thị tên và mô tả của các tham số
    for arg in registered_args:
        if arg.help is not argparse.SUPPRESS:
            default = arg.default if arg.default is not argparse.SUPPRESS else None
            print(f"\t --{arg.dest}\t {default}\t {arg.help}")

    print("Setup Configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(f"\t{name} {value}")
    print("=" * 100)

    newest_model = find_newest_model_with_prefix(args.folder_model, args.model)
    name_saved = newest_model

    # load model da duoc huan luyen
    json_file = open(args.folder_model + "/" + newest_model + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(args.folder_model + "/" + newest_model + ".h5")

    path_data = "data"
    test_path = f"{path_data}/test_data.csv"
    X_test, y_test, class_names = load_csv(test_path)

    processed_X_test = preprocess_data(X_test)

    y_pred = model.predict(processed_X_test)

    ytrue = np.argmax(y_test, axis=1)
    ypred = np.argmax(y_pred, axis=1)

    plot_confusion_matrix_path = f"figures/confusion_matrix_{name_saved}.png"
    plot_confusion_matrix_nor_path = f"figures/confusion_matrix_nor_{name_saved}.png"
    # Plot the confusion matrix
    cm = confusion_matrix(ytrue, ypred)
    plot_confusion_matrix(
        plot_confusion_matrix_path,
        cm,
        class_names,
        title="Confusion Matrix of Yoga Pose Model",
    )

    plot_confusion_matrix(
        plot_confusion_matrix_nor_path,
        cm,
        class_names,
        normalize=True,
        title="Normalized Confusion Matrix of Yoga Pose Model",
    )

    # Print the classification report
    print(
        "\nClassification Report:\n",
        classification_report(ytrue, ypred, target_names=class_names, zero_division=0),
    )

    with open(f"statistics/classification_report_{name_saved}.txt", "w") as f:
        f.writelines(
            classification_report(
                ytrue, ypred, target_names=class_names, zero_division=0
            )
        )
    f.close()
