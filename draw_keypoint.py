"""
Default command line parameter:
    python draw_keypoint.py --image_path images/SidePlankPose.jpg --save_image no
"""
import os
from datetime import datetime

import argparse
import tensorflow as tf
import cv2
from def_lib import detect, draw_prediction_on_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default="images/SidePlanKPose.jpg",
        help="the image path needs to draw the skeleton",
    )
    parser.add_argument(
        "--save_image",
        type=str,
        default="no",
        help="image save options: yes/no",
    )
    args = parser.parse_args()
    # Lấy danh sách tất cả các tham số đã đăng ký
    registered_args = parser._actions
    # Hiển thị tên và mô tả của các tham số
    print("Description:")
    for arg in registered_args:
        if arg.help is not argparse.SUPPRESS:
            default = arg.default if arg.default is not argparse.SUPPRESS else None
            print(f"\t --{arg.dest}\t {default}\t {arg.help}")
    print("Setup Configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(f"\t{name} {value}")
    print("=" * 100)

    # Tạo các thư mục cần thiết
    list_dir = ["./skelecton_images"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    ####################
    # DRAW SKELECTON ON IMG #
    ####################

    image = tf.io.read_file(args.image_path)
    image = tf.image.decode_image(image)
    print("shape",image.shape )
    person = detect(image)
    image = draw_prediction_on_image(image.numpy(), person, crop_region=None, 
                                close_figure=False, keep_input_size=True)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Lấy ngày tháng năm hiện tại
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Tạo đường dẫn tới tập tin lưu
    file_name = os.path.splitext(os.path.basename(args.image_path))[0]
    img_saved_path = os.path.join(
        "skelecton_images", f"{file_name}_{current_time}.png"
    )
    # Lưu tùy chọn
    if args.save_image == "yes":
        cv2.imwrite(img_saved_path, image)
        print(f"Ảnh đã được lưu tại: {img_saved_path}")
    else:
        print("Không lưu ảnh.")

    cv2.imshow('IMAGE SKELECTON', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()