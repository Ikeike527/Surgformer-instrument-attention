# -----------------------------
# Cut black margin for surgical video
# Copyright (c) CUHK 2021.
# IEEE TMI 'Temporal Relation Network for Workflow Recognition from Surgical Video'
# -----------------------------

import argparse
import multiprocessing
import os

import cv2
from tqdm import tqdm


def create_directory_if_not_exists(path):
    os.makedirs(path, exist_ok=True)


def filter_black(image):
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10, y - 10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x:
        return image

    left = min(edges_x)
    right = max(edges_x)
    width = right - left
    bottom = min(edges_y)
    top = max(edges_y)
    height = top - bottom
    return image[left : left + width, bottom : bottom + height]


def process_image(image_source, image_save):
    frame = cv2.imread(image_source)
    dim = (int(frame.shape[1] / frame.shape[0] * 300), 300)
    frame = cv2.resize(frame, dim)
    frame = filter_black(frame)
    img_result = cv2.resize(frame, (250, 250))
    cv2.imwrite(image_save, img_result)


def process_video(video_source, video_save):
    create_directory_if_not_exists(video_save)
    for image_id in sorted(os.listdir(video_source)):
        if image_id == ".DS_Store":
            continue
        process_image(
            os.path.join(video_source, image_id),
            os.path.join(video_save, image_id),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Cut black margins from Cholec80 frames.")
    parser.add_argument("--source-dir", default="data/Cholec80/frames")
    parser.add_argument("--save-dir", default="data/Cholec80/frames_cutmargin")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_directory_if_not_exists(args.save_dir)

    processes = []
    for video_id in tqdm(sorted(os.listdir(args.source_dir))):
        if video_id == ".DS_Store":
            continue
        process = multiprocessing.Process(
            target=process_video,
            args=(
                os.path.join(args.source_dir, video_id),
                os.path.join(args.save_dir, video_id),
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print("Cut Done")
