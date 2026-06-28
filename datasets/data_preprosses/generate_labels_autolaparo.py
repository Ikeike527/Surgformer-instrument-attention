import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AutoLaparo pickle labels.")
    parser.add_argument("--root-dir", default="data/AutoLaparo")
    parser.add_argument("--fps-tag", default="1fps")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = args.root_dir
    video_names = sorted(
        x for x in os.listdir(os.path.join(root_dir, "frames")) if "DS" not in x
    )

    train_numbers = np.arange(1, 11).tolist()
    val_numbers = np.arange(11, 15).tolist()
    test_numbers = np.arange(15, 22).tolist()

    train_frame_numbers = 0
    val_frame_numbers = 0
    test_frame_numbers = 0
    train_pkl = {}
    val_pkl = {}
    test_pkl = {}
    unique_id_train = 0
    unique_id_val = 0
    unique_id_test = 0

    id2phase = {
        0: "Preparation",
        1: "Dividing Ligament and Peritoneum",
        2: "Dividing Uterine Vessels and Ligament",
        3: "Transecting the Vagina",
        4: "Specimen Removal",
        5: "Suturing",
        6: "Washing",
    }

    for video_id in video_names:
        vid_id = int(video_id)
        if vid_id in train_numbers:
            unique_id = unique_id_train
        elif vid_id in val_numbers:
            unique_id = unique_id_val
        else:
            unique_id = unique_id_test

        video_path = os.path.join(root_dir, "frames", video_id)
        frames_list = sorted(os.listdir(video_path))

        phase_path = os.path.join(root_dir, "labels", f"label_{video_id}.txt")
        with open(phase_path, "r", encoding="utf-8") as phase_file:
            phase_results = phase_file.readlines()[1:]

        frame_infos = []
        for frame_id in tqdm(range(len(frames_list))):
            phase = phase_results[frame_id].strip().split()
            assert int(phase[0]) == frame_id + 1
            phase_id = int(phase[1])
            frame_infos.append(
                {
                    "unique_id": unique_id,
                    "frame_id": frame_id,
                    "original_frame_id": frame_id,
                    "video_id": video_id,
                    "tool_gt": None,
                    "frames": len(frames_list),
                    "phase_gt": phase_id,
                    "phase_name": id2phase[phase_id],
                    "fps": 1,
                }
            )
            unique_id += 1

        if vid_id in train_numbers:
            train_pkl[video_id] = frame_infos
            train_frame_numbers += len(frames_list)
            unique_id_train = unique_id
        elif vid_id in val_numbers:
            val_pkl[video_id] = frame_infos
            val_frame_numbers += len(frames_list)
            unique_id_val = unique_id
        else:
            test_pkl[video_id] = frame_infos
            test_frame_numbers += len(frames_list)
            unique_id_test = unique_id

    train_save_dir = os.path.join(root_dir, "labels_pkl", "train")
    val_save_dir = os.path.join(root_dir, "labels_pkl", "val")
    test_save_dir = os.path.join(root_dir, "labels_pkl", "test")
    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(val_save_dir, exist_ok=True)
    os.makedirs(test_save_dir, exist_ok=True)

    with open(os.path.join(train_save_dir, f"{args.fps_tag}train.pickle"), "wb") as handle:
        pickle.dump(train_pkl, handle)
    with open(os.path.join(val_save_dir, f"{args.fps_tag}val.pickle"), "wb") as handle:
        pickle.dump(val_pkl, handle)
    with open(os.path.join(test_save_dir, f"{args.fps_tag}test.pickle"), "wb") as handle:
        pickle.dump(test_pkl, handle)

    print("TRAIN Frames", train_frame_numbers, unique_id_train)
    print("VAL Frames", val_frame_numbers, unique_id_val)
    print("TEST Frames", test_frame_numbers, unique_id_test)


if __name__ == "__main__":
    main()
