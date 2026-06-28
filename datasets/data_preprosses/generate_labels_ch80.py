import argparse
import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Cholec80 pickle labels.")
    parser.add_argument("--root-dir", default="data/Cholec80")
    parser.add_argument("--fps-tag", default="1fps")
    parser.add_argument("--output-fps", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = args.root_dir
    video_names = sorted(
        x for x in os.listdir(os.path.join(root_dir, "videos")) if x.endswith(".mp4")
    )
    train_numbers = np.arange(1, 41).tolist()
    val_numbers = np.arange(41, 49).tolist()
    test_numbers = np.arange(49, 81).tolist()

    train_frame_numbers = 0
    val_frame_numbers = 0
    test_frame_numbers = 0
    train_pkl = {}
    val_pkl = {}
    test_pkl = {}
    unique_id_train = 0
    unique_id_val = 0
    unique_id_test = 0

    phase2id = {
        "Preparation": 0,
        "CalotTriangleDissection": 1,
        "ClippingCutting": 2,
        "GallbladderDissection": 3,
        "GallbladderPackaging": 4,
        "CleaningCoagulation": 5,
        "GallbladderRetraction": 6,
    }

    for video_name in video_names:
        video_id = video_name.replace(".mp4", "")
        vid_id = int(video_id.replace("video", ""))
        if vid_id in train_numbers:
            unique_id = unique_id_train
        elif vid_id in val_numbers:
            unique_id = unique_id_val
        elif vid_id in test_numbers:
            unique_id = unique_id_test
        else:
            continue

        vidcap = cv2.VideoCapture(os.path.join(root_dir, "videos", video_name))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise RuntimeError(f"Could not read FPS from {video_name}")
        sample_every = max(int(round(fps / args.output_fps)), 1)
        frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vidcap.release()

        tool_path = os.path.join(root_dir, "tool_annotations", video_name.replace(".mp4", "-tool.txt"))
        tool_dict = {}
        with open(tool_path, "r", encoding="utf-8") as tool_file:
            _ = tool_file.readline()
            for line in tool_file:
                parts = line.strip().split()
                if parts:
                    values = list(map(int, parts))
                    tool_dict[str(values[0])] = values[1:]

        phase_path = os.path.join(root_dir, "phase_annotations", video_name.replace(".mp4", "-phase.txt"))
        with open(phase_path, "r", encoding="utf-8") as phase_file:
            phase_results = phase_file.readlines()[1:]

        frame_infos = []
        sampled_frame_id = 0
        sampled_frames = int(np.ceil(frames / sample_every))
        for frame_id in tqdm(range(frames)):
            if frame_id % sample_every != 0:
                continue

            phase = phase_results[frame_id].strip().split()
            assert int(phase[0]) == frame_id

            frame_infos.append(
                {
                    "unique_id": unique_id,
                    "frame_id": sampled_frame_id,
                    "video_id": video_id,
                    "tool_gt": tool_dict.get(str(frame_id)),
                    "phase_gt": phase2id[phase[1]],
                    "phase_name": phase[1],
                    "fps": args.output_fps,
                    "original_frames": frames,
                    "frames": sampled_frames,
                }
            )
            unique_id += 1
            sampled_frame_id += 1

        if vid_id in train_numbers:
            train_pkl[video_id] = frame_infos
            train_frame_numbers += frames
            unique_id_train = unique_id
        elif vid_id in val_numbers:
            val_pkl[video_id] = frame_infos
            val_frame_numbers += frames
            unique_id_val = unique_id
        else:
            test_pkl[video_id] = frame_infos
            test_frame_numbers += frames
            unique_id_test = unique_id

    train_save_dir = os.path.join(root_dir, "labels", "train")
    val_save_dir = os.path.join(root_dir, "labels", "val")
    test_save_dir = os.path.join(root_dir, "labels", "test")
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
