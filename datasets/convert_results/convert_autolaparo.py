import argparse
import glob
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert distributed AutoLaparo txt outputs.")
    parser.add_argument("--main-path", required=True)
    return parser.parse_args()


def normalize_video_id(video_id):
    return video_id[5:] if video_id.startswith("video") else video_id


def main():
    args = parse_args()
    txt_files = sorted(glob.glob(os.path.join(args.main_path, "[0-9]*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No rank txt files found in {args.main_path}")

    anns_path = os.path.join(args.main_path, "phase_annotations")
    pred_path = os.path.join(args.main_path, "prediction")
    os.makedirs(anns_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)

    video_records = {}
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as handle:
            for line in handle.readlines()[1:]:
                parts = line.split()
                if len(parts) < 4:
                    continue
                video_id = parts[1]
                frame_id = int(parts[2])
                target = int(parts[-1])
                prediction = int(
                    np.fromstring(
                        line.split("[")[1].split("]")[0],
                        dtype=np.float32,
                        sep=",",
                    ).argmax()
                )
                video_records.setdefault(video_id, {})[frame_id] = (target, prediction)

    for video_id in sorted(video_records, key=lambda item: int(normalize_video_id(item))):
        save_id = normalize_video_id(video_id)
        ann_file = os.path.join(anns_path, f"video-{save_id}.txt")
        pred_file = os.path.join(pred_path, f"video-{save_id}.txt")
        frames = sorted(video_records[video_id].items())

        with open(ann_file, "w", encoding="utf-8") as handle:
            handle.write("Frame\tPhase\n")
            for frame_id, (target, _) in frames:
                handle.write(f"{frame_id}\t{target}\n")

        with open(pred_file, "w", encoding="utf-8") as handle:
            handle.write("Frame\tPhase\n")
            for frame_id, (_, prediction) in frames:
                handle.write(f"{frame_id}\t{prediction}\n")


if __name__ == "__main__":
    main()
