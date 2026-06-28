import argparse
import glob
import os

import numpy as np


DATASET_CONFIG = {
    "Cholec80": {"name_format": "video{id}"},
    "AutoLaparo": {"name_format": "{id}"},
    "CATARACTS": {"name_format": "video{id}"},
    "PmLR50": {"name_format": "video{id}"},
    "M2CAI16-Workflow": {"name_format": "video_{id}"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert distributed txt outputs to per-video files.")
    parser.add_argument("--main-path", required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), required=True)
    parser.add_argument("--auto-mapping-mode", action="store_true")
    return parser.parse_args()


def normalize_video_name(video_name):
    if video_name.startswith("video_"):
        return video_name[6:]
    if video_name.startswith("video"):
        return video_name[5:]
    return video_name


def load_txt_lines(main_path):
    txt_files = sorted(glob.glob(os.path.join(main_path, "[0-9]*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No rank txt files found in {main_path}")
    return [open(path, "r", encoding="utf-8").readlines() for path in txt_files]


def main():
    args = parse_args()
    all_lines = load_txt_lines(args.main_path)
    anns_path = os.path.join(args.main_path, "phase_annotations")
    pred_path = os.path.join(args.main_path, "prediction")
    os.makedirs(anns_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)

    if args.auto_mapping_mode:
        video_ids = set()
        for lines in all_lines:
            for line in lines[1:]:
                parts = line.split()
                if len(parts) > 1:
                    video_ids.add(normalize_video_name(parts[1]))
        ordered_video_ids = sorted(video_ids, key=lambda item: int(item))
    else:
        ordered_video_ids = sorted(
            {normalize_video_name(line.split()[1]) for lines in all_lines for line in lines[1:] if len(line.split()) > 1},
            key=lambda item: int(item),
        )

    video_name_format = DATASET_CONFIG[args.dataset]["name_format"]

    for video_id in ordered_video_ids:
        video_data_dict = {}
        for lines in all_lines:
            for line in lines[1:]:
                parts = line.split()
                if len(parts) < 4:
                    continue
                file_video_name = parts[1]
                expected_video_name = video_name_format.format(id=video_id)
                if file_video_name == expected_video_name or normalize_video_name(file_video_name) == video_id:
                    frame_num = int(parts[2])
                    phase = int(parts[-1]) if args.dataset != "M2CAI16-Workflow" else int(parts[11])
                    prediction = int(
                        np.fromstring(
                            line.split("[")[1].split("]")[0],
                            dtype=np.float32,
                            sep=",",
                        ).argmax()
                    )
                    video_data_dict[frame_num] = (phase, prediction)

        output_filename = f"video-{video_id}.txt"
        items = sorted(video_data_dict.items())
        with open(os.path.join(anns_path, output_filename), "w", encoding="utf-8") as handle:
            handle.write("Frame\tPhase\n")
            for frame_num, (phase, _) in items:
                handle.write(f"{frame_num}\t{phase}\n")
        with open(os.path.join(pred_path, output_filename), "w", encoding="utf-8") as handle:
            handle.write("Frame\tPhase\n")
            for frame_num, (_, prediction) in items:
                handle.write(f"{frame_num}\t{prediction}\n")

    print("Done")


if __name__ == "__main__":
    main()
