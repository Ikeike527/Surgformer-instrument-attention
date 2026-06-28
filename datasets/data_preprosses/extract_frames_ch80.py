import argparse
import os

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from Cholec80 videos.")
    parser.add_argument("--root-dir", default="data/Cholec80")
    parser.add_argument("--video-subdir", default="videos")
    parser.add_argument("--frame-subdir", default="frames")
    parser.add_argument("--output-fps", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    video_root = os.path.join(args.root_dir, args.video_subdir)
    frame_root = os.path.join(args.root_dir, args.frame_subdir)
    video_names = sorted(x for x in os.listdir(video_root) if x.endswith(".mp4"))
    total_frames = 0

    for video_name in video_names:
        print(video_name)
        video_path = os.path.join(video_root, video_name)
        vidcap = cv2.VideoCapture(video_path)
        video_fps = vidcap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            raise RuntimeError(f"Could not read FPS from {video_path}")

        sample_every = max(int(round(video_fps / args.output_fps)), 1)
        frame_index = 0
        saved_index = 0
        save_dir = os.path.join(frame_root, video_name.replace(".mp4", ""))
        os.makedirs(save_dir, exist_ok=True)

        while True:
            success, image = vidcap.read()
            if not success:
                break
            if frame_index % sample_every == 0:
                save_path = os.path.join(save_dir, f"{saved_index:05d}.png")
                cv2.imwrite(save_path, image)
                saved_index += 1
            frame_index += 1

        vidcap.release()
        cv2.destroyAllWindows()
        print(f"source frames={frame_index}, saved frames={saved_index}, fps={video_fps}")
        total_frames += frame_index

    print("Total Frames", total_frames)


if __name__ == "__main__":
    main()
