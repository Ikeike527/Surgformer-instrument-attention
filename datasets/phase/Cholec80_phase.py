import os
import cv2
import numpy as np
import torch
import decord
import pickle
from pathlib import Path
from PIL import Image
from torchvision import transforms
from datasets.transforms.random_erasing import RandomErasing
import warnings
from torch.utils.data import Dataset
import random
import datasets.transforms.video_transforms as video_transforms
import datasets.transforms.volume_transforms as volume_transforms


PHASE2ID_CHOLEC80 = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}

CHOLEC80_TRAIN_VIDEO_IDS = tuple(range(1, 41))
CHOLEC80_VAL_VIDEO_IDS = tuple(range(41, 49))
CHOLEC80_TEST_VIDEO_IDS = tuple(range(49, 81))
CHOLEC80_SPLIT_TAG = "split_tr01-40_val41-48_test49-80"


def get_cholec80_video_ids(mode):
    if mode == "train":
        return set(CHOLEC80_TRAIN_VIDEO_IDS)
    if mode == "val":
        return set(CHOLEC80_VAL_VIDEO_IDS)
    if mode == "test":
        return set(CHOLEC80_TEST_VIDEO_IDS)
    raise ValueError(f"Unsupported Cholec80 split mode: {mode}")

def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


class PhaseDataset_Cholec80(Dataset):
    """Load video phase recognition dataset."""

    def __init__(
        self,
        anno_path="data/cholec80/labels/train/train.pickle",
        data_path="data/cholec80",
        mode="train",  # val/test
        data_strategy="online",  # offline
        output_mode="key_frame",  # all_frame
        cut_black=True,
        clip_len=16,
        frame_sample_rate=2,  # 0表示指数级间隔，-1表示随机间隔设置, -2表示递增间隔
        crop_size=224,
        short_side_size=256,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        args=None,
    ):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.data_strategy = data_strategy
        self.output_mode = output_mode
        self.cut_black = cut_black
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.args = args

        # 器具マスク (SAM3) による Attention 強調用。有効時のみマスクを読み込み、
        # test モードの返り値に (T, S, S) の float マスクを 5 要素目として付加する。
        self.instr_attn_bias = bool(getattr(args, "instr_attn_bias", False))
        self.instr_mask_dirname = getattr(args, "instr_mask_dirname", "instrument_masks")

        self.frame_span = self.clip_len * self.frame_sample_rate

        # Augment
        self.aug = False
        self.rand_erase = False
        if self.mode in ["train"]:
            self.aug = True
            if self.args.reprob > 0:  # default: 0.25
                self.rand_erase = True
        self.infos = self._load_infos()
        self.dataset_samples = self._make_dataset(self.infos)

        if mode == "train":
            pass

        elif mode == "val":
            self.data_transform = video_transforms.Compose(
                [
                    video_transforms.Resize(
                        (self.short_side_size, self.short_side_size),
                        interpolation="bilinear",
                    ),
                    # video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif mode == "test":
            self.data_resize = video_transforms.Compose(
                [
                    video_transforms.Resize(
                        size=(short_side_size, short_side_size),
                        interpolation="bilinear",
                    ),
                    # video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                ]
            )
            self.data_transform = video_transforms.Compose(
                [
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __getitem__(self, index):
        if self.mode == "train":
            args = self.args
            frames_info = self.dataset_samples[index]
            video_id, frame_id, frames = (
                frames_info["video_id"],
                frames_info["frame_id"],
                frames_info["frames"],
            )
            if self.data_strategy == "online":
                buffer, phase_labels, sampled_list = self._video_batch_loader(
                    frames, frame_id, video_id, index, self.cut_black
                )  # T H W C
            elif self.data_strategy == "offline":
                (
                    buffer,
                    phase_labels,
                    sampled_list,
                ) = self._video_batch_loader_for_key_frames(
                    frames, frame_id, video_id, index, self.cut_black
                )  # T H W C

            buffer = self._aug_frame(buffer, args)

            if self.output_mode == "key_frame":
                if self.data_strategy == "offline":
                    return (
                        buffer,
                        phase_labels[self.clip_len // 2],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        {},
                    )
                elif self.data_strategy == "online":
                    return (
                        buffer,
                        phase_labels[-1],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        {},
                    )
            elif self.output_mode == "all_frame":
                return (
                    buffer,
                    phase_labels,
                    str(index) + "_" + video_id + "_" + str(frame_id),
                    {},
                )

        elif self.mode == "val":
            frames_info = self.dataset_samples[index]
            video_id, frame_id, frames = (
                frames_info["video_id"],
                frames_info["frame_id"],
                frames_info["frames"],
            )
            if self.data_strategy == "online":
                buffer, phase_labels, sampled_list = self._video_batch_loader(
                    frames, frame_id, video_id, index, self.cut_black
                )  # T H W C
            elif self.data_strategy == "offline":
                (
                    buffer,
                    phase_labels,
                    sampled_list,
                ) = self._video_batch_loader_for_key_frames(
                    frames, frame_id, video_id, index, self.cut_black
                )  # T H W C

            buffer = self.data_transform(buffer)

            if len(sampled_list) == len(np.unique(sampled_list)):
                flag = False
            else:
                flag = True

            if self.output_mode == "key_frame":
                if self.data_strategy == "offline":
                    return (
                        buffer,
                        phase_labels[self.clip_len // 2],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        flag,
                    )
                elif self.data_strategy == "online":
                    return (
                        buffer,
                        phase_labels[-1],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        flag,
                    )
            elif self.output_mode == "all_frame":
                return (
                    buffer,
                    phase_labels,
                    str(index) + "_" + video_id + "_" + str(frame_id),
                    flag,
                )

        elif self.mode == "test":
            frames_info = self.dataset_samples[index]
            video_id, frame_id, frames = (
                frames_info["video_id"],
                frames_info["frame_id"],
                frames_info["frames"],
            )
            if self.data_strategy == "online":
                buffer, phase_labels, sampled_list = self._video_batch_loader(
                    frames, frame_id, video_id, index, self.cut_black
                )  # T H W C
            elif self.data_strategy == "offline":
                (
                    buffer,
                    phase_labels,
                    sampled_list,
                ) = self._video_batch_loader_for_key_frames(
                    frames, frame_id, video_id, index, self.cut_black
                )  # T H W C

            # dim = (int(buffer[0].shape[1] / buffer[0].shape[0] * 300), 300)
            # buffer = [cv2.resize(frame, dim) for frame in buffer]
            # buffer = [self.filter_black(frame) for frame in buffer]
            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            buffer = self.data_transform(buffer)

            if len(sampled_list) == len(np.unique(sampled_list)):
                flag = False
            else:
                flag = True

            # 器具マスク (有効時のみ): フレームと同じ sampled_list 順で (T,S,S) を付加
            extra = (
                (self._load_instr_masks(sampled_list),) if self.instr_attn_bias else ()
            )

            if self.output_mode == "key_frame":
                if self.data_strategy == "offline":
                    return (
                        buffer,
                        phase_labels[self.clip_len // 2],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        flag,
                        *extra,
                    )
                elif self.data_strategy == "online":
                    return (
                        buffer,
                        phase_labels[-1],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        flag,
                        *extra,
                    )
            elif self.output_mode == "all_frame":
                return (
                    buffer,
                    phase_labels,
                    str(index) + "_" + video_id + "_" + str(frame_id),
                    flag,
                    *extra,
                )
        else:
            raise NameError("mode {} unkown".format(self.mode))

    def filter_black(self, image):
        binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
        binary_image2 = cv2.medianBlur(
            binary_image2, 19
        )  # filter the noise, need to adjust the parameter based on the dataset
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

        left = min(edges_x)  # left border
        right = max(edges_x)  # right
        width = right - left
        bottom = min(edges_y)  # bottom
        top = max(edges_y)  # top
        height = top - bottom

        pre1_picture = image[left : left + width, bottom : bottom + height]
        return pre1_picture

    def _aug_frame(
        self,
        buffer,
        args,
    ):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )
        # if self.cut_black:
        #     dim = (int(buffer[0].shape[1] / buffer[0].shape[0] * 300), 300)
        #     buffer = [cv2.resize(frame, dim) for frame in buffer]
        #     buffer = [self.filter_black(frame) for frame in buffer]
        #     buffer = [cv2.resize(frame, (250, 250)) for frame in buffer]

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)

        # for k in range(len(buffer)):
        #     img = cv2.cvtColor(np.asarray(buffer[k]), cv2.COLOR_RGB2BGR)
        #     cv2.imshow(str(k), img)
        #     cv2.waitKey()

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.7, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        # Vis
        # for k in range(buffer.shape[1]):
        #     img = cv2.cvtColor(np.asarray(buffer[:,k,:,:]).transpose(1,2,0), cv2.COLOR_RGB2BGR)
        #     cv2.imshow(str(k), img)
        #     cv2.waitKey()
        return buffer

    def _load_infos(self):
        if os.path.exists(self.anno_path):
            with open(self.anno_path, "rb") as handle:
                return pickle.load(handle)
        return self._build_infos_from_native_structure()

    def _build_infos_from_native_structure(self):
        frames_root = Path(self.data_path) / "frames"
        phase_root = Path(self.data_path) / "phase_annotations"
        tool_root = Path(self.data_path) / "tool_annotations"

        if not frames_root.exists():
            raise FileNotFoundError(
                f"Missing frames directory: {frames_root}. "
                f"Expected either {self.anno_path} or a native Cholec80 layout."
            )
        if not phase_root.exists():
            raise FileNotFoundError(
                f"Missing phase annotations directory: {phase_root}. "
                f"Expected either {self.anno_path} or a native Cholec80 layout."
            )

        valid_video_ids = get_cholec80_video_ids(self.mode)

        infos = {}
        unique_id = 0
        for video_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
            video_id = video_dir.name
            if not video_id.startswith("video"):
                continue

            numeric_video_id = int(video_id.replace("video", ""))
            if numeric_video_id not in valid_video_ids:
                continue

            frame_files = sorted(video_dir.glob("*.png"))
            if not frame_files:
                continue

            phase_path = phase_root / f"{video_id}-phase.txt"
            if not phase_path.exists():
                raise FileNotFoundError(f"Missing phase annotation file: {phase_path}")

            with open(phase_path, "r", encoding="utf-8") as phase_file:
                phase_results = phase_file.readlines()[1:]

            sample_stride = self._infer_sample_stride(
                num_annotation_frames=len(phase_results),
                num_sampled_frames=len(frame_files),
            )
            tool_dict = self._load_tool_annotations(tool_root / f"{video_id}-tool.txt")

            video_infos = []
            for frame_idx, frame_file in enumerate(frame_files):
                original_frame_id = min(frame_idx * sample_stride, len(phase_results) - 1)
                phase = phase_results[original_frame_id].strip().split()
                phase_frame_id = int(phase[0])
                phase_name = phase[1]
                video_infos.append(
                    {
                        "unique_id": unique_id,
                        "frame_id": frame_idx,
                        "original_frame_id": phase_frame_id,
                        "video_id": video_id,
                        "tool_gt": tool_dict.get(str(phase_frame_id)),
                        "phase_gt": PHASE2ID_CHOLEC80[phase_name],
                        "phase_name": phase_name,
                        "fps": 1,
                        "frames": len(frame_files),
                        "img_path": str(frame_file),
                    }
                )
                unique_id += 1

            infos[video_id] = video_infos

        if not infos:
            raise RuntimeError(
                f"No Cholec80 samples were found under {frames_root} for mode={self.mode}."
            )
        return infos

    def _infer_sample_stride(self, num_annotation_frames, num_sampled_frames):
        if num_sampled_frames <= 0:
            raise RuntimeError("No sampled frames found while inferring Cholec80 stride.")
        return max(int(round(num_annotation_frames / num_sampled_frames)), 1)

    def _load_tool_annotations(self, tool_path):
        if not tool_path.exists():
            return {}
        tool_dict = {}
        with open(tool_path, "r", encoding="utf-8") as tool_file:
            _ = tool_file.readline()
            for line in tool_file:
                parts = line.strip().split()
                if not parts:
                    continue
                values = list(map(int, parts))
                tool_dict[str(values[0])] = values[1:]
        return tool_dict

    def _resolve_img_path(self, line_info):
        candidates = []
        if "img_path" in line_info:
            candidates.append(line_info["img_path"])

        frames_dir = os.path.join(self.data_path, "frames", line_info["video_id"])
        frame_id = int(line_info["frame_id"])
        candidates.extend(
            [
                os.path.join(frames_dir, f"{frame_id:05d}.png"),
                os.path.join(frames_dir, f"{frame_id + 1:05d}.png"),
                os.path.join(frames_dir, f"{line_info['video_id']}_{frame_id + 1:06d}.png"),
                os.path.join(frames_dir, f"{line_info['video_id']}_{frame_id:06d}.png"),
            ]
        )

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def _resolve_sample_path(self, line_info, cut_black):
        base_path = line_info["img_path"]
        if cut_black:
            cut_black_path = base_path.replace("frames", "frames_cutmargin", 1)
            if os.path.exists(cut_black_path):
                return cut_black_path
        return base_path

    def _resolve_mask_path(self, frame_path):
        """フレーム画像パスから器具マスクパスを導出。
        実際に読むフレームと同じ video/ファイル名で instr_mask_dirname 下を探す。"""
        d = self.instr_mask_dirname
        if "frames_cutmargin" in frame_path:
            return frame_path.replace("frames_cutmargin", d, 1)
        return frame_path.replace("frames", d, 1)

    def _load_instr_masks(self, sampled_list):
        """sampled_list (フレームと同じ index 列・同じ順序) に対応する器具マスクを
        (T, S, S) float[0,1] で返す。S=short_side_size。マスク欠損は全 0。"""
        size = self.short_side_size
        masks = []
        for image_index in sampled_list:
            line_info = self.dataset_samples[image_index]
            frame_path = self._resolve_sample_path(line_info, self.cut_black)
            mask_path = self._resolve_mask_path(frame_path)
            if os.path.exists(mask_path):
                with Image.open(mask_path) as mask_file:
                    arr = np.asarray(mask_file.convert("L"), dtype=np.float32) / 255.0
                m = torch.from_numpy(arr)[None, None]  # (1,1,H,W)
                m = torch.nn.functional.interpolate(
                    m, size=(size, size), mode="nearest"
                )[0, 0]
            else:
                m = torch.zeros(size, size, dtype=torch.float32)
            masks.append(m)
        return torch.stack(masks, 0)  # (T, S, S)

    def _make_dataset(self, infos):
        frames = []
        for video_id in infos.keys():
            data = infos[video_id]
            for line_info in data:
                # line format: unique_id, frame_id, video_id, tool_gt, phase_gt, phase_name, fps, frames
                if len(line_info) < 8:
                    raise (
                        RuntimeError(
                            "Video input format is not correct, missing one or more element. %s"
                            % line_info
                        )
                    )
                img_path = self._resolve_img_path(line_info)
                # 当使用1fps采样时，line_info["frame_id"]类似于对应的序号，line_info["original_frame_id"]表示对应的图像序号
                line_info["img_path"] = img_path
                frames.append(line_info)
        return frames

    def _video_batch_loader(self, duration, indice, video_id, index, cut_black):
        offset_value = index - indice
        frame_sample_rate = self.frame_sample_rate
        sampled_list = []
        frame_id_list = []
        for i, _ in enumerate(range(0, self.clip_len)):
            frame_id = indice
            frame_id_list.append(frame_id)
            if self.frame_sample_rate == -1:
                frame_sample_rate = random.randint(1, 5)
            elif self.frame_sample_rate == 0:
                frame_sample_rate = 2**i
            elif self.frame_sample_rate == -2:
                frame_sample_rate = 1 if 2 * i == 0 else 2 * i
            if indice - frame_sample_rate >= 0:
                indice -= frame_sample_rate
        sampled_list = sorted([i + offset_value for i in frame_id_list])
        sampled_image_list = []
        sampled_label_list = []
        image_name_list = []
        for num, image_index in enumerate(sampled_list):
            try:
                image_name_list.append(self.dataset_samples[image_index]["img_path"])
                path = self._resolve_sample_path(self.dataset_samples[image_index], cut_black)
                # with で開いて画素を確定読み込み後に閉じる (ファイルハンドル/遅延ロードの蓄積を防ぐ)
                with Image.open(path) as image_file:
                    image_data = np.asarray(image_file)
                phase_label = self.dataset_samples[image_index]["phase_gt"]
                # PIL可视化
                # image_data.show()
                # cv2可视化
                # img = cv2.cvtColor(np.asarray(image_data), cv2.COLOR_RGB2BGR)
                # cv2.imshow(str(num), img)
                # cv2.waitKey()
                sampled_image_list.append(image_data)
                sampled_label_list.append(phase_label)
            except:
                raise RuntimeError(
                    "Error occured in reading frames {} from video {} of path {} (Unique_id: {}).".format(
                        frame_id_list[num],
                        video_id,
                        self.dataset_samples[image_index]["img_path"],
                        image_index,
                    )
                )
        video_data = np.stack(sampled_image_list)
        phase_data = np.stack(sampled_label_list)

        return video_data, phase_data, sampled_list

    def _video_batch_loader_for_key_frames(self, duration, timestamp, video_id, index, cut_black):
        # 永远控制的只有对应帧序号和整个视频序列有效视频数目，不受采样FPS影响，根据标签映射回对应image path
        # 当前视频内帧序号为timestamp,
        # 当前数据集内帧序号为index
        # 为了保证偶数输入的前序帧以及后续帧数目保持一致，中间double了关键帧
        # 如果为奇数，则中间帧位于中间，但是3D卷积不适用于偶数kernel及stride
        right_len = self.clip_len // 2
        left_len = self.clip_len - right_len
        offset_value = index - timestamp

        # load right
        right_sample_rate = self.frame_sample_rate
        cur_t = timestamp
        right_frames = []
        if right_len == left_len:
            for i, _ in enumerate(range(0, right_len)):
                right_frames.append(cur_t)
                if self.frame_sample_rate == -1:
                    right_sample_rate = random.randint(1, 5)
                elif self.frame_sample_rate == 0:
                    right_sample_rate = 2**i
                elif self.frame_sample_rate == -2:
                    right_sample_rate = 1 if 2 * i == 0 else 2 * i
                if cur_t + right_sample_rate <= duration:
                    cur_t += right_sample_rate
        else:
            for i, _ in enumerate(range(0, right_len)):
                if self.frame_sample_rate == -1:
                    right_sample_rate = random.randint(1, 5)
                elif self.frame_sample_rate == 0:
                    right_sample_rate = 2**i
                elif self.frame_sample_rate == -2:
                    right_sample_rate = 1 if 2 * i == 0 else 2 * i
                if cur_t + right_sample_rate <= duration:
                    cur_t += right_sample_rate
                right_frames.append(cur_t)

        # load left
        left_sample_rate = self.frame_sample_rate
        cur_t = timestamp
        left_frames = []
        for j, _ in enumerate(range(0, left_len)):
            left_frames = [cur_t] + left_frames
            if self.frame_sample_rate == -1:
                left_sample_rate = random.randint(1, 5)
            elif self.frame_sample_rate == 0:
                left_sample_rate = 2**j
            elif self.frame_sample_rate == -2:
                left_sample_rate = 1 if 2 * j == 0 else 2 * j
            if cur_t - left_sample_rate >= 0:
                cur_t -= left_sample_rate

        frame_id_list = left_frames + right_frames
        assert len(frame_id_list) == self.clip_len
        sampled_list = [i + offset_value for i in frame_id_list]
        sampled_image_list = []
        sampled_label_list = []
        image_name_list = []
        for num, image_index in enumerate(sampled_list):
            try:
                image_name_list.append(self.dataset_samples[image_index]["img_path"])
                path = self._resolve_sample_path(self.dataset_samples[image_index], cut_black)
                # with で開いて画素を確定読み込み後に閉じる (ファイルハンドル/遅延ロードの蓄積を防ぐ)
                with Image.open(path) as image_file:
                    image_data = np.asarray(image_file)
                phase_label = self.dataset_samples[image_index]["phase_gt"]
                # PIL可视化
                # image_data.show()
                # cv2可视化
                # img = cv2.cvtColor(np.asarray(image_data), cv2.COLOR_RGB2BGR)
                # cv2.imshow(str(num), img)
                # cv2.waitKey()
                sampled_image_list.append(image_data)
                sampled_label_list.append(phase_label)
            except:
                raise RuntimeError(
                    "Error occured in reading frames {} from video {} of path {} (Unique_id: {}).".format(
                        frame_id_list[num],
                        video_id,
                        self.dataset_samples[image_index]["img_path"],
                        image_index,
                    )
                )
        video_data = np.stack(sampled_image_list)
        phase_data = np.stack(sampled_label_list)
        return video_data, phase_data, sampled_list

    def __len__(self):
        return len(self.dataset_samples)


def build_dataset(is_train, test_mode, fps, args):
    if args.data_set == "Cholec80":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels", mode, fps + "test.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels", mode, fps + "val.pickle")

        dataset = PhaseDataset_Cholec80(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy="online",
            output_mode="key_frame",
            cut_black=False,
            clip_len=8,
            frame_sample_rate=8,  # 0表示指数级间隔，-1表示随机间隔设置, -2表示递增间隔
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 7
    assert nb_classes == args.nb_classes
    print("%s %s - %s : Number of the class = %d" % ("Cholec80", mode, fps, args.nb_classes))

    return dataset, nb_classes
