# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[List[ImageObject], Tuple[List[ImageObject], List[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example: Dict[str, Any] = dict(self.dataset[index])
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)
    
        # ---------- prompt 编码（和你现有一致） ----------
        if self.image_key in example and self.processor is not None:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                images = [os.path.join(self.image_dir, img) for img in images]
            processed_images: Optional[List[ImageObject]] = [] if len(images) != 0 else None
            for img in images:
                processed_images.append(process_image(img, self.min_pixels, self.max_pixels))
            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example and self.processor is not None:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):
                videos = [os.path.join(self.image_dir, vid) for vid in videos]
            processed_videos: Optional[List[List[ImageObject]]] = [] if len(videos) != 0 else None
            video_fps_list: List[float] = []
            for vid in videos:
                processed_video, video_fps = process_video(vid, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True)
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)
            model_inputs = self.processor(
                videos=processed_videos,
                text=[prompt],
                add_special_tokens=False,
                return_tensors="pt",
            )
            if "second_per_grid_ts" in getattr(self.processor, "model_input_names", []):
                model_inputs["second_per_grid_ts"] = [2.0 / fps for fps in video_fps_list]
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
    
        try:
            from verl.models.transformers.qwen2_vl import get_rope_index  # type: ignore
            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw", None),
                    video_grid_thw=model_inputs.get("video_grid_thw", None),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                    attention_mask=attention_mask,
                )
            else:
                raise ImportError
        except Exception:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)
    
        # ---------- 左填充/右截断（与你现有一致） ----------
        def left_pad_to_length(x: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
            if x.size(0) >= length:
                return x[-length:]
            pad = x.new_full((length - x.size(0),), pad_value)
            return torch.cat([pad, x], dim=0)
    
        max_len = self.max_prompt_length
        if len(input_ids) > max_len:
            if self.truncation == "left":
                input_ids = input_ids[-max_len:]
                attention_mask = attention_mask[-max_len:]
                position_ids = position_ids[-max_len:]
            elif self.truncation == "right":
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                position_ids = position_ids[:max_len]
            else:
                raise RuntimeError(f"Prompt length {len(input_ids)} is longer than {max_len}.")
        else:
            input_ids = left_pad_to_length(input_ids, max_len, self.tokenizer.pad_token_id)
            attention_mask = left_pad_to_length(attention_mask, max_len, 0)
            position_ids = left_pad_to_length(position_ids, max_len, 0)
    
        raw_prompt_ids: List[int] = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > max_len:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-max_len:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:max_len]
            else:
                raw_prompt_ids = raw_prompt_ids[-max_len:]
    
        # ---------- 新增 SFT：答案处理 + SFT 三件套 ----------
        # [新增] 取答案并归一化为纯文本
        answer_msgs = example.pop(self.answer_key)
        if isinstance(answer_msgs, list) and answer_msgs and isinstance(answer_msgs[0], dict):
            answer_text = "".join(m.get("content", "") for m in answer_msgs)
        else:
            answer_text = str(answer_msgs)
    
        # [新增] raw_ground_truth_ids：答案纯内容分词
        example["raw_ground_truth_ids"] = self.tokenizer.encode(answer_text, add_special_tokens=False)
    
        # [新增] 构造完整对话并展开为 SFT 序列（这里用 tokenizer；多模态可用 processor）
        assistant_msgs = [{"role": "assistant", "content": answer_text}]
        sft_prompt = self.tokenizer.apply_chat_template(
            messages + assistant_msgs,
            add_generation_prompt=False,
            tokenize=False,
        )
        sft_inputs = self.tokenizer([sft_prompt], add_special_tokens=False, return_tensors="pt")
        sft_input_ids = sft_inputs["input_ids"][0]
        sft_attention_mask = sft_inputs["attention_mask"][0]
    
        # [新增] 构造 labels：prompt 段置 -100，答案段保留
        labels = sft_input_ids.clone()
        prompt_len_plain = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels[:prompt_len_plain] = -100
    
        # [新增] 对 SFT 三件套做左填充/右截断（复用 max_len，也可单独定义 max_sft_len）
        if sft_input_ids.size(0) > max_len:
            sft_input_ids = sft_input_ids[-max_len:]
            sft_attention_mask = sft_attention_mask[-max_len:]
            labels = labels[-max_len:]
            # 再次确保窗口内 prompt 段被忽略
            keep_prompt = max(0, prompt_len_plain - (sft_inputs["input_ids"].size(1) - max_len))
            labels[:keep_prompt] = -100
        else:
            sft_input_ids = left_pad_to_length(sft_input_ids, max_len, self.tokenizer.pad_token_id)
            sft_attention_mask = left_pad_to_length(sft_attention_mask, max_len, 0)
            labels = left_pad_to_length(labels, max_len, -100)
    
        # ---------- 输出 ----------
        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
    
        # [新增] SFT 三件套
        example["sft_input_ids"] = sft_input_ids
        example["sft_attention_mask"] = sft_attention_mask
        example["labels"] = labels
    
        # ground_truth 文本仅用于可视化
        example["ground_truth"] = answer_text
        return example
