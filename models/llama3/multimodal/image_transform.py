import math
from collections import defaultdict
from logging import getLogger
from typing import Any, Optional, Set, Tuple
import torch
import torchvision.transforms as tv
from PIL import Image
from torchvision.transforms import functional as F

IMAGE_RES = 224

logger = getLogger()


class VariableSizeImageTransform(object):
       def __init__(self, size: int = IMAGE_RES) -> None:
        self.size = size
        logger.info(f"VariableSizeImageTransform size: {self.size}")
        self.to_tensor = tv.ToTensor()
        self._mean = (0.48145466, 0.4578275, 0.40821073)
        self._std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = tv.Normalize(
            mean=self._mean,
            std=self._std,
            inplace=True,
        )
        self.resample = tv.InterpolationMode.BILINEAR

    @staticmethod
    def get_factors(n: int) -> Set[int]:
        factors_set = set()

        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                factors_set.add(i)
                factors_set.add(n // i)
        return factors_set

    def find_supported_resolutions(self, max_num_chunks: int, patch_size: int) -> torch.Tensor:
        asp_dict = defaultdict(list)
        for chunk_size in range(max_num_chunks, 0, -1):
            _factors = sorted(self.get_factors(chunk_size))
            _asp_ratios = [(factor, chunk_size // factor) for factor in _factors]
            for height, width in _asp_ratios:
                ratio_float = height / width
                asp_dict[ratio_float].append((height, width))

        possible_resolutions = []
        for key, value in asp_dict.items():
            for height, depth in value:
                possible_resolutions.append((height * patch_size, depth * patch_size))

        return possible_resolutions

    @staticmethod
    def get_max_res_without_distortion(
        image_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[int, int]:

        original_width, original_height = image_size
        target_width, target_height = target_size

        scale_w = target_width / original_width
        scale_h = target_height / original_height

        if scale_w < scale_h:
            new_width = target_width
            new_height = min(math.floor(original_height * scale_w), target_height)
        else:
            new_height = target_height
            new_width = min(math.floor(original_width * scale_h), target_width)

        return new_width, new_height

    def _pad(self, image: Image.Image, target_size) -> Image.Image:
        new_width, new_height = target_size
        new_im = Image.new(mode="RGB", size=(new_width, new_height), color=(0, 0, 0))  # type: ignore
        new_im.paste(image)
        return new_im

    def _split(self, image: torch.Tensor, ncw: int, nch: int) -> torch.Tensor:
        # Split image into number of required tiles (width x height)
        num_channels, height, width = image.size()
        image = image.view(num_channels, nch, height // nch, ncw, width // ncw)
        # Permute dimensions to reorder the axes
        image = image.permute(1, 3, 0, 2, 4).contiguous()
        # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
        image = image.view(ncw * nch, num_channels, height // nch, width // ncw)
        return image

    def resize_without_distortion(
        self,
        image: torch.Tensor,
        target_size: Tuple[int, int],
        max_upscaling_size: Optional[int],
    ) -> torch.Tensor:
 
        image_width, image_height = image.size
        image_size = (image_width, image_height)

        if max_upscaling_size is not None:
            new_target_width = min(max(image_width, max_upscaling_size), target_size[0])
            new_target_height = min(max(image_height, max_upscaling_size), target_size[1])
            target_size = (new_target_width, new_target_height)

        new_size_without_distortion = self.get_max_res_without_distortion(image_size, target_size)

        image = F.resize(
            image,
            (new_size_without_distortion[1], new_size_without_distortion[0]),
            interpolation=self.resample,
        )

        return image

    def get_best_fit(
        self,
        image_size: Tuple[int, int],
        possible_resolutions: torch.Tensor,
        resize_to_max_canvas: bool = False,
    ) -> Tuple[int, int]:
       

        original_width, original_height = image_size

        target_widths, target_heights = (
            possible_resolutions[:, 0],
            possible_resolutions[:, 1],
        )

        scale_w = target_widths / original_width
        scale_h = target_heights / original_height

        scales = torch.where(scale_w > scale_h, scale_h, scale_w)

        upscaling_options = scales[scales >= 1]
        if len(upscaling_options) > 0:
            if resize_to_max_canvas:
                selected_scale = torch.max(upscaling_options)
            else:
                selected_scale = torch.min(upscaling_options)
        else:
            downscaling_options = scales[scales < 1]
            selected_scale = torch.max(downscaling_options)

        chosen_canvas = possible_resolutions[scales == selected_scale]

            areas = chosen_canvas[:, 0] * chosen_canvas[:, 1]
            optimal_idx = torch.argmin(areas)
            optimal_canvas = chosen_canvas[optimal_idx]
        else:
            optimal_canvas = chosen_canvas[0]

        return tuple(optimal_canvas.tolist())

    def __call__(
        self,
        image: Image.Image,
        max_num_chunks: int,
        normalize_img: bool = True,
        resize_to_max_canvas: bool = False,
    ) -> Tuple[Any, Any]:
       
        assert max_num_chunks > 0
        assert isinstance(image, Image.Image), type(image)
        w, h = image.size

        possible_resolutions = self.find_supported_resolutions(max_num_chunks=max_num_chunks, patch_size=self.size)
        possible_resolutions = torch.tensor(possible_resolutions)

        best_resolution = self.get_best_fit(
            image_size=(w, h),
            possible_resolutions=possible_resolutions,
            resize_to_max_canvas=resize_to_max_canvas,
        )

        max_upscaling_size = None if resize_to_max_canvas else self.size
        image = self.resize_without_distortion(image, best_resolution, max_upscaling_size)
        image = self._pad(image, best_resolution)

        image = self.to_tensor(image)

        if normalize_img:
            image = self.normalize(image)

        ratio_w, ratio_h = (
            best_resolution[0] // self.size,
            best_resolution[1] // self.size,
        )

        image = self._split(image, ratio_w, ratio_h) 

        ar = (ratio_h, ratio_w)
        return image, ar
