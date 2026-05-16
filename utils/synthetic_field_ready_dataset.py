"""Synthetic field-ready dataset generation for Project M.A.L.L.I.

This module builds a synthetic malaria microscopy dataset that mimics the
field conditions described in the project brief:

* thick-smear bunching via stacked cell overlays
* partial occlusions via a custom cutmix routine
* foldscope optics via chromatic aberration, blur, vignetting, dust, and bubbles
* stain variability via hue and saturation jitter
* curriculum scheduling for staged augmentation intensity during training

The output is a directory of PNG images plus a labels.csv file containing soft
labels derived from the mixing operations.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    import albumentations as A
    import cv2
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "This module requires albumentations, opencv-python, and torch. "
        "Install the pinned dependencies from requirements.txt before running the generator."
    ) from exc


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CellSample:
    """A single NIH cell image path and its binary label."""

    path: Path
    label: int


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an image to RGB uint8 layout for downstream processing."""

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return image


def load_rgb_image(image_path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    """Load a cell image with OpenCV, convert to RGB, and resize to target size."""

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != image_size:
        image = cv2.resize(image, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    return image


def save_rgb_image(output_path: Path, image: np.ndarray) -> None:
    """Save an RGB image to disk as PNG using OpenCV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), bgr):
        raise IOError(f"Failed to write image: {output_path}")


def _torch_blend(images: Sequence[np.ndarray], alphas: Sequence[float]) -> np.ndarray:
    """Blend images sequentially with PyTorch tensors for efficient alpha compositing."""

    if len(images) != len(alphas):
        raise ValueError("`images` and `alphas` must have the same length.")
    if not images:
        raise ValueError("At least one image is required for blending.")

    blended = torch.from_numpy(images[0].astype(np.float32) / 255.0)
    for image, alpha in zip(images[1:], alphas[1:]):
        overlay = torch.from_numpy(image.astype(np.float32) / 255.0)
        alpha_tensor = torch.tensor(float(alpha), dtype=torch.float32)
        blended = blended.mul(1.0 - alpha_tensor).add(overlay.mul(alpha_tensor))

    return (blended.clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy())


class ChromaticAberrationShift(A.ImageOnlyTransform):
    """Shift red and blue channels independently to mimic optical misalignment."""

    def __init__(self, shift_range: Tuple[int, int] = (2, 5), always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.shift_range = shift_range

    def get_params(self) -> dict:
        magnitude = int(np.random.randint(self.shift_range[0], self.shift_range[1] + 1))
        return {
            "red_dx": int(np.random.randint(-magnitude, magnitude + 1)),
            "red_dy": int(np.random.randint(-magnitude, magnitude + 1)),
            "blue_dx": int(np.random.randint(-magnitude, magnitude + 1)),
            "blue_dy": int(np.random.randint(-magnitude, magnitude + 1)),
        }

    def apply(
        self,
        img: np.ndarray,
        red_dx: int = 0,
        red_dy: int = 0,
        blue_dx: int = 0,
        blue_dy: int = 0,
        **params,
    ) -> np.ndarray:  # type: ignore[override]
        height, width = img.shape[:2]
        transform_red = np.eye(2, 3, dtype=np.float32)
        transform_red[0, 2] = float(red_dx)
        transform_red[1, 2] = float(red_dy)
        transform_blue = np.eye(2, 3, dtype=np.float32)
        transform_blue[0, 2] = float(blue_dx)
        transform_blue[1, 2] = float(blue_dy)

        red = cv2.warpAffine(
            img[:, :, 0],
            transform_red,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        green = img[:, :, 1]
        blue = cv2.warpAffine(
            img[:, :, 2],
            transform_blue,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        return np.stack([red, green, blue], axis=-1)


class Vignetting(A.ImageOnlyTransform):
    """Darken image periphery with a radial falloff to mimic foldscope optics."""

    def __init__(self, strength_range: Tuple[float, float] = (0.30, 0.60), always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.strength_range = strength_range

    def get_params(self) -> dict:
        return {"strength": float(np.random.uniform(self.strength_range[0], self.strength_range[1]))}

    def apply(self, img: np.ndarray, strength: float = 0.4, **params) -> np.ndarray:  # type: ignore[override]
        height, width = img.shape[:2]
        yy, xx = np.ogrid[:height, :width]
        center_y = height * 0.5
        center_x = width * 0.5
        radius = np.sqrt(((yy - center_y) / max(center_y, 1.0)) ** 2 + ((xx - center_x) / max(center_x, 1.0)) ** 2)
        mask = np.clip(1.0 - strength * np.square(radius), 0.25, 1.0)
        vignetted = img.astype(np.float32) * mask[..., None]
        return np.clip(vignetted, 0, 255).astype(np.uint8)


class DustAndBubbleArtifacts(A.ImageOnlyTransform):
    """Add small dust clusters and occasional bubble-like occlusions."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)

    def get_params(self) -> dict:
        dust_count = int(np.random.randint(1, 4))
        dust_specs: List[Tuple[int, int, int, float]] = []
        for _ in range(dust_count):
            dust_specs.append(
                (
                    int(np.random.randint(0, 224)),
                    int(np.random.randint(0, 224)),
                    int(np.random.randint(2, 6)),
                    float(np.random.uniform(40.0, 100.0)),
                )
            )
        return {
            "dust_specs": dust_specs,
            "bubble": bool(np.random.rand() < 0.05),
            "bubble_x": int(np.random.randint(24, 200)),
            "bubble_y": int(np.random.randint(24, 200)),
            "bubble_r": int(np.random.randint(12, 32)),
        }

    def apply(
        self,
        img: np.ndarray,
        dust_specs: Sequence[Tuple[int, int, int, float]] = (),
        bubble: bool = False,
        bubble_x: int = 0,
        bubble_y: int = 0,
        bubble_r: int = 0,
        **params,
    ) -> np.ndarray:  # type: ignore[override]
        overlay = img.astype(np.float32).copy()
        height, width = overlay.shape[:2]

        if dust_specs:
            dust_layer = np.zeros((height, width), dtype=np.float32)
            for x, y, radius, intensity in dust_specs:
                cv2.circle(dust_layer, (x, y), radius, (float(intensity),), thickness=-1)
            dust_layer = cv2.GaussianBlur(dust_layer, (0, 0), sigmaX=1.5, sigmaY=1.5)
            dust_layer = np.clip(dust_layer / 255.0, 0.0, 0.45)
            overlay = overlay * (1.0 - dust_layer[..., None])

        if bubble:
            bubble_mask = np.zeros((height, width), dtype=np.float32)
            cv2.circle(bubble_mask, (bubble_x, bubble_y), bubble_r, (255.0,), thickness=-1)
            bubble_mask = cv2.GaussianBlur(bubble_mask, (0, 0), sigmaX=2.0, sigmaY=2.0)
            bubble_mask = np.clip(bubble_mask / 255.0, 0.0, 1.0)
            bubble_layer = np.full_like(overlay, 255.0)
            overlay = overlay * (1.0 - bubble_mask[..., None]) + bubble_layer * bubble_mask[..., None]

        return np.clip(overlay, 0, 255).astype(np.uint8)


def stain_jitter(image: np.ndarray, hue_shift: int | None = None, saturation_scale: float | None = None) -> np.ndarray:
    """Apply stain variance by perturbing hue and saturation in HSV space.

    Hue is shifted by up to approximately ±15 percent of the OpenCV hue range,
    which corresponds to field-to-field Giemsa staining variation in a compact
    foldscope workflow.
    """

    image = _as_rgb_uint8(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    if hue_shift is None:
        hue_shift = int(np.random.randint(-27, 28))
    if saturation_scale is None:
        saturation_scale = float(np.random.uniform(0.80, 1.25))

    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0.0, 255.0)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


class StainJitterTransform(A.ImageOnlyTransform):
    """Albumentations wrapper around the stain_jitter function."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:  # type: ignore[override]
        return stain_jitter(img)


def get_foldscope_pipeline(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """Build the optical and environmental simulation pipeline.

    The pipeline is deliberately foldscope-like: chromatic aberration, strong
    focus blur, vignetting, dust, and bubble occlusion. The returned Compose
    object exposes a ``curriculum_targets`` attribute that can be used by the
    CurriculumAugmentor to ramp probabilities over training.
    """

    chromatic = ChromaticAberrationShift(shift_range=(2, 5), p=0.85)
    vignette = Vignetting(strength_range=(0.30, 0.60), p=0.90)
    focus_blur = A.OneOf(
        [
            A.GaussianBlur(blur_limit=(7, 13), sigma_limit=(1.5, 3.5), p=1.0),
            A.MotionBlur(blur_limit=13, p=1.0),
            A.GlassBlur(sigma=0.9, max_delta=5, iterations=2, mode="fast", p=1.0),
        ],
        p=0.95,
    )
    artifacts = DustAndBubbleArtifacts(p=0.90)
    stain = StainJitterTransform(p=0.85)

    pipeline = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            chromatic,
            vignette,
            focus_blur,
            artifacts,
            stain,
        ]
    )
    pipeline.curriculum_targets = [chromatic, vignette, focus_blur, artifacts, stain]  # type: ignore[attr-defined]
    return pipeline


class CurriculumAugmentor:
    """Linearly schedule augmentation probabilities across training epochs."""

    def __init__(self, pipeline: A.Compose, max_epoch: int = 40) -> None:
        self.pipeline = pipeline
        self.max_epoch = max(2, int(max_epoch))
        self.targets = list(getattr(pipeline, "curriculum_targets", self._discover_targets(pipeline)))
        self.base_probabilities = {id(transform): float(getattr(transform, "p", 1.0)) for transform in self.targets}

    def _discover_targets(self, pipeline: A.Compose) -> List[A.BasicTransform]:
        """Fallback target discovery when the pipeline did not publish explicit targets."""

        targets: List[A.BasicTransform] = []
        for transform in getattr(pipeline, "transforms", []):
            if isinstance(transform, (A.Resize,)):
                continue
            if hasattr(transform, "p"):
                targets.append(transform)
        return targets

    def update_epoch(self, epoch: int) -> float:
        """Update transform probabilities for a given epoch and return the scale."""

        epoch = int(epoch)
        progress = np.clip((epoch - 1) / float(self.max_epoch - 1), 0.0, 1.0)
        for transform in self.targets:
            base_probability = self.base_probabilities[id(transform)]
            transform.p = float(base_probability * progress)
        return float(progress)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply the scheduled Albumentations pipeline to a single image."""

        result = self.pipeline(image=image)
        return result["image"]


class ThickSmearGenerator:
    """Generate synthetic NIH malaria images for field-ready thick-smear simulation."""

    def __init__(
        self,
        dataset_root: str | Path,
        image_size: Tuple[int, int] = (224, 224),
        seed: int = 42,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)
        self.infected_paths, self.clean_paths = self._collect_dataset_paths()

    def _collect_dataset_paths(self) -> Tuple[List[Path], List[Path]]:
        """Collect infected and uninfected NIH cell image paths."""

        candidate_roots = [
            self.dataset_root,
            self.dataset_root / "cell_images",
            self.dataset_root / "nih_data" / "cell_images",
        ]
        for candidate in candidate_roots:
            parasitized = candidate / "Parasitized"
            uninfected = candidate / "Uninfected"
            if parasitized.is_dir() and uninfected.is_dir():
                infected_paths = sorted(path for path in parasitized.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
                clean_paths = sorted(path for path in uninfected.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
                if not infected_paths or not clean_paths:
                    continue
                logging.info("Found NIH dataset folders: %s | %s", parasitized, uninfected)
                return infected_paths, clean_paths
        raise FileNotFoundError(
            f"Could not locate NIH dataset folders under {self.dataset_root}. Expected Parasitized and Uninfected subfolders."
        )

    def _sample_paths(self, count: int, infected: bool | None = None) -> List[Path]:
        """Sample image paths with replacement from the requested class pool."""

        if infected is True:
            pool = self.infected_paths
        elif infected is False:
            pool = self.clean_paths
        else:
            pool = self.infected_paths + self.clean_paths

        if not pool:
            raise ValueError("The requested class pool is empty.")

        indices = self.rng.integers(0, len(pool), size=count)
        return [pool[int(index)] for index in indices]

    def _load_images(self, paths: Sequence[Path]) -> List[np.ndarray]:
        """Load and resize a sequence of image paths."""

        return [load_rgb_image(path, self.image_size) for path in paths]

    def stacking(self) -> Tuple[np.ndarray, float, List[str]]:
        """Overlay 2-4 cell images with weighted alpha-blend to simulate bunching."""

        num_cells = int(self.rng.integers(2, 5))
        paths = self._sample_paths(num_cells, infected=None)
        images = self._load_images(paths)
        alphas = [1.0]
        alphas.extend(float(self.rng.uniform(0.40, 0.70)) for _ in range(1, num_cells))

        blended_image = _torch_blend(images, alphas)

        soft_label = float(self._blend_soft_label([self._infer_label(path) for path in paths], alphas))
        return blended_image, soft_label, [str(path) for path in paths]

    def custom_cutmix(self) -> Tuple[np.ndarray, float, List[str]]:
        """Replace 10-30 percent of a clean cell with irregular infected fragments."""

        clean_path = self._sample_paths(1, infected=False)[0]
        infected_path = self._sample_paths(1, infected=True)[0]
        clean_image = load_rgb_image(clean_path, self.image_size)
        infected_image = load_rgb_image(infected_path, self.image_size)

        mask = self._fragment_mask((int(clean_image.shape[0]), int(clean_image.shape[1])))
        mixed_image = clean_image.astype(np.float32) * (1.0 - mask[..., None]) + infected_image.astype(np.float32) * mask[..., None]
        mixed_image = np.clip(mixed_image, 0, 255).astype(np.uint8)
        soft_label = float(mask.mean())

        return mixed_image, soft_label, [str(clean_path), str(infected_path)]

    def clean_sample(self) -> Tuple[np.ndarray, float, List[str]]:
        """Return a single image after foldscope simulation and stain jitter."""

        sampled_path = self._sample_paths(1, infected=None)[0]
        sampled_image = load_rgb_image(sampled_path, self.image_size)
        return sampled_image, float(self._infer_label(sampled_path)), [str(sampled_path)]

    def _infer_label(self, image_path: Path) -> int:
        """Infer the binary label from the source directory name."""

        return 1 if "Parasitized" in image_path.parts else 0

    def _blend_soft_label(self, labels: Sequence[int], alphas: Sequence[float]) -> float:
        """Blend labels with the same recurrence used for the pixel compositing."""

        label_value = float(labels[0])
        for label, alpha in zip(labels[1:], alphas[1:]):
            label_value = label_value * (1.0 - alpha) + float(label) * alpha
        return float(np.clip(label_value, 0.0, 1.0))

    def _fragment_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create a soft irregular mask that covers 10-30 percent of the image."""

        height, width = shape
        target_fraction = float(self.rng.uniform(0.10, 0.30))
        mask = np.zeros((height, width), dtype=np.float32)

        fragment_count = int(self.rng.integers(2, 5))
        for _ in range(fragment_count):
            center_x = int(self.rng.integers(0, width))
            center_y = int(self.rng.integers(0, height))
            radius_x = int(self.rng.integers(max(4, width // 16), max(6, width // 5)))
            radius_y = int(self.rng.integers(max(4, height // 16), max(6, height // 5)))
            angle = float(self.rng.uniform(0.0, 180.0))
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), angle, 0.0, 360.0, (1.0,), thickness=-1)

        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=4.0, sigmaY=4.0)
        threshold = np.quantile(mask, max(0.0, 1.0 - target_fraction))
        binary_mask = (mask >= threshold).astype(np.float32)
        if binary_mask.mean() < target_fraction * 0.8:
            binary_mask = np.clip(mask / (mask.max() + 1e-6), 0.0, 1.0)
        return np.clip(binary_mask, 0.0, 1.0)

    def generate_example(
        self,
        pipeline: A.Compose,
        mode: str = "auto",
    ) -> Tuple[np.ndarray, float, str, List[str]]:
        """Generate one synthetic image and its soft label."""

        if mode == "auto":
            mode = str(self.rng.choice(["stack", "cutmix", "clean"], p=[0.45, 0.30, 0.25]))

        if mode == "stack":
            image, soft_label, sources = self.stacking()
        elif mode == "cutmix":
            image, soft_label, sources = self.custom_cutmix()
        elif mode == "clean":
            image, soft_label, sources = self.clean_sample()
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

        image = pipeline(image=image)["image"]
        return image, soft_label, mode, sources


def generate_synthetic_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    num_samples: int,
    seed: int = 42,
    image_size: Tuple[int, int] = (224, 224),
    curriculum_epoch: int = 40,
) -> Path:
    """Generate the synthetic image folder and labels CSV.

    Parameters
    ----------
    dataset_root:
        Root directory containing NIH cell images.
    output_dir:
        Directory where PNGs and labels.csv will be written.
    num_samples:
        Number of synthetic samples to create.
    seed:
        Random seed for reproducibility.
    image_size:
        Final output size for all synthetic images.
    curriculum_epoch:
        Epoch number used to scale augmentation probability.

    Returns
    -------
    Path
        The directory that contains the generated dataset.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ThickSmearGenerator(dataset_root=dataset_root, image_size=image_size, seed=seed)
    pipeline = get_foldscope_pipeline(image_size=image_size)
    curriculum = CurriculumAugmentor(pipeline, max_epoch=40)
    curriculum.update_epoch(curriculum_epoch)

    labels_path = output_dir / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filename", "soft_label", "hard_label", "operation", "source_paths"],
        )
        writer.writeheader()

        for index in range(num_samples):
            image, soft_label, operation, sources = generator.generate_example(pipeline=pipeline)
            filename = f"synthetic_{index:06d}.png"
            save_rgb_image(output_dir / filename, image)

            writer.writerow(
                {
                    "filename": filename,
                    "soft_label": f"{soft_label:.6f}",
                    "hard_label": int(soft_label >= 0.5),
                    "operation": operation,
                    "source_paths": "|".join(sources),
                }
            )

    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI used to generate the synthetic dataset."""

    parser = argparse.ArgumentParser(description="Generate a synthetic field-ready malaria dataset")
    parser.add_argument("--dataset-root", type=str, default="nih_data", help="Root directory containing Parasitized/ and Uninfected/ folders.")
    parser.add_argument("--output-dir", type=str, default="synthetic_field_ready", help="Directory where PNGs and labels.csv will be written.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of synthetic images to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epoch", type=int, default=40, help="Curriculum epoch used to scale augmentation probability.")
    return parser


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = build_arg_parser().parse_args()
    output_dir = generate_synthetic_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        curriculum_epoch=args.epoch,
    )
    logging.info("Generated synthetic dataset at %s", output_dir)


if __name__ == "__main__":
    main()