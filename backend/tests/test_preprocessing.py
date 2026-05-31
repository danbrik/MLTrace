from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.preprocessing.base import BasePreprocessingStep, ImageSpec
from app.preprocessing.pipeline import execute_preview, validate_linear_graph
from app.preprocessing.registry import PreprocessingRegistry, registry
from app.preprocessing.steps.crop import CropStep
from app.preprocessing.steps.gaussian_blur import GaussianBlurStep
from app.preprocessing.steps.grayscale import GrayscaleStep
from app.preprocessing.steps.load_image import LoadImageStep
from app.preprocessing.steps.resize import ResizeStep
from app.schemas import PreprocessingGraph


def graph(nodes: list[dict], edges: list[dict]) -> PreprocessingGraph:
    return PreprocessingGraph(nodes=nodes, edges=edges)


def test_registry_discovers_core_steps() -> None:
    step_types = {definition.type for definition in registry.list_definitions()}

    assert {
        "load_image",
        "warp_perspective",
        "resize",
        "crop",
        "grayscale",
        "normalize_for_preview",
        "gaussian_blur",
    }.issubset(step_types)


def test_core_steps_transform_image_shapes() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    resized = ResizeStep().apply(image, {"width": 7, "height": 5}, {})
    gray = GrayscaleStep().apply(image, {}, {})

    assert resized.shape == (5, 7, 3)
    assert gray.shape == (10, 20)


def test_gaussian_blur_keeps_shape_and_forces_odd_kernel() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    # An even kernel size is bumped to the next odd value instead of raising.
    blurred = GaussianBlurStep().apply(image, {"kernel_size": 4}, {})

    assert blurred.shape == image.shape


def test_linear_validation_rejects_invalid_graphs() -> None:
    with pytest.raises(ValueError, match="load_image"):
        validate_linear_graph(graph([{"id": "resize", "type": "resize", "config": {}}], []))

    with pytest.raises(ValueError, match="branch"):
        validate_linear_graph(
            graph(
                [
                    {"id": "load", "type": "load_image", "config": {}},
                    {"id": "a", "type": "resize", "config": {}},
                    {"id": "b", "type": "crop", "config": {}},
                ],
                [
                    {"source": "load", "target": "a"},
                    {"source": "load", "target": "b"},
                ],
            )
        )


def test_execute_preview_returns_one_preview_per_node(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.tif"
    Image.new("RGB", (20, 10), color=(128, 64, 32)).save(image_path)

    preview = execute_preview(
        graph(
            [
                {"id": "load", "type": "load_image", "config": {}},
                {"id": "resize", "type": "resize", "config": {"width": 8, "height": 4}},
            ],
            [{"source": "load", "target": "resize"}],
        ),
        str(image_path),
    )

    assert [item.step_type for item in preview] == ["load_image", "resize"]
    assert preview[0].width == 20
    assert preview[1].width == 8
    assert preview[1].height == 4
    assert preview[1].image_data_url.startswith("data:image/png;base64,")


def test_load_image_size_lock_accepts_matching_size(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.tif"
    Image.new("RGB", (20, 10), color=(128, 64, 32)).save(image_path)

    image = LoadImageStep().apply(
        None,
        {"lock_size": True, "lock_width": 20, "lock_height": 10},
        {"source_image_path": str(image_path)},
    )

    assert image.shape == (10, 20, 3)


def test_load_image_converts_dtype(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.tif"
    Image.new("RGB", (20, 10), color=(128, 64, 32)).save(image_path)

    image = LoadImageStep().apply(
        None,
        {"dtype": "float32"},
        {"source_image_path": str(image_path)},
    )

    assert image.dtype == np.float32


def test_load_image_size_lock_rejects_mismatched_size(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.tif"
    Image.new("RGB", (20, 10), color=(128, 64, 32)).save(image_path)

    with pytest.raises(ValueError, match="Input size is locked to 21x10"):
        LoadImageStep().apply(
            None,
            {"lock_size": True, "lock_width": 21, "lock_height": 10},
            {"source_image_path": str(image_path)},
        )


def test_grayscale_propagates_single_channel(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.tif"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image_path)

    preview = execute_preview(
        graph(
            [
                {"id": "load", "type": "load_image", "config": {}},
                {"id": "gray", "type": "grayscale", "config": {}},
            ],
            [{"source": "load", "target": "gray"}],
        ),
        str(image_path),
    )

    assert preview[0].channels == 3
    assert preview[1].channels == 1


def test_type_chain_rejects_incompatible_step() -> None:
    class ColorOnlyStep(BasePreprocessingStep):
        type = "color_only"
        label = "Color only"
        category = "Test"

        def output_spec(self, spec_in: ImageSpec | None, config: dict) -> ImageSpec:
            if spec_in is None:
                raise ValueError("color_only requires an input image.")
            if spec_in.channels != 3:
                raise ValueError("color_only requires a 3-channel image.")
            return spec_in

        def apply(self, image, config, context):  # pragma: no cover - not executed here
            return image

    reg = PreprocessingRegistry()
    reg.register(LoadImageStep())
    reg.register(GrayscaleStep())
    reg.register(ColorOnlyStep())

    grayscale_chain = graph(
        [
            {"id": "load", "type": "load_image", "config": {"mode": "grayscale"}},
            {"id": "color", "type": "color_only", "config": {}},
        ],
        [{"source": "load", "target": "color"}],
    )
    with pytest.raises(ValueError, match="3-channel"):
        validate_linear_graph(grayscale_chain, active_registry=reg)

    rgb_chain = graph(
        [
            {"id": "load", "type": "load_image", "config": {"mode": "rgb"}},
            {"id": "color", "type": "color_only", "config": {}},
        ],
        [{"source": "load", "target": "color"}],
    )
    validate_linear_graph(rgb_chain, active_registry=reg)  # does not raise


def test_validate_step_config_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        validate_linear_graph(
            graph(
                [
                    {"id": "load", "type": "load_image", "config": {}},
                    {"id": "blur", "type": "gaussian_blur", "config": {"kernel_size": 0}},
                ],
                [{"source": "load", "target": "blur"}],
            )
        )


def test_crop_output_size_modes() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    region = {"x": 2, "y": 1, "width": 5, "height": 4}

    cropped = CropStep().apply(image, {**region, "output_size": "cropped"}, {})
    assert cropped.shape == (4, 5, 3)

    to_input = CropStep().apply(image, {**region, "output_size": "input"}, {})
    assert to_input.shape == (10, 20, 3)

    to_source = CropStep().apply(image, {**region, "output_size": "source"}, {"source_shape": (30, 40, 3)})
    assert to_source.shape == (30, 40, 3)
