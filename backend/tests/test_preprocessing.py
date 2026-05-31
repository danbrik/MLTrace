from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.preprocessing.pipeline import execute_preview, validate_linear_graph
from app.preprocessing.registry import registry
from app.preprocessing.steps.gaussian_blur import GaussianBlurStep
from app.preprocessing.steps.grayscale import GrayscaleStep
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
