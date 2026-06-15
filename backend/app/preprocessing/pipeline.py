from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.preprocessing.base import ImageSpec
from app.preprocessing.registry import PreprocessingRegistry, registry
import app.preprocessing.steps  # noqa: F401 ensures core steps are registered
from app.schemas import PreprocessingGraph, PreprocessingPreviewImage


def validate_step_config(step, config: dict) -> None:
    """Validate a step's config against its config_schema (type, min/max, enum)."""
    properties = step.config_schema.get("properties", {})
    merged = step.merged_config(config)
    for key, prop in properties.items():
        if key not in merged:
            continue
        value = merged[key]
        enum = prop.get("enum")
        if enum is not None:
            if value not in enum:
                raise ValueError(f"{step.type}.{key} must be one of {enum}, got {value!r}.")
            continue
        if prop.get("type") in ("integer", "number"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{step.type}.{key} must be a number, got {value!r}.")
            minimum = prop.get("minimum")
            maximum = prop.get("maximum")
            if minimum is not None and value < minimum:
                raise ValueError(f"{step.type}.{key} must be >= {minimum}, got {value}.")
            if maximum is not None and value > maximum:
                raise ValueError(f"{step.type}.{key} must be <= {maximum}, got {value}.")


@dataclass(frozen=True)
class OrderedNode:
    id: str
    type: str
    config: dict
    position: dict | None = None


def validate_linear_graph(graph: PreprocessingGraph, active_registry: PreprocessingRegistry = registry) -> list[OrderedNode]:
    nodes = graph.nodes
    edges = graph.edges
    ids = [node.id for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Pipeline node ids must be unique.")
    if sum(1 for node in nodes if node.type == "load_image") != 1:
        raise ValueError("V1 pipelines must contain exactly one load_image node.")

    node_by_id = {node.id: node for node in nodes}
    for node in nodes:
        active_registry.get(node.type)

    if len(edges) != len(nodes) - 1:
        raise ValueError("V1 pipelines must be one connected linear chain.")

    indegree = {node_id: 0 for node_id in ids}
    outgoing: dict[str, str] = {}
    for edge in edges:
        if edge.source not in node_by_id or edge.target not in node_by_id:
            raise ValueError("Pipeline edge references an unknown node.")
        if edge.source in outgoing:
            raise ValueError("V1 pipelines cannot branch.")
        outgoing[edge.source] = edge.target
        indegree[edge.target] += 1
        if indegree[edge.target] > 1:
            raise ValueError("V1 pipelines cannot merge branches.")

    starts = [node_id for node_id, count in indegree.items() if count == 0]
    if len(starts) != 1:
        raise ValueError("V1 pipelines must have exactly one start node.")
    start_id = starts[0]
    if node_by_id[start_id].type != "load_image":
        raise ValueError("The first pipeline step must be load_image.")

    ordered: list[OrderedNode] = []
    visited: set[str] = set()
    current = start_id
    spec: ImageSpec | None = None
    while current:
        if current in visited:
            raise ValueError("Pipeline contains a cycle.")
        visited.add(current)
        node = node_by_id[current]
        step = active_registry.get(node.type)
        merged = {**step.default_config, **(node.config or {})}
        validate_step_config(step, merged)
        # Thread the symbolic image spec through the chain; raises if a step cannot
        # consume the previous step's output (the type chain).
        spec = step.output_spec(spec, merged)
        ordered.append(OrderedNode(id=node.id, type=node.type, config=merged, position=node.position))
        current = outgoing.get(current, "")

    if len(visited) != len(nodes):
        raise ValueError("Pipeline contains disconnected nodes.")

    return ordered


def image_metadata(image: np.ndarray) -> tuple[int, int, int, str, float, float]:
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    return width, height, channels, str(image.dtype), float(np.min(image)), float(np.max(image))


def encode_png_data_url(image: np.ndarray) -> str:
    array = image
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        if maximum > minimum:
            array = ((array - minimum) / (maximum - minimum) * 255).astype(np.uint8)
        else:
            array = np.zeros_like(array, dtype=np.uint8)

    pil_image = Image.fromarray(array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def execute_with_previews(
    graph: PreprocessingGraph, source_image_path: str
) -> tuple[list[PreprocessingPreviewImage], np.ndarray]:
    """Run the pipeline on one image, returning per-step previews and the final array.

    The final numpy array is what downstream consumers (e.g. a model forward
    pass) operate on; the previews are display-normalized PNG snapshots.
    """
    ordered_nodes = validate_linear_graph(graph)
    context = {"source_image_path": source_image_path}
    image: np.ndarray | None = None
    previews: list[PreprocessingPreviewImage] = []

    for index, node in enumerate(ordered_nodes):
        step = registry.get(node.type)
        image = step.apply(image, node.config, context)
        if index == 0:
            # Remember the original image size for steps that interpolate back to it (crop "source").
            context["source_shape"] = image.shape
        width, height, channels, dtype, value_min, value_max = image_metadata(image)
        previews.append(
            PreprocessingPreviewImage(
                node_id=node.id,
                step_type=node.type,
                label=step.label,
                width=width,
                height=height,
                channels=channels,
                dtype=dtype,
                value_min=value_min,
                value_max=value_max,
                image_data_url=encode_png_data_url(image),
            )
        )

    assert image is not None  # validate_linear_graph guarantees at least the load_image node
    return previews, image


def execute_preview(graph: PreprocessingGraph, source_image_path: str) -> list[PreprocessingPreviewImage]:
    previews, _ = execute_with_previews(graph, source_image_path)
    return previews


def run_pipeline_array(graph: PreprocessingGraph, source_image_path: str) -> np.ndarray:
    """Run the pipeline on one image and return only the final numpy array.

    Used by training (which processes many images): unlike execute_with_previews
    it skips the per-step PNG encoding, so it is much cheaper in a loop.
    """
    ordered_nodes = validate_linear_graph(graph)
    context = {"source_image_path": source_image_path}
    image: np.ndarray | None = None
    for index, node in enumerate(ordered_nodes):
        step = registry.get(node.type)
        image = step.apply(image, node.config, context)
        if index == 0:
            context["source_shape"] = image.shape
    assert image is not None  # validate_linear_graph guarantees the load_image node
    return image
