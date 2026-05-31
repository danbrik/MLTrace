from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.preprocessing.registry import PreprocessingRegistry, registry
import app.preprocessing.steps  # noqa: F401 ensures core steps are registered
from app.schemas import PreprocessingGraph, PreprocessingPreviewImage


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
    while current:
        if current in visited:
            raise ValueError("Pipeline contains a cycle.")
        visited.add(current)
        node = node_by_id[current]
        step = active_registry.get(node.type)
        ordered.append(
            OrderedNode(
                id=node.id,
                type=node.type,
                config={**step.default_config, **(node.config or {})},
                position=node.position,
            )
        )
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


def execute_preview(graph: PreprocessingGraph, source_image_path: str) -> list[PreprocessingPreviewImage]:
    ordered_nodes = validate_linear_graph(graph)
    context = {"source_image_path": source_image_path}
    image: np.ndarray | None = None
    previews: list[PreprocessingPreviewImage] = []

    for node in ordered_nodes:
        step = registry.get(node.type)
        image = step.apply(image, node.config, context)
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

    return previews
