import copy

import cv2
import numpy as np
import pytest
from PIL import Image

from app.image_geometry import RotatedRectangle, aligned_crop, selection_mask, validate_selection
from app.preprocessing.base import ImageSpec
from app.preprocessing.pipeline import compile_pipeline, execute_preview, validate_linear_graph
from app.preprocessing.steps.crop import CropStep
from app.preprocessing.utils import interpolation_flag
from tests.test_preprocessing import graph


def rectangle(angle=0, **changes):
    return dict(version=2, center_x=4, center_y=4, width=4, height=4, angle_degrees=angle, **changes)


@pytest.mark.parametrize('angle', [0, -30, 30, 45, 90, 180])
@pytest.mark.parametrize('dtype,channels', [('uint8', 1), ('uint16', 1), ('float64', 1), ('uint16', 3)])
def test_rotated_crop_preserves_values_dtype_and_channels(angle, dtype, channels):
    data = (np.arange(64 * channels).reshape((8, 8) if channels == 1 else (8, 8, channels)) * 127).astype(dtype)
    cfg = {'roi': rectangle(angle)}
    output = CropStep().apply(data, cfg, {})
    assert output.dtype == data.dtype and output.shape == ((4, 4) if channels == 1 else (4, 4, channels))
    np.testing.assert_array_equal(output, aligned_crop(data, RotatedRectangle(**cfg['roi'])))
    assert set(output.ravel()).issubset(set(data.ravel()))
    if angle == 0:
        np.testing.assert_array_equal(output, data[2:6, 2:6])
    if angle in (90, 180):
        np.testing.assert_array_equal(output, np.rot90(data[2:6, 2:6], angle // 90))


@pytest.mark.parametrize('mode,target', [('cropped', (4, 4)), ('input', (8, 8)), ('source', (12, 10))])
@pytest.mark.parametrize('interpolation', ['nearest', 'linear', 'area', 'cubic'])
def test_resize_only_after_alignment(mode, target, interpolation):
    image = np.arange(64, dtype=np.uint16).reshape(8, 8) * 300
    roi = RotatedRectangle(**rectangle(30))
    output = CropStep().apply(image, {'roi': roi.model_dump(), 'output_size': mode, 'interpolation': interpolation}, {'source_shape': (10, 12)})
    expected = aligned_crop(image, roi)
    if mode != 'cropped':
        expected = cv2.resize(expected, target, interpolation=interpolation_flag(interpolation))
    np.testing.assert_array_equal(output, expected)
    spec = CropStep().output_spec(ImageSpec(width=8, height=8, dtype='uint16', channels=1), {'roi': roi.model_dump(), 'output_size': mode})
    assert (spec.width, spec.height) == (target if mode != 'source' else (None, None))


@pytest.mark.parametrize('roi', [
    {**rectangle(45), 'width': 8, 'height': 8},
    {**rectangle(), 'center_x': -1},
    {**rectangle(), 'width': 0},
    {**rectangle(), 'width': 1.5},
    {**rectangle(), 'angle_degrees': float('nan')},
    {**rectangle(), 'center_y': float('inf')},
    {**rectangle(), 'version': 3},
    {'version': 2},
    dict(version=2, center_x=1, center_y=1, width=1, height=1, angle_degrees=45),
])
def test_invalid_selection_rejected_at_validation_and_execution(roi):
    with pytest.raises(ValueError):
        CropStep().output_spec(ImageSpec(width=8, height=8), {'roi': roi})
    with pytest.raises(ValueError):
        CropStep().apply(np.zeros((8, 8)), {'roi': roi}, {})


def test_legacy_bounds_and_geometry_precedence():
    image = np.arange(64).reshape(8, 8)
    old = {'x': 6, 'y': 7, 'width': 10, 'height': 8}
    frozen = copy.deepcopy(old)
    np.testing.assert_array_equal(CropStep().apply(image, old, {}), image[7:, 6:])
    assert old == frozen
    np.testing.assert_array_equal(CropStep().apply(image, {'roi': rectangle(), 'x': -999, 'width': -1}, {}), image[2:6, 2:6])
    schema = graph([{'id': 'load', 'type': 'load_image', 'config': {}},
                    {'id': 'crop', 'type': 'crop', 'config': {'roi': rectangle(), 'x': -999, 'width': -1}}],
                   [{'source': 'load', 'target': 'crop'}])
    validate_linear_graph(schema)


def test_source_pixel_validation_matches_statistics_mask():
    for angle in [0, 30, -30, 45, 90, 180]:
        for center in [1, 1.1, 1.5, 2]:
            roi = RotatedRectangle(version=2, center_x=center, center_y=center, width=1, height=1, angle_degrees=angle)
            try:
                selection_mask(roi, 4, 4)
            except ValueError:
                with pytest.raises(ValueError):
                    validate_selection(roi, 4, 4)
            else:
                validate_selection(roi, 4, 4)


def test_pipeline_after_resize_multiple_crops_and_runtime_bounds(tmp_path):
    image = np.arange(144, dtype=np.uint16).reshape(12, 12) * 200
    path = tmp_path / 'image.tiff'; Image.fromarray(image).save(path)
    nodes = [
        {'id': 'load', 'type': 'load_image', 'config': {}},
        {'id': 'resize', 'type': 'resize', 'config': {'width': 8, 'height': 8, 'interpolation': 'nearest'}},
        {'id': 'first', 'type': 'crop', 'config': {'roi': rectangle(90)}},
        {'id': 'second', 'type': 'crop', 'config': {'roi': dict(version=2, center_x=2, center_y=2, width=2, height=2, angle_degrees=0)}},
    ]
    config = graph(nodes, [{'source': a['id'], 'target': b['id']} for a, b in zip(nodes, nodes[1:])])
    frozen = config.model_dump()
    output = compile_pipeline(config).run(str(path))
    expected = np.rot90(cv2.resize(image, (8, 8), interpolation=cv2.INTER_NEAREST)[2:6, 2:6])[1:3, 1:3]
    np.testing.assert_array_equal(output, expected)
    assert config.model_dump() == frozen
    previews = execute_preview(config, str(path))
    assert [(p.width, p.height) for p in previews] == [(12, 12), (8, 8), (4, 4), (2, 2)]
    # Unlocked input sizes are checked on every concrete image, not only on compilation.
    direct = graph([nodes[0], nodes[2]], [{'source': 'load', 'target': 'first'}])
    pipeline = compile_pipeline(direct)
    Image.fromarray(np.zeros((3, 3), dtype=np.uint16)).save(path)
    with pytest.raises(ValueError, match='innerhalb'):
        pipeline.run(str(path))
