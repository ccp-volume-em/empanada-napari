"""Segmentations are placed with the whole transform of the image layer."""
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel
from napari.layers import Image, Labels

from empanada_napari._slice_inference import SliceInferenceWidget
from empanada_napari._volume_inference import VolumeInferenceWidget
from empanada_napari.utils import spatial_kwargs

from .conftest import MODEL_NAMES

NO_PBAR = SimpleNamespace(hide=lambda: None)


def transformed_image(shape, **overrides):
    transform = dict(scale=(0.5, 0.25), translate=(100.0, -40.0), rotate=-135,
                     shear=[0.1], units=('um', 'um'))
    transform.update(overrides)
    return Image(np.zeros(shape, dtype=np.uint8), **transform)


def corners(shape):
    return np.array(np.meshgrid(*[[0, n - 1] for n in shape], indexing='ij')).reshape(len(shape), -1).T


def assert_same_place(labels, image, offset=(0, 0)):
    """Every corner pixel of the labels lies on the image pixel it was computed from."""
    for point in corners(labels.data.shape):
        np.testing.assert_allclose(labels.data_to_world(point),
                                   image.data_to_world(point + np.asarray(offset)),
                                   atol=1e-9)


def test_labels_take_the_whole_transform_of_the_image():
    image = transformed_image((40, 60), affine=[[1, 0, 5], [0, 1, 7], [0, 0, 1]])
    labels = Labels(np.zeros((40, 60), dtype=np.int32), **spatial_kwargs(image))
    assert_same_place(labels, image)
    assert tuple(str(unit) for unit in labels.units) == ('micrometer', 'micrometer')


@pytest.mark.parametrize('rotate', [0, 30, -135])
def test_a_crop_lands_where_it_was_cut_from(rotate):
    image = transformed_image((40, 60), rotate=rotate)
    offset = (12, 25)
    labels = Labels(np.zeros((10, 20), dtype=np.int32), **spatial_kwargs(image, offset))
    assert_same_place(labels, image, offset)


def test_axes_leave_out_a_channel_axis():
    image = Image(np.zeros((3, 40, 60), dtype=np.uint8), scale=(1, 0.5, 0.25),
                  translate=(0, 100, -40), rotate=[[1, 0, 0],
                                                   [0, np.cos(1), -np.sin(1)],
                                                   [0, np.sin(1), np.cos(1)]])
    labels = Labels(np.zeros((40, 60), dtype=np.int32), **spatial_kwargs(image, axes=[1, 2]))
    for point in corners((40, 60)):
        np.testing.assert_allclose(labels.data_to_world(point),
                                   image.data_to_world([0, *point])[1:], atol=1e-9)


def test_slice_inference_places_an_roi_result_on_a_rotated_image():
    viewer = ViewerModel()
    image = viewer.add_image(np.zeros((40, 60), dtype=np.uint8), scale=(0.5, 0.25),
                             translate=(100, -40), rotate=-135)
    widget = SliceInferenceWidget(viewer=viewer, image_layer=image,
                                  model_config=MODEL_NAMES['MitoNet_mini'], pbar=NO_PBAR)
    seg = np.ones((10, 20), dtype=np.int32)
    widget._show_test_result((seg, None, None, 12, 25))
    assert_same_place(viewer.layers['empanada_seg_2d'], image, (12, 25))


def test_volume_inference_places_the_segmentation_on_a_rotated_volume():
    viewer = ViewerModel()
    image = viewer.add_image(np.zeros((5, 40, 60), dtype=np.uint8), scale=(2, 0.5, 0.25),
                             translate=(3, 100, -40), rotate=[[1, 0, 0],
                                                              [0, np.cos(1), -np.sin(1)],
                                                              [0, np.sin(1), np.cos(1)]])
    widget = VolumeInferenceWidget(viewer=viewer, image_layer=image,
                                   model_config=MODEL_NAMES['MitoNet_mini'], pbar=NO_PBAR)
    widget._new_layers(np.zeros((5, 40, 60), dtype=np.int32), 'panoptic')
    assert_same_place(viewer.layers[f'{image.name}-panoptic'], image, (0, 0, 0))
