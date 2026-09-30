from dataclasses import replace
import json

import numpy as np
import pytest

from sam3_vlm.core.config import AssociationConfig
from sam3_vlm.core.geometry import Box, GeometryRef, MaskGeometry, detection_mask_geometry, mask_overlap
from sam3_vlm.core.id_generator import IDGenerator
from sam3_vlm.core.types import Detection
from sam3_vlm.scene.association_dual import IoUIoMAssociationPolicy
from sam3_vlm.scene.graph import SceneGraph
from sam3_vlm.scene.node import Node


def det(name, mask, offset=(0, 0), score=.9, clipped=False):
    # Intentionally identical boxes; association must depend on pixels.
    return Detection(name, GeometryRef(Box(0, 0, 100, 100)), score,
                     raw_metadata={'mask': mask, 'mask_offset_x': offset[0], 'mask_offset_y': offset[1],
                                   'crop_boundary_clipped': clipped})


def _ids(graph):
    graph._test_ids = IDGenerator()
    return graph._test_ids


def associate(graph, detections, config=None):
    return IoUIoMAssociationPolicy().associate(
        graph, detections, 'sam1', 'a1', 'target', getattr(graph, '_test_ids', None) or _ids(graph),
        config=config or AssociationConfig(mask_only=True, enable_iom_dedup=True))


def test_identical_boxes_with_disjoint_masks_create_two_nodes():
    a = np.eye(10, dtype=bool)
    b = np.fliplr(a)
    graph = SceneGraph()
    result = associate(graph, [det('a', a), det('b', b)])
    assert len(result.new_nodes) == 2
    assert all(node.diagnostics.duplicate_risk == 0 for node in graph.active_nodes())


def test_cross_pass_disjoint_masks_with_identical_boxes_do_not_match():
    graph = SceneGraph()
    associate(graph, [det('a', np.eye(10, dtype=bool))])
    result = associate(graph, [det('b', np.fliplr(np.eye(10, dtype=bool)))])
    assert len(result.new_nodes) == 1 and not result.matched_observations


def test_iom_deduplicates_offset_tile_fragment_and_keeps_complete_mask():
    graph = SceneGraph()
    associate(graph, [det('complete', np.ones((20, 20), dtype=bool), (100, 200))])
    result = associate(graph, [det('fragment', np.ones((5, 5), dtype=bool), (110, 210), clipped=True)])
    assert not result.new_nodes and len(result.matched_observations) == 1
    assert result.matched_observations[0][1].association_score == 1
    node = graph.active_nodes()[0]
    assert node.geometry.area() == 400
    assert node.geometry.bbox().as_tuple() == (100, 200, 120, 220)


def test_larger_cross_pass_mask_becomes_canonical():
    graph = SceneGraph()
    associate(graph, [det('fragment', np.ones((5, 5), dtype=bool), (110, 210), clipped=True)])
    result = associate(graph, [det('complete', np.ones((20, 20), dtype=bool), (100, 200))])
    assert not result.new_nodes and graph.active_nodes()[0].geometry.area() == 400


def test_score_greedy_within_call_mask_suppression():
    graph = SceneGraph()
    result = associate(graph, [det('low', np.ones((5, 5), dtype=bool), score=.6),
                               det('high', np.ones((5, 5), dtype=bool), score=.9)])
    assert len(result.new_nodes) == 1
    assert result.new_nodes[0].observations[0].detection_id == 'high'


def test_iou_only_mode_does_not_use_containment():
    graph = SceneGraph()
    result = associate(graph, [det('big', np.ones((20, 20), dtype=bool)),
                               det('tiny', np.ones((2, 2), dtype=bool), (5, 5))],
                       AssociationConfig(mask_only=True, enable_iom_dedup=False))
    assert len(result.new_nodes) == 2


def test_iou_only_mode_does_not_apply_containment_group_filter():
    result = associate(SceneGraph(), [det('parent', np.ones((20, 20), dtype=bool)),
        det('child1', np.ones((3, 3), dtype=bool), (2, 2)),
        det('child2', np.ones((3, 3), dtype=bool), (12, 12))],
        AssociationConfig(mask_only=True, enable_iom_dedup=False))
    assert len(result.new_nodes) == 3


def test_trimming_preserves_offset_and_serializes_only_mask_reference():
    pixels = np.zeros((10, 10), dtype=bool)
    pixels[3:5, 2:6] = True
    detection = det('trim', pixels, (100, 200))
    detection.mask_artifact = 'artifacts/masks/trim.npz'
    geometry = detection_mask_geometry(detection)
    assert geometry.offset == (102, 203)
    assert geometry.mask.shape == (2, 4) and geometry.area() == 8
    assert geometry.bbox().as_tuple() == (102, 203, 106, 205)
    node = Node('node', geometry)
    data = json.loads(json.dumps(node.to_dict()))
    restored = Node.from_dict(data)
    assert json.loads(json.dumps(restored.to_dict())) == data
    assert restored.geometry.mask is None
    assert restored.geometry.mask_artifact == 'artifacts/masks/trim.npz'


@pytest.mark.parametrize('mask', [np.zeros((3, 3)), np.ones(3), np.array([[float('nan')]]), np.array([[.5]])])
def test_invalid_masks_fail_closed(mask):
    with pytest.raises(ValueError):
        associate(SceneGraph(), [det('bad', mask)])


def test_missing_mask_never_falls_back_to_box():
    with pytest.raises(ValueError, match='no mask'):
        associate(SceneGraph(), [Detection('missing', GeometryRef(Box(0, 0, 10, 10)), .9)])


def test_partial_iou_duplicates_use_same_rule_within_and_across_calls():
    masks = [det('a', np.ones((20, 20), dtype=bool)),
             det('b', np.ones((20, 20), dtype=bool), (5, 0))]
    graph = SceneGraph()
    assert len(associate(graph, masks).new_nodes) == 1  # IoU=0.6, below old intra-call 0.7.
    assert not associate(graph, masks).new_nodes
    assert len(graph.active_nodes()) == 1


def test_full_image_small_instance_is_not_a_crop_fragment():
    graph = SceneGraph()
    associate(graph, [det('large', np.ones((20, 20), dtype=bool))])
    result = associate(graph, [det('small', np.ones((3, 3), dtype=bool), (3, 3))])
    assert len(result.new_nodes) == 1
    assert len(graph.active_nodes()) == 2
    assert max(n.diagnostics.duplicate_risk for n in graph.active_nodes()) < .1


@pytest.mark.parametrize('parent_first', [False, True])
def test_group_mask_cannot_swallow_disjoint_individual_instances(parent_first):
    parent = det('parent', np.ones((20, 20), dtype=bool), score=.99)
    children = [det('child1', np.ones((3, 3), dtype=bool), (2, 2)),
                det('child2', np.ones((3, 3), dtype=bool), (12, 12))]
    graph = SceneGraph()
    if parent_first:
        associate(graph, [parent])
        result = associate(graph, children)
        assert len(result.rejected_group_nodes) == 1
    else:
        associate(graph, [parent, *children])
    assert len(graph.active_nodes()) == 2
    assert {n.observations[0].detection_id for n in graph.active_nodes()} == {'child1', 'child2'}
    assert not associate(graph, [parent]).new_nodes


def test_negative_query_does_not_retire_target_parent():
    graph = SceneGraph()
    associate(graph, [det('parent', np.ones((20, 20), dtype=bool))])
    result = IoUIoMAssociationPolicy().associate(graph,
        [det('part1', np.ones((3, 3), dtype=bool), (2, 2)),
         det('part2', np.ones((3, 3), dtype=bool), (12, 12))],
        'sam2', 'a2', 'confounder1', graph._test_ids,
        config=AssociationConfig(mask_only=True, enable_iom_dedup=True))
    assert not result.rejected_group_nodes
    assert graph.active_nodes()[0].observations[0].detection_id == 'parent'
