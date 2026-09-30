import numpy as np
import pytest

from sam3_vlm.sensing.adaptive_tiling import plan_adaptive_tiles


def dense_boxes():
    return [(100 + (i % 10) * 8, 100 + (i // 10) * 8,
             106 + (i % 10) * 8, 106 + (i // 10) * 8) for i in range(60)]


def test_sparse_and_empty_scenes_skip_tiles():
    for boxes in ([], [(100, 100, 300, 300)]):
        plan = plan_adaptive_tiles(boxes, 1000, 1000)
        assert not plan.trigger and plan.tiles == ()
        assert plan.image_region == (0, 0, 1000, 1000)


def test_dense_scene_uses_mvp_density_and_covers_entire_image():
    boxes = dense_boxes()
    plan = plan_adaptive_tiles(boxes, 1000, 1000)
    assert plan.trigger
    coverage = 60 * 36 / 1000000
    assert plan.density_score == pytest.approx(.3 * coverage + .5 + .2 * (1 - 36 / 100000))
    assert plan.tile_size == 250 and plan.overlap == 62
    assert len(plan.tiles) == 25 and len(set(plan.tiles)) == 25
    covered = np.zeros((1000, 1000), dtype=bool)
    for x1, y1, x2, y2 in plan.tiles:
        covered[y1:y2, x1:x2] = True
    assert covered.all()  # Includes areas without any Stage-1 detections.


def test_small_image_generates_full_image_only_once():
    plan = plan_adaptive_tiles([(i, i, i + 1, i + 1) for i in range(50)], 80, 80)
    assert plan.trigger and plan.tiles == ((0, 0, 80, 80),)


@pytest.mark.parametrize('width,height', [(32, 384), (4000, 100), (403, 777)])
def test_rectangular_images_are_covered_and_tiles_are_unique(width, height):
    plan = plan_adaptive_tiles([(0, 0, 1, 1)], width, height, density_threshold=0)
    covered = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in plan.tiles:
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        covered[y1:y2, x1:x2] = True
    assert covered.all() and len(plan.tiles) == len(set(plan.tiles))


@pytest.mark.parametrize('kwargs', [{'width': 0}, {'density_threshold': float('nan')},
                                    {'min_tile_size': 200, 'max_tile_size': 100}])
def test_invalid_parameters_fail(kwargs):
    params = dict(width=1000, height=1000)
    params.update(kwargs)
    with pytest.raises(ValueError):
        plan_adaptive_tiles([], **params)


def test_missed_dense_scene_triggers_on_small_candidates_below_density_cutoff():
    boxes = dense_boxes()[:14]
    plan = plan_adaptive_tiles(boxes, 1000, 1000, enable_fallback=True)
    assert plan.density_score < .69 and plan.trigger
    assert plan.trigger_reason == 'small_objects' and plan.small_candidate_count == 14
    assert plan.tile_size == 250


def test_no_seeds_get_coarse_full_image_fallback():
    plan = plan_adaptive_tiles([], 1000, 1000, enable_fallback=True)
    assert plan.trigger and plan.trigger_reason == 'uncertain_scale'
    assert plan.tile_size == 600 and len(plan.tiles) == 4
    covered = np.zeros((1000, 1000), dtype=bool)
    for x1, y1, x2, y2 in plan.tiles:
        covered[y1:y2, x1:x2] = True
    assert covered.all()


def test_fallback_does_not_tile_confident_large_sparse_scene():
    plan = plan_adaptive_tiles([(100, 100, 500, 500)], 1000, 1000, enable_fallback=True)
    assert not plan.trigger


@pytest.mark.parametrize('kwargs', [{'small_object_area_ratio': 0},
                                   {'small_object_min_count': 0},
                                   {'candidate_boxes': [(0, 0, 1001, 2)]}])
def test_invalid_fallback_settings_fail(kwargs):
    with pytest.raises(ValueError):
        plan_adaptive_tiles([], 1000, 1000, enable_fallback=True, **kwargs)
