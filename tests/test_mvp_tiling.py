from sam3_vlm.mvp.tiling import plan_adaptive_tiles


def test_sparse_scene_skips_tiling():
    plan = plan_adaptive_tiles([(100, 100, 300, 300)], 1000, 1000)
    assert not plan.trigger
    assert plan.tiles == ()
    assert plan.roi == (84, 84, 316, 316)


def test_dense_scene_selects_overlapping_tiles_near_padded_roi():
    boxes = [(100 + (i % 10) * 8, 100 + (i // 10) * 8,
              106 + (i % 10) * 8, 106 + (i // 10) * 8) for i in range(60)]
    plan = plan_adaptive_tiles(boxes, 1000, 1000)
    assert plan.trigger
    assert plan.density_score > .69
    assert plan.tile_size == 250
    assert plan.overlap == 62
    assert len(plan.tiles) > 0
    assert len(plan.tiles) < 36
    assert all(t[2] - t[0] <= 250 and t[3] - t[1] <= 250 for t in plan.tiles)
    assert len(set(plan.tiles)) == len(plan.tiles)


def test_small_image_tile_is_full_image_once():
    plan = plan_adaptive_tiles([(i, i, i + 1, i + 1) for i in range(50)], 80, 80)
    assert plan.trigger
    assert plan.tiles == ((0, 0, 80, 80),)


def test_roi_sliver_keeps_boundary_tile():
    boxes = [(251, 251, 252, 252) for _ in range(50)]
    plan = plan_adaptive_tiles(boxes, 1000, 1000)
    assert plan.trigger
    assert plan.roi == (235, 235, 268, 268)
    assert (0, 0, 250, 250) in plan.tiles


def test_vlm_roi_uses_local_density_and_tile_scale_even_without_seeds():
    roi = (300, 300, 700, 700)
    boxes = [(310 + 7 * (i % 10), 310 + 7 * (i // 10),
              312 + 7 * (i % 10), 312 + 7 * (i // 10)) for i in range(60)]
    plan = plan_adaptive_tiles(boxes, 1000, 1000, roi_override=roi)
    assert plan.trigger and plan.tile_size == 100
    assert all(roi[0] <= x1 < x2 <= roi[2] and roi[1] <= y1 < y2 <= roi[3]
               for x1, y1, x2, y2 in plan.tiles)
    empty = plan_adaptive_tiles([], 1000, 1000, roi_override=roi, force=True)
    assert empty.trigger and empty.tiles and empty.roi == roi
