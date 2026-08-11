import pytest
from sam3_vlm.evaluation.metrics import compute_count_metrics, compute_discovery_metrics
from sam3_vlm.evaluation.matching import compute_matching
from sam3_vlm.core.geometry import BoxGeometry, Box

def test_count_metrics():
    # standard
    res = compute_count_metrics(2.5, 3)
    assert res["absolute_error"] == 0.5
    assert res["signed_error"] == -0.5
    assert res["squared_error"] == 0.25
    assert res["relative_error"] == 0.5 / 3
    
    # GT = 0
    res0 = compute_count_metrics(1.0, 0)
    assert res0["absolute_error"] == 1.0
    assert res0["signed_error"] == 1.0
    assert res0["relative_error"] is None

def test_matching_and_discovery():
    preds = [
        BoxGeometry(Box(0, 0, 10, 10, "image")),
        BoxGeometry(Box(20, 20, 30, 30, "image")), # FP
    ]
    gts = [
        BoxGeometry(Box(1, 1, 9, 9, "image")), # matches 0
        BoxGeometry(Box(50, 50, 60, 60, "image")), # FN
    ]
    
    matches, unmatched_preds, unmatched_gts = compute_matching(preds, gts, iou_threshold=0.5)
    
    assert len(matches) == 1
    assert matches[0] == (0, 0)
    assert unmatched_preds == [1]
    assert unmatched_gts == [1]
    
    res = compute_discovery_metrics(matches, unmatched_preds, unmatched_gts)
    assert res["true_positives"] == 1
    assert res["false_positives"] == 1
    assert res["false_negatives"] == 1
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1"] == 0.5
