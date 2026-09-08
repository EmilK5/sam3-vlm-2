"""Small, observational summaries of changes in target probability."""

from collections import Counter


def confidence_step(graph, previous, target_class="target", action=None, observation=None):
    nodes = {n.node_id: n for n in graph.active_nodes()}
    current = {key: n.class_belief.probabilities.get(target_class, 0.0) for key, n in nodes.items()}
    added = current.keys() - previous.keys()
    retained = current.keys() & previous.keys()
    removed = previous.keys() - current.keys()
    relations = Counter()
    changes = []
    for key in sorted(retained):
        node = nodes[key]
        refs = [o for o in node.observations if action is not None and o.action_id == action.action_id]
        latest = refs[-1] if refs else None
        relation = latest.relation.value if latest else "NO_OBSERVATION"
        relations[relation] += 1
        changes.append({
            "node_id": key, "before": previous[key], "after": current[key],
            "delta": current[key] - previous[key], "relation": relation,
            "sensor_score": latest.score if latest else None,
        })
    row = {
        "stage": "bootstrap" if action is None else "sensing",
        "action_id": action.action_id if action else None,
        "prompt": action.prompt if action else None,
        "family": action.family.value if action else None,
        "threshold": action.threshold if action else None,
        "positive_exemplar_count": len(action.positive_exemplar_ids) if action else None,
        "raw_detections": len(observation.detections) if observation else None,
        "node_count": len(current), "new_nodes": len(added),
        "raw_soft_count": sum(current.values()),
        "previous_raw_soft_count": sum(previous.values()),
        "new_node_target_mass": sum(current[key] for key in added),
        "existing_node_target_mass_change": sum(current[key] - previous[key] for key in retained),
        "removed_node_target_mass": sum(previous[key] for key in removed),
        "nodes_below_half": sum(p < 0.5 for p in current.values()),
        "existing_node_observation_relations": dict(relations),
        "largest_gains": sorted((r for r in changes if r["delta"] > 0), key=lambda r: (-r["delta"], r["node_id"]))[:5],
        "largest_losses": sorted((r for r in changes if r["delta"] < 0), key=lambda r: (r["delta"], r["node_id"]))[:5],
    }
    return row, current
