"""Optional E stopping policy, based only on executed Qwen discoveries."""


def zero_gain_streak(trace, qwen_discovery_ids):
    streak = 0
    for step in trace:
        if step.get("action_id") not in qwen_discovery_ids:
            continue
        streak = streak + 1 if step.get("new_nodes", 0) == 0 else 0
    return streak
