"""Auditable gate for proposed negative queries; no fixed category allowlist."""

from sam3_vlm.sensing.prompts import singularize_prompt


def confounder_rejection(label, target, relationship, reason):
    if not isinstance(target, str) or not target.strip():
        return "Target counting unit is missing"
    if relationship != "distinct_object":
        return f"Confounder relationship is {relationship or 'unassessed'}; distinct_object is required"
    if not isinstance(reason, str) or not reason.strip():
        return "Distinct-object assessment needs a reason"
    target_words = singularize_prompt(target).lower().split()
    label_words = singularize_prompt(label).lower().split()
    if target_words[-1] in label_words:
        return "Negative query contains the target noun; it may name its subtype, part or container"
    return None
