"""Scene Graph container interface, lifecycle management, and serialization (V4 Design Spec §3.2 / §16.1 / §34.1)."""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional
from sam3_vlm.core.types import NodeStatus
from sam3_vlm.scene.node import Node

SCHEMA_VERSION = 1


@dataclass
class SceneGraph:
    """Evolving scene graph holding object hypothesis nodes."""

    nodes: Dict[str, Node] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        """Add a node hypothesis to the graph. Explodes loudly on duplicate persistent node ID (Spec §3.2)."""
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate persistent node ID: '{node.node_id}' already exists in SceneGraph.")
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[Node]:
        """Lookup node by ID."""
        return self.nodes.get(node_id)

    def active_nodes(self) -> List[Node]:
        """Return all nodes with ACTIVE status."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]

    def resolve_node(self, node_id: str) -> None:
        """Mark a node hypothesis as RESOLVED."""
        node = self.get_node(node_id)
        if node:
            node.status = NodeStatus.RESOLVED

    def reject_node(self, node_id: str, reason: str = "") -> None:
        """Mark a node hypothesis as REJECTED."""
        node = self.get_node(node_id)
        if node:
            node.status = NodeStatus.REJECTED

    def merge_nodes(self, primary_id: str, secondary_ids: List[str]) -> Node:
        """Merge secondary node hypotheses into primary node (Spec §25.3).

        Secondary nodes are marked REJECTED, their observations and support count
        are accumulated into the primary node, and their IDs are appended to merged_from.
        """
        primary = self.get_node(primary_id)
        if not primary:
            raise KeyError(f"Primary node ID '{primary_id}' not found in graph.")

        for sec_id in secondary_ids:
            secondary = self.get_node(sec_id)
            if secondary and secondary.status == NodeStatus.ACTIVE:
                secondary.status = NodeStatus.REJECTED
                # Accumulate lineage and observations
                if sec_id not in primary.merged_from:
                    primary.merged_from.append(sec_id)
                primary.observations.extend(secondary.observations)
                primary.diagnostics.support_count += secondary.diagnostics.support_count
                primary.diagnostics.merge_risk = max(
                    primary.diagnostics.merge_risk, 0.5
                )

        return primary

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scene graph state to a dictionary with schema_version."""
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneGraph":
        """Deserialize scene graph state from a dictionary with schema validation."""
        version = data.get("schema_version")
        if version is not None and version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported graph schema_version {version}, expected {SCHEMA_VERSION}"
            )

        graph = cls()
        nodes_dict = data.get("nodes", {})
        for node_id, node_data in nodes_dict.items():
            graph.add_node(Node.from_dict(node_data))
        return graph

    def to_json(self, indent: int = 2) -> str:
        """Serialize graph state to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "SceneGraph":
        """Deserialize graph state from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
