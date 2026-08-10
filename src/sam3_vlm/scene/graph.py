"""Scene Graph container interface and management for SAM3-VLM V4."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from sam3_vlm.scene.node import Node


@dataclass
class SceneGraph:
    """Evolving scene graph holding object hypothesis nodes."""

    nodes: Dict[str, Node] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def active_nodes(self) -> List[Node]:
        from sam3_vlm.core.types import NodeStatus
        return [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]
