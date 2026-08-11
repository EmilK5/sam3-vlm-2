"""Matplotlib-based final scene rendering (V4 Design Spec §17.3)."""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from sam3_vlm.scene.state import SceneGraph
from sam3_vlm.core.types import NodeStatus

def render_final_scene(
    graph: SceneGraph, 
    target_class: str, 
    image_path: Optional[str] = None, 
    output_path: Optional[str] = None
) -> None:
    """Renders final active nodes with IDs and target posteriors."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Load and display image if available
    if image_path and Path(image_path).exists():
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
        except Exception:
            pass
            
    for node in graph.active_nodes():
        box = node.geometry.bbox()
        posterior = node.class_belief.probabilities.get(target_class, 0.0)
        
        # Color based on posterior (0 = red, 1 = green)
        color = (1.0 - posterior, posterior, 0.0, 0.5)
        
        rect = patches.Rectangle(
            (box.xmin, box.ymin), 
            box.xmax - box.xmin, 
            box.ymax - box.ymin, 
            linewidth=2, 
            edgecolor=color, 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.text(
            box.xmin, box.ymin - 5,
            f"{node.node_id[:6]}\n{posterior:.2f}",
            color='white',
            fontsize=8,
            bbox=dict(facecolor=color, edgecolor='none', alpha=0.7)
        )
        
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
    plt.close(fig)
