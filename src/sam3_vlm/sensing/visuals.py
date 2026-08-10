"""Visual asset extraction and rendering utilities (V4 Design Spec §5.3 / §6.1)."""

import cv2
import numpy as np
from pathlib import Path
from typing import Any, List, Optional
from sam3_vlm.core.geometry import Box


def to_numpy_image(image: Any) -> Optional[np.ndarray]:
    """Convert a generic image object (PIL or numpy) to an OpenCV BGR numpy array."""
    if image is None:
        return None
    
    if isinstance(image, np.ndarray):
        # Assume it's already an image array, ensure it's uint8
        if image.dtype != np.uint8:
            # simple normalization if it's float [0, 1]
            if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # If RGB, convert to BGR for cv2 saving if needed, but assuming BGR input for cv2 compatibility
        # Actually it's safer to keep the format as is and rely on cv2.imwrite which expects BGR.
        return image
        
    # Check for PIL image
    if hasattr(image, "convert") and hasattr(image, "size"):
        img = image.convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
    # Check for string/Path (load from disk)
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        return img
        
    return None


def crop_image_region(image: Any, box: Box) -> Optional[np.ndarray]:
    """Extract a cropped region from the image based on the box coordinates."""
    img_array = to_numpy_image(image)
    if img_array is None:
        return None

    h, w = img_array.shape[:2]
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(w, int(box.x2)), min(h, int(box.y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    return img_array[y1:y2, x1:x2]


def save_image(image: np.ndarray, path: str) -> bool:
    """Save an image to disk, creating parent directories if needed."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, image)
        return True
    except Exception:
        return False


def render_contact_sheet(crops: List[np.ndarray], path: str, grid_cols: int = 4, target_size: int = 256) -> bool:
    """Stitch multiple crop arrays into a single contact sheet grid and save it."""
    if not crops:
        return False
        
    resized_crops = []
    for crop in crops:
        if crop is None or crop.size == 0:
            resized = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        else:
            resized = cv2.resize(crop, (target_size, target_size))
            
        # Ensure 3 channels
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)
            
        resized_crops.append(resized)
        
    num_crops = len(resized_crops)
    grid_rows = (num_crops + grid_cols - 1) // grid_cols
    if grid_rows == 0:
        return False
        
    total_slots = grid_rows * grid_cols
    blank_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    while len(resized_crops) < total_slots:
        resized_crops.append(blank_image)
        
    rows = []
    for r in range(grid_rows):
        row_crops = resized_crops[r * grid_cols : (r + 1) * grid_cols]
        rows.append(cv2.hconcat(row_crops))
        
    contact_sheet = cv2.vconcat(rows)
    return save_image(contact_sheet, path)
