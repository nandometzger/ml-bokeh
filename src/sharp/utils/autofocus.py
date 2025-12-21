"""Autofocus engine using Grounding DINO.

Copyright (C) 2025 Nando Metzger.
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image

# Add GroundingDINO to system path dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[3]
GROUNDING_DINO_ROOT = PROJECT_ROOT / "GroundingDINO"

if str(GROUNDING_DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(GROUNDING_DINO_ROOT))

try:
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import GroundingDINO: {e}")
    # We will handle the missing module gracefully in the class usage if needed, 
    # but for now we assume it works or will work after proper setup.

LOGGER = logging.getLogger(__name__)


class AutofocusEngine:
    """Handles autofocus logic using GroundingDINO detection and depth maps."""

    # Priority list: Lower index = Higher priority
    PRIORITY_CATEGORIES = [
        "eye",
        "face",
        "person",
        "animal",
        "insect",
        "vehicle",
        "object"
    ]

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        """Initialize the autofocus engine.
        
        Args:
            config_path: Path to GroundingDINO config file.
            checkpoint_path: Path to model weights.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.model = load_model(config_path, checkpoint_path)
        # load_model inside GroundingDINO usually puts it on cpu or device specified in args?
        # Actually load_model doesn't take device arg, it uses torch.load map_location='cpu' 
        # and then we probably need to move it.
        # But predict() moves inputs to device of model.
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        # Pre-compile the text prompt
        # We use a single prompt with all categories separated by dots
        self.text_prompt = " . ".join(self.PRIORITY_CATEGORIES)

    def compute_focus_depth(
        self,
        image_rgb: np.ndarray,
        depth_map: np.ndarray,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        border_margin: float = 0.15,
        debug: bool = False
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Compute the optimal focus depth.

        Args:
            image_rgb: Input image (H, W, 3) in RGB format (0-255, uint8).
            depth_map: Depth map (H, W) corresponding to the image.
            box_threshold: Confidence threshold for bounding boxes.
            text_threshold: Confidence threshold for text matching.
            border_margin: Fraction of image size (0.0-0.5) to ignore at borders. 
                           Detections with centers in this margin are ignored.
            debug: If True, returns an annotated debug image.

        Returns:
            Tuple containing:
            - focus_depth: The selected depth value.
            - debug_image: Annotated image (if debug=True), else None.
        """
        # 1. Prepare Image
        image_pil = Image.fromarray(image_rgb)
        
        # Transform for GroundingDINO
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image_pil, None)

        # 2. Run Inference
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=self.text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )

        # prediction returns:
        # boxes: (N, 4) unnormalized [cx, cy, w, h] (0-1 range)
        # logits: (N) confidence scores
        # phrases: list of strings

        # 3. Process Detections
        candidates = []
        h, w = depth_map.shape

        # Convert boxes to pixel coordinates (xyxy)
        # boxes are (cx, cy, w, h) normalized
        boxes_xyxy = self._box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h], device=boxes.device)
        boxes_xyxy = boxes_xyxy.cpu().numpy()
        
        for i in range(len(phrases)):
            phrase = phrases[i]
            score = logits[i].item()
            
            # Border Check: Skip if center is within the margin
            norm_cx = boxes[i, 0].item()
            norm_cy = boxes[i, 1].item()
            
            if (norm_cx < border_margin or norm_cx > (1.0 - border_margin) or
                norm_cy < border_margin or norm_cy > (1.0 - border_margin)):
                # LOGGER.debug(f"Skipping border detection: {phrase} at ({norm_cx:.2f}, {norm_cy:.2f})")
                continue

            box = boxes_xyxy[i]
            
            # Determine priority
            priority = self._get_priority(phrase)
            
            # Compute depth for this box
            # Extract depth roi
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            depth_roi = depth_map[y1:y2, x1:x2]
            # Use median depth to be robust against background pixels
            box_depth = np.median(depth_roi)
            
            candidates.append({
                "index": i,
                "phrase": phrase,
                "score": score,
                "box": box, # xyxy
                "priority": priority,
                "depth": box_depth
            })

        # 4. Selection Logic
        selected_candidate = None
        focus_depth = float(depth_map[h // 2, w // 2]) # Fallback: Center depth

        if candidates:
            # Sort by Priority (asc), then Depth (asc -> closer is usually smaller value?)
            # Assuming depth map stores metric distance. Closer objects have smaller depth.
            candidates.sort(key=lambda x: (x["priority"], x["depth"]))
            
            selected_candidate = candidates[0]
            focus_depth = selected_candidate["depth"]
            LOGGER.info(f"Autofocus selected: '{selected_candidate['phrase']}' (Priority {selected_candidate['priority']}) at depth {focus_depth:.4f}")
        else:
            LOGGER.warning("Autofocus found no targets. Falling back to center focus.")

        # 5. Debug Visualization
        debug_image = None
        if debug:
            debug_image = self._create_debug_image(image_rgb, candidates, selected_candidate, focus_depth)

        return focus_depth, debug_image

    def _get_priority(self, phrase: str) -> int:
        """Map phrase to priority index (lower is better)."""
        phrase = phrase.lower()
        for idx, category in enumerate(self.PRIORITY_CATEGORIES):
            if category in phrase:
                return idx
        return len(self.PRIORITY_CATEGORIES) # Lowest priority

    def _box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def _create_debug_image(
        self, 
        image: np.ndarray, 
        candidates: List[Dict], 
        selected: Optional[Dict], 
        focus_depth: float
    ) -> np.ndarray:
        """Draw bounding boxes and annotations."""
        # Work on a copy
        vis_img = image.copy()
        h, w = vis_img.shape[:2]
        
        # Scale visualization elements based on image size
        # Base on 1000px height
        scale_factor = max(h, w) / 1000.0
        thickness = max(1, int(2 * scale_factor))
        font_scale = 0.5 * scale_factor
        font_thick = max(1, int(1 * scale_factor))
        
        # Draw all candidates
        for c in candidates:
            box = c["box"].astype(int)
            is_selected = (selected and c == selected)
            
            # Colors (BGR)
            if is_selected:
                color = (0, 255, 0)     # Neon Green
                border_thick = thickness * 2
            else:
                color = (0, 165, 255)   # Orange
                border_thick = thickness
            
            # Draw Box
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), color, border_thick)
            
            # Prepare Label
            label = f"{c['phrase']} ({c['score']:.2f}) d={c['depth']:.2f}"
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            
            # Drawn filled background for text
            cv2.rectangle(vis_img, 
                          (box[0], box[1] - text_h - 10), 
                          (box[0] + text_w + 10, box[1]), 
                          color, -1) # Filled
            
            # Draw Text (Black or White regarding contrast, here Black on bright colors)
            cv2.putText(vis_img, label, (box[0] + 5, box[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick)

        # Draw selected info on top
        status_text = f"FOCUS: {focus_depth:.2f}m"
        if selected:
            status_text += f" | Start: {selected['phrase'].upper()}"
        else:
            status_text += " | Fallback: CENTER"
            
        # Top-left info box
        (st_w, st_h), st_base = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5, font_thick * 2)
        cv2.rectangle(vis_img, (10, 10), (10 + st_w + 20, 10 + st_h + 20), (0, 0, 0), -1)
        cv2.putText(vis_img, status_text, (20, 10 + st_h + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5, (0, 255, 255), font_thick * 2)
        
        return vis_img

def get_default_autofocus_engine() -> AutofocusEngine:
    """Create an AutofocusEngine with default paths."""
    config_path = str(GROUNDING_DINO_ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    # Weights are in project_root/weights
    weights_path = PROJECT_ROOT / "weights/groundingdino_swint_ogc.pth"
    
    if not weights_path.exists():
        LOGGER.info(f"Downloading GroundingDINO weights to {weights_path}...")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        try:
            torch.hub.download_url_to_file(url, str(weights_path))
        except Exception as e:
            raise RuntimeError(f"Failed to download weights from {url}: {e}")
        
    return AutofocusEngine(config_path, str(weights_path))

