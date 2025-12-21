"""Module for rendering bokeh effects on 3D Gaussians.

Copyright (C) 2025 Nando Metzger.
This file is part of the Sharp Bokeh Extension, a fork of the original SHARP repository.
"""

import logging
from pathlib import Path
from typing import List

import imageio
import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm

from sharp.utils import camera, gsplat, io
from sharp.utils.gaussians import Gaussians3D, SceneMetaData

LOGGER = logging.getLogger(__name__)


def render_single_bokeh(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    aperture_size: float,

    num_samples: int,
    autofocus: bool = False,
    debug: bool = False,
) -> None:
    """Render a single bokeh image focused at the center.
    
    Args:
        gaussians: The 3D Gaussians scene.
        metadata: Scene metadata containing camera parameters.
        output_path: Path to save the output image.
        aperture_size: Size of the synthetic aperture disk.
        num_samples: Number of samples to use for aperture accumulation.
    """
    (width, height) = metadata.resolution_px
    f_px = metadata.focal_length_px
    device = torch.device("cuda")
    gaussians = gaussians.to(device)

    # 1. Setup Canonical Camera
    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2.0, 0],
            [0, f_px, (height - 1) / 2.0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )

    # 2. Determine Focus Point (from center pixel depth)
    renderer = gsplat.GSplatRenderer(color_space="linearRGB")
    extrinsics_canonical = torch.eye(4, device=device).unsqueeze(0)
    intrinsics_canonical = intrinsics.unsqueeze(0)

    with torch.no_grad():
        canonical_output = renderer(
            gaussians,
            extrinsics=extrinsics_canonical,
            intrinsics=intrinsics_canonical,
            image_width=width,
            image_height=height,
        )
        depth_map = canonical_output.depth[0, 0] # (H, W)
        
        focus_depth = 0.0
        
        if autofocus:
            focus_depth = _run_autofocus(
                canonical_output, 
                width, 
                height, 
                output_path, 
                debug
            )
        else:
            center_y, center_x = height // 2, width // 2
            focus_depth = depth_map[center_y, center_x].item()
            LOGGER.info(f"Focus depth determined at center: {focus_depth:.4f}")

    final_image_uint8 = _accumulate_frame(
        gaussians, renderer, intrinsics, metadata, aperture_size, num_samples, focus_depth
    )
    io.save_image(final_image_uint8, output_path)
    LOGGER.info(f"Saved bokeh image to {output_path}")


def render_focus_rack_video(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    aperture_size: float,
    num_samples: int,
    num_frames: int,
) -> None:
    """Render a focus racking video.
    
    Generates a video where the focus plane moves smoothly from the nearest
    scene depth to the furthest and back.
    
    Args:
        gaussians: The 3D Gaussians scene.
        metadata: Scene metadata.
        output_path: Path to save the MP4 video.
        aperture_size: Size of the synthetic aperture.
        num_samples: Number of samples per frame.
        num_frames: Number of frames for the forward sweep.
    """
    (width, height) = metadata.resolution_px
    f_px = metadata.focal_length_px
    device = torch.device("cuda")
    gaussians = gaussians.to(device)

    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2.0, 0],
            [0, f_px, (height - 1) / 2.0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    renderer = gsplat.GSplatRenderer(color_space="linearRGB")

    # Determine depth range using a temporary camera model
    temp_cam = camera.PinholeCameraModel(
        gaussians,
        screen_extrinsics=torch.eye(4), # cpu ok
        screen_intrinsics=intrinsics.cpu(),
        screen_resolution_px=metadata.resolution_px
    )
    min_depth = temp_cam.depth_quantiles.min
    max_depth = temp_cam.depth_quantiles.max
    LOGGER.info(f"Racking focus from {min_depth:.2f}m to {max_depth:.2f}m")

    # Generate depths uniformly in inverse depth (disparity) space for natural pacing
    safe_min_depth = max(min_depth, 1e-5)
    max_disp = 1.0 / safe_min_depth
    min_disp = 1.0 / max_depth
    
    disparities = torch.linspace(max_disp, min_disp, num_frames)
    depths = (1.0 / disparities).tolist()

    frames = []
    for i, focus_depth in enumerate(tqdm(depths, desc="Rendering frames")):
        frame_uint8 = _accumulate_frame(
            gaussians, renderer, intrinsics, metadata, aperture_size, num_samples, focus_depth
        )
        frames.append(frame_uint8)

    # Ping pong loop (pad with reverse)
    frames += frames[::-1]

    # Save as MP4
    with imageio.get_writer(output_path, fps=16, codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)
            
    LOGGER.info(f"Saved bokeh video to {output_path}")


def _accumulate_frame(
    gaussians: Gaussians3D,
    renderer: gsplat.GSplatRenderer,
    intrinsics: torch.Tensor,
    metadata: SceneMetaData, 
    aperture_size: float, 
    num_samples: int, 
    focus_depth: float
) -> np.ndarray:
    """Helper to accumulate a single frame at specific focus depth using synthetic aperture array."""
    device = gaussians.mean_vectors.device
    (width, height) = metadata.resolution_px
    
    # Shift-Lens Rendering (Parallel Image Planes)
    # To avoid perspective distortion ("swirly bokeh"):
    # 1. Translate camera by 'offset' (samples aperture).
    # 2. Shift intrinsics principal point (cx, cy) to re-center the focus plane.
    #    cx_new = cx + f * offset_x / focus_depth
    
    # Base intrinsics / extrinsics
    base_extrinsics = torch.eye(4, device=device)
    base_intrinsics = intrinsics.clone() # (1, 4, 4) if batched or (4, 4)
    if base_intrinsics.ndim == 3: base_intrinsics = base_intrinsics[0]
    
    fx = base_intrinsics[0, 0]
    fy = base_intrinsics[1, 1]
    cx = base_intrinsics[0, 2]
    cy = base_intrinsics[1, 2]

    # Use Golden Ratio Spiral logic from camera module
    eye_positions = camera.create_aperture_samples(aperture_size, num_samples)
    
    accumulation_buffer = torch.zeros((3, height, width), device=device, dtype=torch.float32)

    for eye_pos in eye_positions:
        # eye_pos is (x, y, 0) in camera frame
        offset = eye_pos.to(device)
        
        # 1. Modify Extrinsics: Translate by offset
        # World-to-Camera (Extrinsics) T_new = T_translate @ T_base
        # Since T_base is identity here (canonical view is usually at identity), T_new is just translation.
        # But wait, input gaussians are already transformed to canonical view??
        # The render_single_bokeh function sets extrinsics_canonical = Identity.
        # So we just set the translation column.
        # Note: Extrinsics matrix usually maps World -> Camera.
        # If Camera moves by 'offset' in World frame, the point P_w becomes P_c = R(P_w - C_new).
        # C_new = C_old + offset.
        # P_c = P_w - offset (assuming R=I).
        # So we subtract offset from the translation part.
        
        current_extrinsics = base_extrinsics.clone()
        current_extrinsics[:3, 3] -= offset
        
        # 2. Modify Intrinsics: Shift principal point
        current_intrinsics = base_intrinsics.clone()
        
        # Shift amount = f * offset / focus_depth
        # Note: offset is (x, y, 0).
        # Careful with signs. 
        # P_c_new = P_original_cam - offset.
        # x_new = x_old - offset_x
        # u_new = f * (x_old - offset_x) / z + cx_new
        # We want u_new = u_old = f * x_old / z + cx
        # f * x_old / z - f * offset_x / z + cx_new = f * x_old / z + cx
        # cx_new = cx + f * offset_x / z
        
        shift_x = fx * offset[0] / focus_depth
        shift_y = fy * offset[1] / focus_depth
        
        current_intrinsics[0, 2] += shift_x
        current_intrinsics[1, 2] += shift_y
        
        with torch.no_grad():
            rendering_output = renderer(
                gaussians,
                extrinsics=current_extrinsics.unsqueeze(0),
                intrinsics=current_intrinsics.unsqueeze(0),
                image_width=width,
                image_height=height,
            )
            # Accumulate in Linear RGB space
            accumulation_buffer += rendering_output.color[0]

    averaged_image = accumulation_buffer / num_samples

    # Convert to sRGB at the very end if required or if output is standard 8-bit
    # We always convert to sRGB for displayable formats to avoid dark images
    from sharp.utils import color_space as cs_utils
    final_image = cs_utils.linearRGB2sRGB(averaged_image)

    return (final_image.permute(1, 2, 0).clamp(0, 1) * 255.0).to(dtype=torch.uint8).cpu().numpy()


def _run_autofocus(
    canonical_output: gsplat.RenderingOutputs,
    width: int,
    height: int,
    output_path: Path,
    debug: bool
) -> float:
    """Run the autofocus engine on a rendered view."""
    from sharp.utils import color_space as cs_utils
    from sharp.utils.autofocus import get_default_autofocus_engine
    
    LOGGER.info("Autofocus enabled. Determining focus point...")
    engine = get_default_autofocus_engine()
    
    # Extract depth and color
    depth_map = canonical_output.depth[0, 0] # (H, W)
    color_tensor = canonical_output.color[0] # (3, H, W)
    
    # Convert Linear RGB -> sRGB for detection
    color_srgb = cs_utils.linearRGB2sRGB(color_tensor.permute(1, 2, 0))
    image_np = (color_srgb.clamp(0, 1) * 255).byte().cpu().numpy()
    depth_np = depth_map.detach().cpu().numpy()
    
    focus_depth, debug_img = engine.compute_focus_depth(
        image_np, depth_np, debug=debug
    )
    
    if debug and debug_img is not None:
        debug_path = output_path.with_name(output_path.stem + "_debug.jpg")
        import cv2
        cv2.imwrite(str(debug_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        LOGGER.info(f"Saved debug autofocus output to {debug_path}")
        
    return focus_depth
