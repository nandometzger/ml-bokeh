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
    
    lookat_point = (0.0, 0.0, focus_depth)
    camera_model = camera.PinholeCameraModel(
        gaussians,
        screen_extrinsics=torch.eye(4, device=device),
        screen_intrinsics=intrinsics,
        screen_resolution_px=metadata.resolution_px,
        lookat_point=lookat_point,
        lookat_mode="point",
    )

    # Use Golden Ratio Spiral logic from camera module
    eye_positions = camera.create_aperture_samples(aperture_size, num_samples)
    
    accumulation_buffer = torch.zeros((3, height, width), device=device, dtype=torch.float32)

    for eye_pos in eye_positions:
        camera_info = camera_model.compute(eye_pos.to(device))
        
        with torch.no_grad():
            rendering_output = renderer(
                gaussians,
                extrinsics=camera_info.extrinsics[None].to(device),
                intrinsics=camera_info.intrinsics[None].to(device),
                image_width=camera_info.width,
                image_height=camera_info.height,
            )
            # Accumulate in Linear RGB space
            accumulation_buffer += rendering_output.color[0]

    averaged_image = accumulation_buffer / num_samples

    # Convert to sRGB at the very end if required
    if metadata.color_space == "sRGB":
        from sharp.utils import color_space as cs_utils
        final_image = cs_utils.linearRGB2sRGB(averaged_image)
    else:
        final_image = averaged_image

    return (final_image.permute(1, 2, 0).clamp(0, 1) * 255.0).to(dtype=torch.uint8).cpu().numpy()
