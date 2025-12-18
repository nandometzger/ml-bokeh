"""Contains `sharp bokeh` CLI implementation.

Copyright (C) 2025 Nando Metzger.
This file is part of the Sharp Bokeh Extension, a fork of the original SHARP repository.
For licensing see accompanying LICENSE file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import torch
import torch.utils.data
from tqdm import tqdm

from sharp.utils.gaussians import Gaussians3D, SceneMetaData, load_ply
from sharp.utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the ply or a list of plys.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the rendered images.",
    required=True,
)
@click.option(
    "--aperture-size",
    type=float,
    default=0.01,
    help="Size of the synthetic aperture.",
)
@click.option(
    "--num-samples",
    type=int,
    default=128,
    help="Number of samples for the synthetic aperture.",
)
@click.option("--video", "make_video", is_flag=True, help="Render a video with focus racking.")
@click.option(
    "--num-frames", 
    type=int, 
    default=48, 
    help="Number of frames for video generation."
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def bokeh_cli(
    input_path: Path,
    output_path: Path,
    aperture_size: float,
    num_samples: int,
    verbose: bool,
    make_video: bool,
    num_frames: int,
):
    """Render Bokeh images from input Gaussians.
    
    Can render single images focused at the center, or focus-racking videos.
    Uses synthetic aperture rendering.
    """
    from sharp.utils import bokeh_renderer

    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if not torch.cuda.is_available():
        LOGGER.error("Rendering requires CUDA.")
        exit(1)

    output_path.mkdir(exist_ok=True, parents=True)

    if input_path.suffix == ".ply":
        scene_paths = [input_path]
    elif input_path.is_dir():
        scene_paths = list(input_path.glob("*.ply"))
    else:
        LOGGER.error("Input path must be either directory or single PLY file.")
        exit(1)

    for scene_path in scene_paths:
        LOGGER.info("Rendering bokeh for %s", scene_path)
        gaussians, metadata = load_ply(scene_path)
        
        if make_video:
            filename = (output_path / scene_path.stem).with_suffix(".mp4")
            bokeh_renderer.render_focus_rack_video(
                gaussians=gaussians,
                metadata=metadata,
                output_path=filename,
                aperture_size=aperture_size,
                num_samples=num_samples,
                num_frames=num_frames,
            )
        else:
            filename = (output_path / scene_path.stem).with_suffix(".png")
            bokeh_renderer.render_single_bokeh(
                gaussians=gaussians,
                metadata=metadata,
                output_path=filename,
                aperture_size=aperture_size,
                num_samples=num_samples,
            )


