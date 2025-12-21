
import argparse
import sys
import logging
import subprocess
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import imageio

# Add src to path
sys.path.append("/scratch3/metzgern/random/ml-sharp/src")

from sharp.utils import io
from sharp.utils.bokeh_renderer import render_single_bokeh
from sharp.utils.gaussians import load_ply

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run autofocus benchmarks and visualization.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples to process (0 = all)")
    parser.add_argument("--skip-prediction", action="store_true", help="Skip the SHARP prediction/inference step")
    parser.add_argument("--skip-rendering", action="store_true", help="Skip the bokeh rendering step")
    parser.add_argument("--aperture-size", type=float, default=0.03, help="Aperture size for bokeh rendering")
    args = parser.parse_args()

    sample_dir = Path("/scratch3/metzgern/random/ml-sharp/sample_data")
    output_dir = Path("/scratch3/metzgern/random/ml-sharp/bokeh_output_debug")
    output_dir.mkdir(exist_ok=True, parents=True) # Ensure output dir exists

    str_sample_dir = str(sample_dir)
    str_output_dir = str(output_dir / "gaussians")
    
    # 1. Prediction Phase
    if not args.skip_prediction:
        print("Running SHARP prediction on sample_data...")
        # Note: Using absolute path to venv python executable for consistency
        venv_python = "/scratch3/metzgern/random/ml-sharp/.venv/bin/sharp"
        cmd = [
            venv_python, "predict",
            "-i", str_sample_dir,
            "-o", str_output_dir
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Prediction failed: {e}")
            return
    
    # 2. Rendering Phase
    if not args.skip_rendering:
        print("Running Autofocus Rendering...")
        gaussian_files = sorted(list(Path(str_output_dir).glob("*.ply")))
        
        if args.limit > 0:
            gaussian_files = gaussian_files[:args.limit]

        for ply_path in tqdm(gaussian_files, desc="Rendering Samples"):
            try:
                name = ply_path.stem
                # Load Gaussians
                gaussians, metadata = load_ply(ply_path)
                
                # Output path for the bokeh image
                render_out = output_dir / f"{name}_bokeh.jpg"
                debug_out = output_dir / f"{name}_bokeh_debug.jpg"
                
                # Render with Autofocus and Debug
                render_single_bokeh(
                    gaussians=gaussians,
                    metadata=metadata,
                    output_path=render_out,
                    aperture_size=args.aperture_size, 
                    num_samples=256,   
                    autofocus=True,
                    debug=True
                )
                        
            except Exception as e:
                LOGGER.error(f"Failed to render {ply_path.name}: {e}")
    else:
        # Need gaussian_files list for video generation if skipping rendering loop
        gaussian_files = sorted(list(Path(str_output_dir).glob("*.ply")))
        if args.limit > 0:
            gaussian_files = gaussian_files[:args.limit]

    # --- Create Temporal Video ---
    create_temporal_video(sample_dir, output_dir, gaussian_files)


def create_temporal_video(sample_dir: Path, output_dir: Path, gaussian_files: list):
    """Generate a temporal video summary for portrait images."""
    print("Creating temporal summary video (Portraits only)...")
    video_path = output_dir / "autofocus_summary_temporal.mp4"
    
    # Sequence: Input (2s) -> Debug (2s) -> Output (2s) -> Next Sample
    fps = 30
    duration_per_stage = 1.0 # seconds
    
    frames_buffer = []
    valid_samples = 0
    
    for ply_path in tqdm(gaussian_files, desc="Generating Video Frames"):
        name = ply_path.stem
        
        # Find input path
        input_path = None
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = sample_dir / (name + ext)
            if candidate.exists():
                input_path = candidate
                break
        
        render_out = output_dir / f"{name}_bokeh.jpg"
        debug_out = output_dir / f"{name}_bokeh_debug.jpg"
        
        if input_path and input_path.exists() and render_out.exists() and debug_out.exists():
            img_input = cv2.imread(str(input_path))
            
            if img_input is None: continue

            # Filter Portrait: Height > Width
            h, w = img_input.shape[:2]
            if h < w:
                continue
                
            img_debug = cv2.imread(str(debug_out))
            img_output = cv2.imread(str(render_out))
            
            if img_debug is None or img_output is None: continue

            valid_samples += 1
            
            # Generate frames
            frames_buffer.extend(get_stage_frames(img_input, duration_per_stage, fps))
            frames_buffer.extend(get_stage_frames(img_debug, duration_per_stage, fps))
            frames_buffer.extend(get_stage_frames(img_output, duration_per_stage, fps))

    if frames_buffer:
        print(f"Writing {len(frames_buffer)} frames for {valid_samples} samples...")
        with imageio.get_writer(video_path, fps=fps, codec='libx264') as writer:
            for frame in frames_buffer:
                writer.append_data(frame)
        print(f"Saved temporal video to {video_path}")
    else:
        print("No valid portrait samples found for video.")


def get_stage_frames(img, duration, fps):
    """Create a sequence of frames for a single stage."""
    # Target 720w x 960h (4:3 portrait)
    target_h = 960
    target_w = 720
    
    # Resize/Crop/Pad to target
    h, w = img.shape[:2]
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    return [frame_rgb] * int(duration * fps)

if __name__ == "__main__":
    main()
