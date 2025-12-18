"""Script to create presentation videos comparing input images and bokeh outputs.

Copyright (C) 2025 Nando Metzger.
This file is part of the Sharp Bokeh Extension.
"""

import os
import glob
from pathlib import Path
import imageio.v3 as iio
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_presentation_videos(input_dir, bokeh_dir, output_dir):
    input_path = Path(input_dir)
    bokeh_path = Path(bokeh_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Get list of bokeh videos
    video_files = list(bokeh_path.glob("*.mp4"))
    
    if not video_files:
        print("No videos found in", bokeh_dir)
        return

    for video_file in tqdm(video_files, desc="Creating presentation videos"):
        stem = video_file.stem
        
        # Try to find matching input image
        image_file = input_path / f"{stem}.jpg"
        if not image_file.exists():
            # Try png potentially
            image_file = input_path / f"{stem}.png"
            
        if not image_file.exists():
            print(f"Skipping {stem}: Input image not found.")
            continue
            
        try:
            # Read input image
            raw_image = iio.imread(image_file)
            
            # Prepare initialization from first frame without loading all
            # Use iterator to stream frames
            # Default plugin usually works (ffmpeg)
            reader = iio.imiter(video_file)
            
            # Get first frame to determine dimensions
            first_frame = next(reader)
            vid_h, vid_w, _ = first_frame.shape

            # Resize static image to match video height using PIL
            # Convert numpy -> PIL
            pil_image = Image.fromarray(raw_image)
            img_w, img_h = pil_image.size
            
            scale = vid_h / img_h
            new_w = int(img_w * scale)
            
            # Ensure even width for video encoding compatibility
            if new_w % 2 != 0:
                new_w += 1
                
            pil_resized = pil_image.resize((new_w, vid_h), Image.Resampling.LANCZOS)
            resized_image = np.array(pil_resized)

            # Setup writer
            out_file = output_path / f"presentation_{stem}.mp4"
            
            # Using v2 style writer which is often more robust for simple appending
            with imageio.get_writer(out_file, fps=16, codec='libx264') as writer:
                 # Stream read - reopen generator
                 video_reader = iio.imiter(video_file)
                 for frame in video_reader:
                      combined = np.hstack((resized_image, frame))
                      # Ensure compatible types (sometimes uint8 is safer)
                      if combined.dtype != np.uint8:
                            combined = combined.astype(np.uint8)
                      writer.append_data(combined)
                      
        except Exception as e:
            print(f"Error processing {stem}: {e}")

if __name__ == "__main__":
    create_presentation_videos("sample_data", "bokeh_output", "presentation_videos")
