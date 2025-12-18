
import os
import glob
from pathlib import Path
import imageio.v3 as iio
import imageio
import numpy as np
import cv2
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
            reader = iio.imiter(video_file)
            
            # Get first frame to determine dimensions
            first_frame = next(reader)
            vid_h, vid_w, _ = first_frame.shape

            # Resize static image to match video height
            img_h, img_w, _ = raw_image.shape
            scale = vid_h / img_h
            new_w = int(img_w * scale)
            
            # Ensure even width for video encoding compatibility
            if new_w % 2 != 0:
                new_w += 1
                
            resized_image = cv2.resize(raw_image, (new_w, vid_h), interpolation=cv2.INTER_AREA)

            # Setup writer
            out_file = output_path / f"presentation_{stem}.mp4"
            
            # Using v2 style writer which is often more robust for simple appending
            with imageio.get_writer(out_file, fps=16, codec='libx264') as writer:
                 # Stream read
                 video_reader = iio.imiter(video_file, plugin="pyav")
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
