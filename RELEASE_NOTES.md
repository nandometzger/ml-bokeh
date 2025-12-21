# Sharp Monocular View Synthesis (Bokeh Extension) - Release Notes

## Version 1.1

This release introduces major improvements to the bokeh rendering quality and adds intelligent autofocus capabilities.

### 🌟 New Features

#### 1. Smart Autofocus with GroundingDINO
We have integrated [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to enable zero-shot object detection for automated focus point selection.

*   **Intelligent Subject Detection**: The system scans the scene for subjects using valid text prompts.
*   **Priority Logic**: Focus is determined based on a strict priority hierarchy:
    1.  **Eyes**: Ensures perfect focus for portraits.
    2.  **Face**: Fallback for portraits if eyes are not distinct.
    3.  **Person**: Captures full body subjects.
    4.  **Animals/Objects**: Fallback for general scenes.
*   **Depth-Aware Selection**: If multiple subjects of the same priority are found, the system intelligently selects the one **closest to the camera**.
*   **Usage**: 
    ```bash
    sharp bokeh ... --autofocus --debug
    ```

#### 2. Shift-Lens Bokeh Rendering
We have completely rewritten the bokeh rendering engine to eliminate perspective distortion/swirling artifacts at the image edges.

*   **The Problem**: The previous implementation simulated the aperture by maintaining a fixed camera position and rotating the camera to look at the focus point. This caused the sensor plane to tilt relative to the scene, introducing non-physical perspective distortion ("swirly bokeh") especially visible in the corners of wide-aperture renders.
*   **The Solution (Shift-Lens)**: We now implement a physically accurate "Shift-Lens" model (simulating a view camera or tilt-shift lens movement):
    *   **Camera Translation**: The camera center physically moves to sample points on the aperture disk.
    *   **Sensor Shift**: Instead of rotating, the camera orientation remains fixed (parallel sensor plane). The image is re-centered by shifting the principal point (cx, cy) of the intrinsics matrix.
*   **Result**: This technique ensures that the image planes of all aperture samples are perfectly parallel, resulting in a physically correct, distortion-free depth-of-field effect that mimics high-quality optical lenses.

### 🛠️ Improvements
*   **Autofocus Debug Mode**: New visualization that draws bounding boxes around detected subjects, displays confidence scores and depth values, and highlights the selected focus target in neon green.
*   **Linear-to-sRGB Correction**: Fixed an issue where rendered outputs were too dark by correctly converting accumulated linear light values to sRGB before saving.


### 📦 Dependencies
*   Added `groundingdino` and its dependencies.
*   Weights are automatically downloaded to `weights/` on first run.

---

## Version 1.0

The initial release of the **Sharp Bokeh Extension**, bringing cinematic depth-of-field rendering to the SHARP monocular view synthesis pipeline.

### Core Features
*   **Realistic Bokeh Rendering**: Render 3D Gaussian Splats with high-quality, synthetic depth-of-field effects.
*   **Synthetic Aperture**: Simulates a physical camera aperture using a Golden Ratio Spiral sampling pattern for smooth, artifact-free blur.
*   **Customizable Optics**:
    *   Control `aperture_size` to adjust the strength of the blur.
    *   Adjust `num_samples` for higher quality rendering.
*   **Focus Racking Videos**: Automatically generate "rack focus" videos that sweep the focal plane from the nearest to the furthest object in the scene.
*   **SHARP Integration**: Built directly on top of the SHARP codebase, leveraging its scene reconstruction capabilities.

### Usage
```bash
# Render a single image with bokeh
sharp bokeh --input_path scene.jpeg --output_path result.jpg --aperture 0.03

# Generate a focus rack video
sharp bokeh --input_path scene.jpeg --output_path result.mp4 --video
```

---
*Maintained by Nando Metzger*
