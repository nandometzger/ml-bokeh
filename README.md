# Sharp Monocular View Synthesis (Bokeh Extension)

<!-- [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685) -->

> **Note:** This repository is an unofficial fork of the original [SHARP](https://github.com/apple/ml-sharp) codebase. It is maintained by **[Nando Metzger](https://nandometzger.github.io)** and extends the original work with **Bokeh Simulation** features.

<!-- This software project accompanies the research paper: _Sharp Monocular View Synthesis in Less Than a Second_ -->
<!-- by _Lars Mescheder, et al._. -->

<!-- ![](data/teaser.jpg) -->
![](data/bokeh_teaser.gif)

_Figure: Real-time focus racking simulation (Left: Input Image, Right: Generated Bokeh Video)_


<!-- We present SHARP, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. -->

## Bokeh Simulation Extension

This fork adds a **Bokeh Simulator** that allows for physically-based depth-of-field rendering from the inferred 3D Gaussians.

Key features include:
*   **Synthetic Aperture**: Simulates a real camera with a configurable aperture size.
*   **Golden Ratio Spiral Sampling**: Uses fibonacci spiral sampling patterns to ensure artifact-free, creamy bokeh even with fewer samples.
*   **Focus Racking**: Generate cinematic videos where the focus transitions smoothly ("racks") from foreground to background.
*   **Linear Light Accumulation**: Correctly accumulates light in linear RGB space for realistic blending.


## Upcoming Features

We plan to continuously improve the Bokeh simulator. Upcoming features include:


*   [x] **Smart Auto-Focus**: Integration of subject and eye detection to automatically determine the most aesthetically pleasing focus depth.
*   [ ] **Custom Aperture Shapes**: Support for non-circular apertures (e.g., polygonal, heart-shaped) for creative bokeh/flare effects.
*   [ ] **Refocusing**: Let the user input an already focused video and refocus it to a different depth.

## Getting started

We recommend to first create a python environment:

```
conda create -n sharp python=3.13
```

Afterwards, you can install the project using

```
pip install -r requirements.txt
```

To test the installation, run

```
sharp --help
```

## Using the CLI

To run prediction:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

The model checkpoint will be downloaded automatically on first run and cached locally at `~/.cache/torch/hub/checkpoints/`.

### Rendering Bokeh (New Feature)

You can render realistic depth-of-field images or videos using the new `bokeh` command.

**Single Image (Center Focus):**
Render a single image where the focus is set to the center of the scene.

```bash
sharp bokeh -i /path/to/input/gaussians -o /path/to/output/bokeh --aperture-size 0.02 --num-samples 128
```

**Focus Racking Video:**
Generate a video that sweeps focus from the nearest object to infinity.

```bash
sharp bokeh -i /path/to/input/gaussians -o /path/to/output/bokeh --video --num-frames 48 --aperture-size 0.02 --num-samples 128
```

**Autofocus Mode (New Feature):**
Automatically detect subjects (Eyes > Persons > Objects) and focus on the one closest to the camera.

```bash
sharp bokeh -i /path/to/input/gaussians -o /path/to/output/bokeh --autofocus --debug
```

**Parameters:**
*   `--aperture-size`: Diameter of the virtual aperture (default: 0.01). Larger = more blur.
*   `--num-samples`: Number of views to accumulate per frame (default: 128). Higher = smoother bokeh but slower.
*   `--video`: Enable video generation mode.
*   `--autofocus`: Enable smart autofocus using GroundingDINO.
*   `--debug`: Output a debug image alongside the render showing detection boxes and the selected focus point.

## Citation

If you find the **Bokeh Extension** useful, please cite this repository:

```bibtex
@misc{SharpBokeh2025,
  title  = {Sharp Monocular View Synthesis (Bokeh Extension)},
  author = {Metzger, Nando},
  year   = {2025},
  url    = {https://github.com/metzgern/ml-sharp},
}
```

If you use the core **SHARP** method, please cite the original paper:

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama\"{e}l Delaunoy and Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```

## Acknowledgements

This project is a fork of the original [SHARP](https://github.com/apple/ml-sharp) repository developed by **Apple**. We thank the original authors for releasing their code and models.

The original codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.

