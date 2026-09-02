"""Render 3D Gaussians on Apple Silicon, via metal-gauss.

gsplat's rasteriser is CUDA-only, so on a Mac `sharp bokeh` could not run at
all. metal-gauss (https://github.com/nandometzger/metal-gauss, MIT) is a
Metal-native differentiable rasteriser with the same job, so this is a drop-in
alternative behind the same interface: `GSplatRenderer.forward` and this take
the same arguments and return the same `RenderingOutputs`.

It is called, not vendored, and it is an optional extra:

    pip install "sharp[metal]"

Nothing here is imported unless the Metal path is actually taken, so a CUDA
install is untouched by its presence.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import torch

from sharp.utils.gaussians import Gaussians3D
from sharp.utils.gsplat import GSplatRenderer, RenderingOutputs


class MetalGaussRenderer(GSplatRenderer):
    """`GSplatRenderer` with the rasterisation done by metal-gauss.

    Subclassed rather than reimplemented so background compositing, colour
    space handling and the output contract stay in one place; only the call to
    gsplat is replaced.
    """

    def forward(
        self,
        gaussians: Gaussians3D,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> RenderingOutputs:
        """Predict images from gaussians. See `GSplatRenderer.forward`."""
        try:
            from metal_gauss import render as metal_render
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "The Metal renderer needs metal-gauss. Install it with "
                '`pip install "sharp[metal]"`, or run on CUDA.'
            ) from exc

        from sharp.utils import color_space as cs_utils

        outputs_list: list[RenderingOutputs] = []
        for ib in range(len(gaussians.mean_vectors)):
            means = gaussians.mean_vectors[ib]
            colors = gaussians.colors[ib]
            # metal-gauss reads K and the view matrix on the HOST; passing MPS
            # tensors there makes every call drain the queue.
            viewmat = extrinsics[ib].detach().cpu()
            K = intrinsics[ib, :3, :3].detach().cpu()

            def _render(color_field: torch.Tensor):
                # `sh` is ignored when `colors` is given, but it is positional.
                rgb, alpha, _ = metal_render(
                    means, gaussians.quaternions[ib], gaussians.singular_values[ib],
                    gaussians.opacities[ib], color_field.unsqueeze(1),
                    K, viewmat, image_width, image_height,
                    backend="metal", colors=color_field,
                    background=(0.0, 0.0, 0.0),
                )
                return rgb, alpha

            rendered, alpha = _render(colors)

            # Depth the same way gsplat's "RGB+D" produces it: composite the
            # per-splat camera-space z exactly like a colour, then normalise by
            # alpha. metal-gauss returns compositing statistics but no depth
            # buffer, so it costs a second pass.
            z = (means @ viewmat[:3, :3].T.to(means.device)
                 + viewmat[:3, 3].to(means.device))[:, 2:3]
            depth_unnormalized, _ = _render(z.expand(-1, 3))

            rendered_color = rendered.permute(2, 0, 1)[None]
            rendered_alpha = alpha.reshape(1, 1, image_height, image_width)
            rendered_depth = depth_unnormalized.permute(2, 0, 1)[None][:, :1]

            rendered_color = self.compose_with_background(
                rendered_color, rendered_alpha, self.background_color
            )
            if self.color_space == "sRGB":
                rendered_color = cs_utils.linearRGB2sRGB(rendered_color)
            elif self.color_space != "linearRGB":
                raise ValueError("Unsupported ColorSpace type.")

            outputs_list.append(RenderingOutputs(
                color=rendered_color,
                depth=rendered_depth / torch.clip(rendered_alpha, min=1e-8),
                alpha=rendered_alpha,
            ))

        return RenderingOutputs(
            color=torch.cat([o.color for o in outputs_list], dim=0).contiguous(),
            depth=torch.cat([o.depth for o in outputs_list], dim=0).contiguous(),
            alpha=torch.cat([o.alpha for o in outputs_list], dim=0).contiguous(),
        )


def default_device() -> torch.device:
    """The device the chosen renderer wants its Gaussians on."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError(
        "Rendering needs CUDA (gsplat) or Apple Silicon (metal-gauss); found neither."
    )


def default_renderer(**kwargs) -> GSplatRenderer:
    """The renderer this machine can actually run.

    CUDA keeps gsplat, which is what the paper's numbers were produced with.
    Apple Silicon gets metal-gauss. Neither available is still an error, since
    silently falling back to something slow and different would be worse than
    saying so.
    """
    if torch.cuda.is_available():
        return GSplatRenderer(**kwargs)
    if torch.backends.mps.is_available():
        return MetalGaussRenderer(**kwargs)
    raise RuntimeError(
        "Rendering needs CUDA (gsplat) or Apple Silicon (metal-gauss); found neither."
    )
