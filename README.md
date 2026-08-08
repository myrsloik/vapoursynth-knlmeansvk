# KNLMeansVk

**KNLMeansVk** is a Non-local means de-noising filter for VapourSynth, running on the
core's Vulkan device. It is a port of Khanattila's OpenCL implementation, **KNLMeansCL**,
and keeps its algorithm exactly: the GPU-friendly NL-means decomposition proposed by
Goossens et al. with per-offset patch distances, a separable box filter and symmetric
pair accumulation.

The NLMeans filter, originally proposed by Buades et al., is a very popular filter for the
removal of white Gaussian noise, due to its simplicity and excellent performance. The
strength of this algorithm is to exploit the repetitive character of the image in order to
de-noise the image, unlike conventional de-noising algorithms which typically operate in a
local neighbourhood.

Frames stay in video memory for the whole chain: the filter takes and produces GPU
resident clips, composes with every other GPU filter without crossing the PCIe bus, and a
CPU clip passed to it is uploaded automatically.

## Requirements

- VapourSynth R80 or later, with a GPU and driver that support Vulkan 1.4. Check with
  `core.vulkan_devices` from Python.
- Works on Linux, Windows and macOS (the latter through MoltenVK).

## Installation

```bash
pip install vapoursynth-knlmeansvk
```

The wheel installs the plugin into VapourSynth's Python package plugin directory.

## Usage

```python3
knlmvk.KNLMeans(clip clip[, int d=1, int a=2, int s=4, float h=1.2, string channels="AUTO", int wmode=0, float wref=1.0, clip rclip=None])
```

- **clip**: the input clip. Gray, YUV or RGB; 8-16 bit integer, half or single precision
  float. Frame properties are copied from it.
- **d**: temporal radius; frames `[n-d, n+d]` take part in the search. `d=0` is purely
  spatial.
- **a**: spatial search radius; the search window is `(2a+1)x(2a+1)x(2d+1)`.
- **s**: similarity neighbourhood (patch) radius, in `[0, 8]`; patches are
  `(2s+1)x(2s+1)`.
- **h**: filtering strength, on a 0-255 scale regardless of bit depth. Larger values
  smooth more.
- **channels**: which channels drive the similarity and are filtered.
  - `"AUTO"`: `"RGB"` for RGB clips, `"Y"` otherwise.
  - `"Y"`: luma only; chroma planes pass through untouched.
  - `"UV"`: chroma jointly, at chroma resolution; luma passes through.
  - `"YUV"`: all three jointly; requires 4:4:4.
  - `"RGB"`: all three jointly, with brightness-adaptive red/blue weighting.
- **wmode**: weighting kernel. `0` Welsch (smooth, never zero), `1`/`2`/`3` bisquare
  variants that cut off hard and therefore need a considerably larger `h`.
- **wref**: weight of the pixel being filtered, as a multiple of the largest neighbour
  weight.
- **rclip**: a reference clip driving the block matching (weights are computed on it, the
  average is taken over `clip`). Must match `clip` in format, dimensions and length.

See the upstream **[Wiki](https://github.com/Khanattila/KNLMeansCL/wiki)** for parameter
guidance; everything about `d`, `a`, `s`, `h`, `channels`, `wmode`, `wref` and `rclip`
carries over.

## Differences from KNLMeansCL

- `device_type`, `device_id`, `ocl_x`, `ocl_y` and `ocl_r` are gone: the core owns the
  Vulkan device (`core.set_vulkan_device()` selects one) and the port sizes its own
  dispatches. `info` is gone with the OpenCL platform text it printed, and
  `mode_9_to_15bits` is gone because the OpenCL image-format modes it toggled between do
  not exist here; there is one normalize-by-maximum path for every integer depth.
- The plugin namespace is `knlmvk` and the function `KNLMeans`, replacing
  `knlm.KNLMeansCL`.
- The filter runs `fmParallel` with per-frame GPU scratch instead of serializing frames
  through one command queue, so concurrent frames actually overlap.
- At the last `d` frames of a clip the OpenCL implementation read stale stack layers for
  the future temporal side. Here the temporal window shrinks symmetrically at both clip
  ends instead, mirroring its start-of-clip behaviour; every interior frame is
  numerically equivalent to the original.
- No 8192x8192 resolution limit (it came from OpenCL image dimensions).
- AviSynth support was removed together with OpenCL: the port is built on VapourSynth's
  GPU API.

## Compilation

Requires a C++20 compiler, meson, ninja, the VapourSynth headers (R80+, provided by the
`vapoursynth` Python package) and the Vulkan headers. Only the headers are needed — the
plugin does not link the Vulkan loader, because the core hands it every entry point
already resolved.

```bash
meson setup build
meson compile -C build
```

If the headers are not found automatically, pass
`-Dvapoursynth_include=/path/to/vapoursynth/include` and/or
`-Dvulkan_include=/path/to/vulkan/include`.

The compiled plugin (`libknlmvulkan.so`, `knlmvulkan.dll` or `libknlmvulkan.dylib`) lands
in the build directory; copy it to your VapourSynth plugins directory. The compute kernel
is embedded with C23 `#embed` where the compiler supports it (clang 19+, gcc 15+) and
through a meson-generated header otherwise, so MSVC builds without a separate step.

Wheels build with [uv](https://github.com/astral-sh/uv): `uv build --wheel`.

## License

This project is licensed under the GNU General Public License v3.0 or later (GPLv3+),
like the KNLMeansCL implementation it is ported from, Copyright© 2015-2020
Edoardo Brunetti.
