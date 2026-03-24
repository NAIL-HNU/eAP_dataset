

# eAP Dataset

**The largest multi-modal dataset with event cameras for autonomous driving perception.**

<p align="center">
<a href="https://www.youtube.com/watch?v=6nuFrPViD3U">
  <img src="https://img.youtube.com/vi/6nuFrPViD3U/maxresdefault.jpg" alt="eAP Dataset Overview" width="600"/>
</a>
</p>

<p align="center">
  <a href="https://nail-hnu.github.io/eAP_dataset/">🌐 Project Page</a>
</p>

## Citation

If you use this dataset, please cite:

```bibtex
@misc{li2026eap,
  title         = {Toward Deep Representation Learning for Event-Enhanced
                   Visual Autonomous Perception: the eAP Dataset},
  author        = {Li, Jinghang and Li, Shichao and Lian, Qing
                   and Li, Peiliang and Chen, Xiaozhi and Zhou, Yi},
  year          = {2026},
  eprint        = {2603.16303},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2603.16303},
}
```

## Repository Contents

```text
release_visualizer/          # Standalone visualization tools
    ├── requirements.txt
    ├── visualize_release_sequence.py
    └── sample_release_frames.py
```

## Data Format

Each released sequence is a self-contained directory:

```text
$DATASET_ROOT/
└── <sequence_id>/
    ├── annotations.pkl
    ├── frames.pkl
    ├── events.h5
    └── rgb/
        └── *.png
```

### `annotations.pkl`

Object-level annotations. Each record contains:


| Field                             | Description                                                   |
| --------------------------------- | ------------------------------------------------------------- |
| `file_name`                       | Corresponding RGB image filename                              |
| `events_file`                     | Path to the associated HDF5 events file                       |
| `sequence_id`                     | Sequence identifier                                           |
| `instance_id`                     | Per-track instance ID                                         |
| `category`                        | Object category label                                         |
| `bbox`                            | 2D bounding box `[x, y, w, h]`                                |
| `bbox_3d`                         | 3D bounding box `[x, y, z, l, h, w, yaw]` in ego frame (Z-up) |
| `T_event_ego`                     | Transform from ego frame to event camera frame                |
| `K_event`                         | Event camera intrinsic matrix                                 |
| `velocity`                        | Object velocity in ego frame                                  |
| `ttc`                             | Time-to-collision in seconds                                  |
| `rgb_exposure_start_timestamp_us` | RGB exposure start timestamp (µs)                             |
| `rgb_exposure_end_timestamp_us`   | RGB exposure end timestamp (µs)                               |


### `frames.pkl`

Full-frame index (includes frames with no annotations). Each record contains:


| Field                             | Description                             |
| --------------------------------- | --------------------------------------- |
| `file_name`                       | RGB image filename                      |
| `events_file`                     | Path to the associated HDF5 events file |
| `sequence_id`                     | Sequence identifier                     |
| `rgb_exposure_start_timestamp_us` | RGB exposure start timestamp (µs)       |
| `rgb_exposure_end_timestamp_us`   | RGB exposure end timestamp (µs)         |


### `events.h5`

Event stream in HDF5 format. Events are accessed via a millisecond-to-index map (`ms_to_idx`) and an `events` group with fields `t` (timestamp µs), `x`, `y`, `p` (polarity). The visualizer uses a 50 ms window centered on each RGB exposure midpoint.

## Visualization

### Setup

```bash
pip install -r release_visualizer/requirements.txt
```

First, list available sequence IDs in a release root:

```bash
python release_visualizer/visualize_release_sequence.py $DATASET_ROOT --list-sequences
```

### Video visualization (`visualize_release_sequence.py`)

Render a full sequence to MP4:

```bash
python release_visualizer/visualize_release_sequence.py $DATASET_ROOT \
  --sequence-id <sequence_id> \
  --output-dir ./outputs
```

Output: `./outputs/<sequence_id>_release_visualization.mp4`  
Layout: RGB + event overlay (left panel) · bird's-eye view with TTC annotations (right panel).

Pass `$DATASET_ROOT/<sequence_id>` directly to skip `--sequence-id`:

```bash
python release_visualizer/visualize_release_sequence.py $DATASET_ROOT/<sequence_id> \
  --output-dir ./outputs
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--fps` | 10 | Output video frame rate |
| `--max-frames N` | — | Render only the first N frames |
| `--frame-step N` | 1 | Render every Nth frame |
| `--image-width` | 960 | Width of the left RGB/event panel (px) |
| `--bev-width` | 760 | Width of the BEV panel (px) |
| `--fwd-max` | 60.0 | BEV forward range (m) |
| `--lat-max` | 30.0 | BEV lateral half-range (m) |

### Frame sampling (`sample_release_frames.py`)

Export a small set of randomly sampled rendered frames (PNG) for quick spot-checks:

```bash
python release_visualizer/sample_release_frames.py $DATASET_ROOT \
  --sequence-id <sequence_id> \
  --output-dir ./release_checks \
  --samples-per-asset 5
```

Output: `./release_checks/<sequence_id>_<frame_idx>.png`

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--samples-per-asset` | 5 | Number of random frames per sequence |
| `--seed` | 0 | Random seed for reproducible sampling |
| `--image-width` | 960 | Width of the left panel (px) |
| `--bev-width` | 760 | Width of the BEV panel (px) |

> If you encounter OpenMP shared-memory errors, set thread counts explicitly:
> ```bash
> env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE \
> python release_visualizer/visualize_release_sequence.py ...
> ```

