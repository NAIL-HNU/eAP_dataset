

# eAP Dataset

**The largest multi-modal dataset with event cameras for autonomous driving perception.**

![banner](assets/imgs/banner.png)

[![Watch on YouTube](https://img.youtube.com/vi/6nuFrPViD3U/maxresdefault.jpg)](https://www.youtube.com/watch?v=6nuFrPViD3U)

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

First, list available sequences:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE \
python release_visualizer/visualize_release_sequence.py $DATASET_ROOT --list-sequences
```

Then render a sequence to MP4:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE \
python release_visualizer/visualize_release_sequence.py $DATASET_ROOT \
  --sequence-id <sequence_id> \
  --output-dir ./outputs
```

The output is `./outputs/<sequence_id>_release_visualization.mp4`. The video shows RGB + event rendering (left) and a bird's-eye view with TTC annotations (right).

> If you encounter OpenMP shared-memory errors, the `env OMP_NUM_THREADS=1 ...` prefix is the fix.


