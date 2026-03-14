"""Compare two bev-anno pkl files (format: {zyt_name: [obj, ...]}).

Checks:
  1. Frame coverage  — which zyt_names are missing in either file
  2. Per-frame object count diff
  3. Per-object field diff  — matched by tracking_id within each frame

Usage:
    python tools/compare_pkl.py <pkl_a> <pkl_b> [--max-frames N] [--verbose]
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

# Fields to compare and how to compare them
ARRAY_FIELDS = ['lidar_3d', 'lidar_3d_opt', 'speed_ego_10x', 'mot_speed_ego_10x']
SCALAR_FIELDS = ['cls_name', 'opt_state', 'traj_opt_state', 'mot_tracking_type', 'mot_speed_available']
ATOL = 1e-4  # tolerance for float arrays


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def arr_eq(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return np.allclose(np.array(a, dtype=float), np.array(b, dtype=float), atol=ATOL)
    except Exception:
        return str(a) == str(b)


def compare_obj(obj_a, obj_b):
    diffs = []
    for f in SCALAR_FIELDS:
        va, vb = obj_a.get(f), obj_b.get(f)
        if va != vb:
            diffs.append(f'{f}: {va!r} vs {vb!r}')
    for f in ARRAY_FIELDS:
        va, vb = obj_a.get(f), obj_b.get(f)
        if not arr_eq(va, vb):
            if va is None or vb is None:
                diffs.append(f'{f}: {va} vs {vb}')
            else:
                delta = np.max(np.abs(np.array(va, dtype=float) - np.array(vb, dtype=float)))
                diffs.append(f'{f}: max_delta={delta:.6f}')
    return diffs


def compare(path_a, path_b, max_frames=None, verbose=False):
    print(f'A: {path_a}')
    print(f'B: {path_b}\n')

    data_a = load(path_a)
    data_b = load(path_b)

    keys_a = set(data_a.keys())
    keys_b = set(data_b.keys())
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    print(f'=== Frame coverage ===')
    print(f'  A frames : {len(keys_a)}')
    print(f'  B frames : {len(keys_b)}')
    print(f'  Common   : {len(common)}')
    print(f'  Only in A: {len(only_a)}')
    print(f'  Only in B: {len(only_b)}')
    if verbose and only_a:
        print(f'    Only-A examples: {only_a[:5]}')
    if verbose and only_b:
        print(f'    Only-B examples: {only_b[:5]}')

    frames_to_check = common[:max_frames] if max_frames else common

    count_diff_frames = 0
    field_diff_frames = 0
    unmatched_ids = 0
    field_diff_counts = defaultdict(int)
    total_objs_checked = 0

    for zyt_name in frames_to_check:
        objs_a = {o['tracking_id']: o for o in data_a[zyt_name]}
        objs_b = {o['tracking_id']: o for o in data_b[zyt_name]}

        # object count
        if len(objs_a) != len(objs_b):
            count_diff_frames += 1
            if verbose:
                print(f'[{zyt_name}] obj count: A={len(objs_a)} B={len(objs_b)}')

        frame_has_diff = False
        for tid, oa in objs_a.items():
            if tid not in objs_b:
                unmatched_ids += 1
                if verbose:
                    print(f'[{zyt_name}] tracking_id={tid} missing in B')
                continue
            diffs = compare_obj(oa, objs_b[tid])
            total_objs_checked += 1
            if diffs:
                frame_has_diff = True
                for d in diffs:
                    field_name = d.split(':')[0]
                    field_diff_counts[field_name] += 1
                if verbose:
                    print(f'[{zyt_name}] id={tid}: {"; ".join(diffs)}')

        if frame_has_diff:
            field_diff_frames += 1

    print(f'\n=== Diff summary (over {len(frames_to_check)} common frames) ===')
    print(f'  Frames with count diff  : {count_diff_frames}')
    print(f'  Frames with field diff  : {field_diff_frames}')
    print(f'  Unmatched tracking_ids  : {unmatched_ids}')
    print(f'  Objects checked         : {total_objs_checked}')
    if field_diff_counts:
        print(f'  Field diff breakdown:')
        for f, cnt in sorted(field_diff_counts.items(), key=lambda x: -x[1]):
            print(f'    {f}: {cnt} objects')
    else:
        print('  No field differences found — files are identical on common objects.')


def main():
    p = argparse.ArgumentParser(description='Compare two bev-anno pkl files.')
    p.add_argument('pkl_a', help='Reference pkl (e.g. company intranet output)')
    p.add_argument('pkl_b', help='Comparison pkl (e.g. local output)')
    p.add_argument('--max-frames', type=int, default=None, help='Limit frames checked (for quick test)')
    p.add_argument('--verbose', action='store_true', help='Print per-frame / per-object diffs')
    args = p.parse_args()
    compare(args.pkl_a, args.pkl_b, args.max_frames, args.verbose)


if __name__ == '__main__':
    main()
