# handle_anno.py  —  Annotation generate / filter / visualize pipeline.
# No core/ imports — fully self-contained (depends only on anno_utils).
#
# Usage:
#   python handle_anno.py --config configs/xxx.json --anno --filter --vis
#   python handle_anno.py --config configs/xxx.json --vis --vis-stage raw --to-video --max-frames 50

import argparse
import dataclasses
import logging
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from anno_utils import (
    Config,
    build_output_dirs,
    load_config,
    namestr_to_int,
    read_image,
    read_json,
    read_pkl,
    read_txt,
    read_yaml,
    setup,
    to_builtin,
    write_image,
    write_json,
    write_pkl,
)

# ===========================================================================
# Constants
# ===========================================================================

SUB_TYPE_MAPPING = {
    'car': 0, 'bus': 1, 'truck': 2, 'suv': 3, 'van': 4,
    'pickupTruck': 5, 'pedestrian': 6, 'dogcat': 7, 'other_animal': 8,
    'motorcycle': 9, 'bicycle': 10, 'tricycle': 11, 'traffic_cone': 12,
    'round_water_barrier': 13, 'sphericalStonePier': 14, 'traffic_pillar': 15,
    'unknown': -1,
}
SUB_TYPE_MAPPING_INV = {v: k for k, v in SUB_TYPE_MAPPING.items()}

SILENT_CLASSES = frozenset([
    'round_water_barrier', 'hydrant', 'trash_can', 'traffic_cone',
    'babycar', 'nostop_plate', 'dogcat',
])

# ===========================================================================
# Generate — data reading helpers
# ===========================================================================

def get_asset_rawdata_zytanno_dir(asset_id: str, storage_list: List[str]) -> Optional[str]:
    for storage_dir in storage_list:
        rawdata_dir = Path(storage_dir) / 'rawdata'
        if not rawdata_dir.exists():
            continue
        for root, dirs, _ in os.walk(rawdata_dir):
            if asset_id in dirs and Path(root).stem == "zyt_anno":
                return str(Path(root) / asset_id)
    logging.warning(f"Cannot find asset {asset_id} in storage_list")
    return None


def read_vins_infos(vins_dir: str) -> Dict:
    vins_dict: Dict = {}
    vins_path = Path(vins_dir)

    if vins_path.exists() and any(vins_path.iterdir()):
        for f in vins_path.iterdir():
            if f.suffix != '.txt' or f.name == 'oss_succ.txt':
                continue
            ts = namestr_to_int(f.stem)
            for line in read_txt(str(f)):
                if 'imu_to_world' in line:
                    tf = np.array(line.split(' ')[1:], dtype=np.float64).reshape(3, 4)
                    vins_dict[ts] = {'pos': tf[:, 3].reshape(1, 3), 'imu_to_world': tf}
    elif (vins_path.parent / 'vins.json').exists():
        data = read_json(str(vins_path.parent / 'vins.json'))
        for name, entry in data.items():
            ts = namestr_to_int(name)
            tf = np.array(entry['imu_to_world']).reshape(3, 4)
            vins_dict[ts] = {'pos': tf[:, 3].reshape(1, 3), 'imu_to_world': tf}
    else:
        logging.error(f'Cannot read vins from {vins_dir}')
        return {}

    # Velocity by finite differences
    ts_ms, half_w = 1e5, 2
    for timestamp in vins_dict:
        pos_list, ts_list = [], []
        for dt in range(-half_w, half_w):
            t = int(timestamp + dt * ts_ms)
            if t in vins_dict:
                pos_list.append(vins_dict[t]['pos'])
                ts_list.append(t)
        if len(pos_list) >= 2:
            pos_arr = np.array(pos_list).reshape(-1, 3)
            ts_arr = np.array(ts_list)
            dx = (pos_arr[1:] - pos_arr[:-1])
            dt_arr = (ts_arr[1:] - ts_arr[:-1]).reshape(-1, 1)
            vel = (dx / (dt_arr / ts_ms / 10)).mean(axis=0)
            vins_dict[timestamp]['vel'] = vel.reshape(1, 3)
    return vins_dict


def read_calib_zyt(calib_path: str) -> Dict:
    if not Path(calib_path).exists():
        raise RuntimeError(f'Cannot find calib: {calib_path}')
    all_calib = read_json(calib_path).get('calibdata')
    calib = all_calib[next(iter(all_calib))]
    keep = {'P_l_4_crop', 'cam_4_rectified_to_ego', '(crop)P_l_4', 'cam4_to_imu', 'P_l_4', 'P_l_11'}
    result = {k: [float(x) for x in v] for k, v in calib.items() if k in keep}
    if 'P_l_4_crop' not in result and '(crop)P_l_4' in result:
        result['P_l_4_crop'] = result['(crop)P_l_4']
    if 'cam_4_rectified_to_ego' not in result and 'cam4_to_imu' in result:
        result['cam_4_rectified_to_ego'] = result['cam4_to_imu']
    return result


def read_calib_hybrid(datablob_dir) -> Dict:
    calib_path = Path(datablob_dir) / 'calibration' / 'cam_to_cam.yaml'
    if not calib_path.exists():
        raise RuntimeError(f'Cannot find hybrid calib: {calib_path}')
    cal = read_yaml(str(calib_path))
    T_rgb_ego = cal['extrinsics'].get('T_rgb_ego')
    if T_rgb_ego is not None:
        T_rgb_ego = np.array(T_rgb_ego)
    else:
        logging.warning(f'T_rgb_ego missing in {calib_path}')
    return {
        'conf_rgb': cal['intrinsics']['cam0']['camera_matrix'],
        'conf_event': cal['intrinsics']['cam1']['camera_matrix'],
        'T_evnet_rgb': np.array(cal['extrinsics']['T_10']),
        'T_rgb_ego': T_rgb_ego,
    }


def read_asset_infos(asset_id: str, zytanno_dir: str) -> Optional[Dict]:
    frames_dir = Path(zytanno_dir) / 'frames'
    if not frames_dir.exists():
        logging.warning(f'Missing frames dir: {frames_dir}')
        return None
    anno_path = Path(zytanno_dir) / "pseudo_7" / "bevdet_tracked" / f"{asset_id}.pkl"
    if not anno_path.exists():
        logging.error(f'Missing anno pkl: {anno_path}')
        return None
    vins_dict = read_vins_infos(str(frames_dir / 'vins'))
    if not vins_dict:
        logging.warning('Empty vins dict')
        return None
    calib_dict = read_calib_zyt(str(frames_dir / 'calibration_all.json'))
    return {
        'asset_rawdata_dir': frames_dir,
        'asset_anno_path': anno_path,
        'vins_dict': vins_dict,
        'zyt_calib_dict': calib_dict,
    }

# ===========================================================================
# Generate — 3D box geometry
# ===========================================================================

def _bbox_to_corner_ego(x, y, z, l, h, w, ry):
    cx = w * np.array([-1, 1, 1, -1, -1, 1, 1, -1]) / 2
    cy = l * np.array([-1, -1, 1, 1, -1, -1, 1, 1]) / 2
    cz = h * np.array([1, 1, 1, 1, -1, -1, -1, -1]) / 2
    cart = np.stack([cx, cy, cz, np.ones_like(cx)], axis=1)
    R = np.array([
        np.cos(ry), np.sin(ry), 0, 0,
        -np.sin(ry), np.cos(ry), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]).reshape(4, 4)
    corners = (R @ cart.T).T
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z
    return corners[:, :3]


def _convert_pse_box3d(pbox3d):
    x, y, z, l, h, w, ry = np.array(pbox3d).flatten()
    return np.array([x, y, z, l, w, h, np.pi / 2 - ry])


def box3d_to_2d(box3d: np.ndarray, calib_dict: Dict, T_ego_camRect4) -> Optional[Dict]:
    def conf_to_K(conf):
        K = np.eye(3)
        K[[0, 1, 0, 1], [0, 1, 2, 2]] = conf
        return K

    def transform_pts(pts, T):
        h = np.ones((pts.shape[0], 1))
        return ((T @ np.hstack([pts, h]).T).T)[:, :3]

    def project(pts, P):
        h = np.ones((pts.shape[0], 1))
        p = (P @ np.hstack([pts, h]).T).T
        p[:, :2] /= p[:, 2:3]
        return p[:, :2]

    x, y, z, l, h, w, ry = box3d.flatten()
    if x < 0:
        return None

    corners_ego = _bbox_to_corner_ego(x, y, z, l, h, w, ry)
    T_ego_cam4 = np.vstack([np.array(T_ego_camRect4).reshape(3, 4), [0, 0, 0, 1]])
    T_rgb_ego = np.array(calib_dict['T_rgb_ego'])
    T_event_rgb = np.array(calib_dict['T_evnet_rgb'])
    P_event = np.hstack([conf_to_K(calib_dict['conf_event']), np.zeros((3, 1))])
    T_event_ego = T_event_rgb @ T_rgb_ego

    corners_cam = transform_pts(corners_ego, T_event_ego)
    front = corners_cam[corners_cam[:, 2] > 0]
    if front.shape[0] < 2:
        return {
            'bbox_large': np.array([-1, -1, -1, -1]),
            'bbox3d_Fcam': corners_cam,
            'bbox3d_Fimg': np.empty((0, 2)),
            'T_event_ego': T_event_ego,
            'P_event': P_event[:3, :3],
        }

    pts_2d = project(front, P_event)
    bbox = np.array([
        pts_2d[:, 0].min(), pts_2d[:, 1].min(),
        pts_2d[:, 0].max(), pts_2d[:, 1].max(),
    ], dtype=int)
    return {
        'bbox3d_Fraw': corners_ego,
        'bbox3d_Fcam': corners_cam,
        'bbox3d_Fimg': pts_2d,
        'bbox_large': bbox,
        'T_event_ego': T_event_ego,
        'P_event': P_event[:3, :3],
    }


def compute_ttc(cuboid_list: List[Dict], ego_vel: np.ndarray, tf: np.ndarray) -> List[Dict]:
    dt = 0.1
    for itm in cuboid_list:
        itm['ego_to_world'] = tf
        min_depth = np.array(itm['box3d_Fcam'])[:, 2].min()
        rel_vx = ego_vel[0] - itm['speed_ego'][0]
        ttc = min_depth / rel_vx if rel_vx != 0 else np.inf
        itm.update({'ttc': ttc, 'ego_vel': ego_vel, 'scale': ttc / (ttc - dt) if ttc != dt else np.inf})
    return cuboid_list

# ===========================================================================
# Generate — occlusion
# ===========================================================================

def _draw_box_mask(box, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
    return mask


def compute_occlusion_ratios(boxes, depths, shape, ids):
    n = len(boxes)
    order = np.argsort(depths)
    occ_mask = np.zeros(shape, dtype=np.uint8)
    areas = np.zeros(n, dtype=np.int32)
    visible = [None] * n
    for idx in order:
        m = _draw_box_mask(boxes[idx], shape)
        areas[idx] = m.sum()
        visible[idx] = cv2.bitwise_and(m, cv2.bitwise_not(occ_mask))
        occ_mask = cv2.bitwise_or(occ_mask, visible[idx])
    ratios = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ratios[i] = 1.0 - visible[i].sum() / areas[i] if areas[i] > 0 else 0.0
    return ratios

# ===========================================================================
# Generate — read cuboid annotation
# ===========================================================================

def read_cuboid_anno(anno_itm, zyt_calib_dict, hybrid_calib_dict,
                     timecorr_anno, zyttimestamp, bev_anno_dict=None):
    ts_name = anno_itm.get('sample_idx')
    ts = namestr_to_int(ts_name)

    mot: Dict = {}
    if bev_anno_dict is not None:
        frame = bev_anno_dict.get(zyttimestamp)
        if not frame:
            return None
        bboxes_3d, cls_names, speeds, track_ids = [], [], [], []
        for obj in frame:
            vote_dict = obj.get('vote_dict')
            if vote_dict is not None and not vote_dict[0].get('vote_score_valid_ratio', 0):
                continue
            # box: prefer opt frame if both states are 'successes' and lidar_3d_opt exists
            use_opt = (obj.get('opt_state') == 'successes'
                       and obj.get('traj_opt_state') == 'successes'
                       and obj.get('lidar_3d_opt') is not None)
            raw_box = obj['lidar_3d_opt'] if use_opt else obj.get('lidar_3d')
            if raw_box is None:
                continue
            b = np.array(raw_box[:7], dtype=np.float32)
            if not (b[3] > 1e-5 and b[4] > 1e-5 and b[5] > 1e-5):
                b = np.array(obj['lidar_3d'][:7], dtype=np.float32)
            # speed: convert from 10x to m/s
            spd_raw = obj.get('speed_ego_10x', [-998, -998, 0])
            if obj.get('mot_tracking_type', -10) == 2:
                mot_spd = obj.get('mot_speed_ego_10x')
                if mot_spd and obj.get('mot_speed_available', False):
                    spd_raw = mot_spd
            if spd_raw[0] == -998 and spd_raw[1] == -998:
                continue
            spd = np.array(spd_raw[:2], dtype=np.float32) * 0.1  # → m/s
            bboxes_3d.append(b)
            cls_names.append(obj.get('cls_name', 'unknown'))
            speeds.append(spd)
            track_ids.append(obj.get('tracking_id', -1))
        if not bboxes_3d:
            return None
        mot['gt_bboxes_3d'] = np.array(bboxes_3d, dtype=np.float32)
        mot['gt_labels_3d'] = cls_names
        mot['gt_speed_ego'] = np.array(speeds, dtype=np.float32)
        mot['gt_track_id'] = np.array(track_ids, dtype=np.int64)
    else:
        bboxes = anno_itm.get('bboxes')
        track_ids = anno_itm.get('tracking_id')
        classes = anno_itm.get('class_names')
        speeds = anno_itm.get('speed_ego_mot') or anno_itm.get('speed_ego')
        if isinstance(speeds, list):
            speeds = np.array(speeds)
        if any(v is None for v in [bboxes, track_ids, classes, speeds]):
            return None
        if len(classes) == 0:
            return None
        mot['gt_bboxes_3d'] = np.array(bboxes, dtype=np.float32)
        if isinstance(classes[0], (int, np.integer)):
            mot['gt_labels_3d'] = [SUB_TYPE_MAPPING_INV.get(int(c), 'unknown') for c in classes]
        else:
            mot['gt_labels_3d'] = list(classes)
        mot['gt_speed_ego'] = np.array(speeds, dtype=np.float32)
        mot['gt_track_id'] = np.array(track_ids, dtype=np.int64)

    rtn = []
    for speed, bbox3d, track_id, cls_name in zip(
        mot['gt_speed_ego'], mot['gt_bboxes_3d'], mot['gt_track_id'], mot['gt_labels_3d']
    ):
        box3d_converted = _convert_pse_box3d(bbox3d)
        info = box3d_to_2d(box3d_converted, hybrid_calib_dict, zyt_calib_dict['cam_4_rectified_to_ego'])
        if info is None:
            continue
        if track_id == -1:
            continue
        rtn.append({
            'timestamp': ts,
            'ts_name': ts_name,
            'box_l': info['bbox_large'],
            'box3d_ego': box3d_converted,
            'box3d_Fcam': info['bbox3d_Fcam'],
            'bbox3d_Fimg': info['bbox3d_Fimg'],
            'T_event_ego': info['T_event_ego'],
            'P_event': info['P_event'],
            'trackid': track_id,
            'cls_name': cls_name,
            'speed_ego': speed,
            'speed_ego_10x': speed * 10,
            'cx': hybrid_calib_dict['conf_event'][2],
            'cy': hybrid_calib_dict['conf_event'][3],
            'corr_exposure_start_timestamp_us': timecorr_anno['corr_exposure_start_timestamp_us'],
            'corr_exposure_end_timestamp_us': timecorr_anno['corr_exposure_end_timestamp_us'],
            'match_diff_ms': timecorr_anno['match_diff_ms'],
        })
    return rtn

# ===========================================================================
# Generate — single frame training data
# ===========================================================================

def _clip_box(box, roi):
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi
    cx1, cy1 = max(bx1, rx1), max(by1, ry1)
    cx2, cy2 = min(bx2, rx2 - 1), min(by2, ry2 - 1)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return [int(cx1), int(cy1), int(cx2), int(cy2)]


def _has_two_corners_in_roi(pts_2d, roi):
    count = sum(1 for p in pts_2d if roi[0] <= p[0] <= roi[2] and roi[1] <= p[1] <= roi[3])
    return count >= 2


def generate_training_data_single_frame(
    hybrid_image_path: Path, cuboid_anno: List[Dict],
    img_roi, vd_w_range, vd_h_range, vd_h_w_ratio_range, vd_area_range,
    target_classes, asset_id, anno_tag, raw_data_tag, save_dir,
):
    image = read_image(str(hybrid_image_path))
    if image is None:
        return None
    height, width = image.shape[:2]

    for itm in cuboid_anno:
        itm['mean_depth'] = float(np.mean(np.array(itm['box3d_Fcam'])[:, 2]))
    cuboid_anno.sort(key=lambda x: x['mean_depth'])

    iou_boxes, iou_depths, iou_ids = [], [], []
    rtn = []
    ts_name = None

    for itm in cuboid_anno:
        if itm['mean_depth'] < 0:
            continue
        vd_box = _clip_box(itm['box_l'], img_roi)
        if vd_box is None:
            continue
        if not _has_two_corners_in_roi(itm['bbox3d_Fimg'], img_roi):
            continue

        iou_boxes.append(vd_box)
        iou_depths.append(float(np.array(itm['box3d_Fcam'])[:, 2].min()))
        iou_ids.append(int(itm['trackid']))

        cls_name = itm['cls_name']
        if cls_name not in target_classes:
            if cls_name not in SILENT_CLASSES:
                logging.info(f'Unused class: {cls_name} {hybrid_image_path}')
            continue

        x1, y1, x2, y2 = vd_box
        w, h = x2 - x1, y2 - y1
        if not (vd_w_range[0] <= w <= vd_w_range[1]):
            continue
        if not (vd_h_range[0] <= h <= vd_h_range[1]):
            continue
        if not (vd_area_range[0] <= w * h <= vd_area_range[1]):
            continue
        if w > 0 and not (vd_h_w_ratio_range[0] <= h / w <= vd_h_w_ratio_range[1]):
            continue

        box_l = itm['box_l']
        vd_area = (vd_box[2] - vd_box[0]) * (vd_box[3] - vd_box[1])
        bl_area = (box_l[2] - box_l[0]) * (box_l[3] - box_l[1])
        vdbox_boxl_iou = vd_area / bl_area if bl_area > 0 else 0

        idx = hybrid_image_path.parts.index(asset_id)
        image_ref_path = Path(*hybrid_image_path.parts[idx + 1:])

        x_, y_, z_, l_, h_, w_, ry_ = np.array(itm['box3d_ego']).flatten()
        box3d_egoZup = np.array([x_, -y_, -z_, l_, h_, w_, -ry_])
        T_event_egoZup = itm['T_event_ego'] @ np.diag([1, -1, -1, 1]).astype(float)

        ts_name = itm['ts_name']
        save_dict = {
            'track_id': f"{asset_id}_{int(itm['trackid']):06d}",
            ts_name: {
                'private_track_id': f"{int(itm['trackid']):06d}",
                'public_track_id': f"{asset_id}_{int(itm['trackid']):06d}",
                'timestamp': int(itm['timestamp']),
                'cls_name': str(cls_name),
                'box': vd_box,
                'box_l': np.array(box_l).tolist() if hasattr(box_l, 'tolist') else list(box_l),
                'box3d_ego': box3d_egoZup,
                'box3d_Fcam': itm['box3d_Fcam'],
                'box3d_Fimg': itm['bbox3d_Fimg'],
                'T_event_ego': T_event_egoZup,
                'speed_ego': itm['speed_ego'],
                'speed_ego_10x': itm['speed_ego_10x'],
                'ego_vel': itm['ego_vel'],
                'ttc': itm['ttc'],
                'ego_to_world': itm['ego_to_world'],
                'meta': {
                    'scores': {'occ_ratio': 0.0, 'vdbox_boxl_iou': vdbox_boxl_iou},
                    'image_shape': image.shape,
                    'image_path': str(image_ref_path),
                    'corr_exposure_start_timestamp_us': itm['corr_exposure_start_timestamp_us'],
                    'corr_exposure_end_timestamp_us': itm['corr_exposure_end_timestamp_us'],
                    'match_diff_ms': itm['match_diff_ms'],
                    'cx': itm['cx'],
                    'cy': itm['cy'],
                    'anno_tag': anno_tag,
                    'raw_data_tag': raw_data_tag,
                },
            },
        }
        rtn.append(to_builtin(save_dict))

    if iou_boxes and ts_name:
        occ_ratios = compute_occlusion_ratios(iou_boxes, iou_depths, (height, width), iou_ids)
        for i, oid in enumerate(iou_ids):
            for item in rtn:
                if int(item[ts_name]['private_track_id']) == oid:
                    item[ts_name]['meta']['scores']['occ_ratio'] = float(occ_ratios[i])
    return rtn

# ===========================================================================
# Generate — single-frame entry point
# ===========================================================================

def generate_patch_single_frame(
    zyt_name, vins_input, zyt_calib_dict, hybrid_calib_dict,
    timecorr_anno, save_dir, asset_id, anno_tag, raw_data_tag,
    zytanno, database_dir, data_source, bev_anno_dict, generate_config,
):
    cuboid_anno = read_cuboid_anno(
        anno_itm=zytanno,
        zyt_calib_dict=zyt_calib_dict,
        hybrid_calib_dict=hybrid_calib_dict,
        timecorr_anno=timecorr_anno,
        zyttimestamp=zyt_name,
        bev_anno_dict=bev_anno_dict,
    )
    if not cuboid_anno:
        return None
    if 'vel' not in vins_input or vins_input['vel'] is None:
        return None

    ego_vel_world = vins_input['vel'][0]
    R_ego_w = vins_input['imu_to_world'][:3, :3].T
    ego_vel = R_ego_w @ ego_vel_world
    cuboid_anno = compute_ttc(cuboid_anno, ego_vel, vins_input['imu_to_world'])

    image_path = Path(database_dir) / timecorr_anno['hybrid_rgb_distored_path']
    if not image_path.exists():
        logging.error(f'Missing image: {image_path}')
        return None

    gc = generate_config or {}
    return generate_training_data_single_frame(
        hybrid_image_path=image_path,
        cuboid_anno=cuboid_anno,
        img_roi=gc.get('img_roi', [0, 0, 1280, 720]),
        vd_w_range=gc.get('vd_w_range', [10, np.inf]),
        vd_h_range=gc.get('vd_h_range', [10, np.inf]),
        vd_h_w_ratio_range=gc.get('vd_h_w_ratio_range', [0, np.inf]),
        vd_area_range=gc.get('vd_area_range', [0, np.inf]),
        target_classes=gc.get('target_classes', [
            'car', 'truck', 'bus', 'other_vehicle',
            'pedestrian', 'motorcycle', 'tricycle', 'bicycle',
        ]),
        asset_id=asset_id,
        anno_tag=anno_tag,
        raw_data_tag=raw_data_tag,
        save_dir=save_dir,
    )

# ===========================================================================
# Generate — top-level multi-asset
# ===========================================================================

def generate_anno_multiple_sequence(
    asset_list, storage_list, datablob_base_dir, anno_tag, raw_data_tag,
    save_dir, config_snapshot, num_workers=1, data_source="pse7_5",
    tdr_manifest_path=None, generate_config=None,
):
    config_save = str(Path(save_dir).parent.parent / 'config.json')
    write_json(config_snapshot, config_save)
    logging.info(f"Saved config to {config_save}")

    bev_manifest = {}
    if tdr_manifest_path and Path(tdr_manifest_path).exists():
        bev_manifest = read_json(tdr_manifest_path)
        logging.info(f"TDR manifest: {len(bev_manifest)} assets")

    def _process_asset(asset_id):
        logging.info(f"Processing asset {asset_id}")
        zytanno_dir = get_asset_rawdata_zytanno_dir(asset_id, storage_list)
        if zytanno_dir is None:
            return
        datablob_dir = Path(datablob_base_dir) / asset_id
        timecorr_pkl = datablob_dir / 'timecorr.pkl'
        if not timecorr_pkl.exists():
            logging.error(f"Missing timecorr.pkl for {asset_id}")
            return
        timecorr_dict = read_pkl(timecorr_pkl).get('anno')

        asset_infos = read_asset_infos(asset_id, zytanno_dir)
        if asset_infos is None:
            return
        zytanno_list = read_pkl(asset_infos['asset_anno_path'])
        vins_dict = asset_infos['vins_dict']
        zyt_calib_dict = asset_infos['zyt_calib_dict']
        hybrid_calib_dict = read_calib_hybrid(datablob_dir)

        gc = generate_config or {}
        if 'T_rgb_ego' in gc:
            hybrid_calib_dict['T_rgb_ego'] = gc['T_rgb_ego']

        bev_anno_dict = None
        if asset_id in bev_manifest:
            bev_anno_dict = read_pkl(bev_manifest[asset_id])
            logging.info(f"Loaded bev_anno for {asset_id} ({len(bev_anno_dict)} frames)")

        train_list = []
        for zyt_name, timecorr_anno in tqdm(timecorr_dict.items(), desc=f"[Gen] {asset_id}"):
            zyt_ts = namestr_to_int(zyt_name)
            if zyt_ts not in vins_dict:
                continue
            zytanno_input = None
            for item in zytanno_list:
                if str(item['sequence_name']) == asset_id and str(item['sample_idx']) == zyt_name:
                    zytanno_input = item
                    break
            if zytanno_input is None:
                continue
            result = generate_patch_single_frame(
                zyt_name=zyt_name,
                vins_input=vins_dict[zyt_ts],
                zyt_calib_dict=zyt_calib_dict,
                hybrid_calib_dict=hybrid_calib_dict,
                timecorr_anno=timecorr_anno,
                save_dir=save_dir,
                asset_id=asset_id,
                anno_tag=anno_tag,
                raw_data_tag=raw_data_tag,
                zytanno=zytanno_input,
                database_dir=datablob_dir,
                data_source=data_source,
                bev_anno_dict=bev_anno_dict,
                generate_config=generate_config,
            )
            if result:
                train_list.extend(result)

        path = Path(save_dir) / f"raw_{asset_id}.pkl"
        write_pkl(to_builtin(train_list), str(path))
        logging.info(f"Saved anno: {path}")

    if num_workers < 2:
        for aid in asset_list:
            _process_asset(aid)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = {ex.submit(_process_asset, aid): aid for aid in asset_list}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    logging.error(f"Failed {futs[f]}: {e}")

# ===========================================================================
# Filter
# ===========================================================================

def _modify_infinity(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                _modify_infinity(v)
            elif v == 'Infinity':
                data[k] = float('inf')
    elif isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, (dict, list)):
                _modify_infinity(v)
            elif v == 'Infinity':
                data[i] = float('inf')


def _is_valid_by_class(ann, cfg_dict):
    if cfg_dict is None:
        return True
    return ann['cls_name'] in cfg_dict['valid_class_list']


def _is_valid_by_box_h(ann, cfg_dict):
    if cfg_dict is None:
        return True
    _, y1, _, y2 = ann['box']
    h = y2 - y1
    cls = ann['cls_name']
    lo, hi = cfg_dict[cls]['valid_range'] if cls in cfg_dict else cfg_dict['valid_range']
    return lo <= h <= hi


def _is_valid_by_occ_ratio(ann, cfg_dict):
    if cfg_dict is None:
        return True
    occ = ann.get('meta', {}).get('scores', {}).get('occ_ratio')
    if occ is None:
        return True
    lo, hi = cfg_dict['valid_range']
    return lo <= occ <= hi


def _is_valid(ann, cfg, false_counter):
    if not _is_valid_by_class(ann, cfg.get('filter_by_class')):
        false_counter['filter_by_class'] += 1
        return False, 'Class'
    if not _is_valid_by_box_h(ann, cfg.get('filter_by_box_h')):
        false_counter['filter_by_box_h'] += 1
        return False, 'boxH'
    if not _is_valid_by_occ_ratio(ann, cfg.get('filter_by_occ_ratio')):
        false_counter['filter_by_occ_ratio'] += 1
        return False, 'occ_ratio'
    return True, None


def get_merge_track_id_list(train_list):
    merged = defaultdict(lambda: {'track_id': None, 'class': None, 'valid_ts_list': [], 'anno': {}})
    for entry in train_list:
        tid = entry['track_id']
        zyt_name = next(k for k in entry if k != 'track_id')
        merged[tid]['track_id'] = str(tid)
        merged[tid]['class'] = entry[zyt_name].get('cls_name')
        if entry[zyt_name].get('valid', False):
            merged[tid]['valid_ts_list'].append(int(namestr_to_int(zyt_name)))
        merged[tid]['anno'][zyt_name] = entry[zyt_name]
    result = []
    for td in merged.values():
        td['valid_ts_list'].sort()
        td['anno'] = dict(sorted(td['anno'].items()))
        result.append(td)
    return result


def filter_data_list_single_asset(pkl_path, save_path, cfg):
    if not Path(pkl_path).exists():
        logging.warning(f'Missing pkl: {pkl_path}')
        return
    train_list = read_pkl(pkl_path)
    false_counter: Dict = defaultdict(int)
    valid_count = 0

    for itm in tqdm(train_list, desc=f'filter [{Path(pkl_path).stem}]'):
        ann_key = next(k for k in itm if isinstance(itm[k], dict))
        ann = itm[ann_key]
        try:
            valid, reason = _is_valid(ann, cfg, false_counter)
            ann['valid'] = valid
            ann['not_valid_reason'] = reason
            if valid:
                valid_count += 1
        except Exception as e:
            logging.error(f'Filter error: {e}')
            ann['valid'] = False
            ann['not_valid_reason'] = str(e)

    merged = get_merge_track_id_list(train_list)
    write_pkl(to_builtin(merged), save_path)
    pprint(dict(false_counter), sort_dicts=False)
    logging.info(f'Filtered {Path(pkl_path).stem}: {len(train_list)} total, {valid_count} valid')


def filter_data_list_multiple_assets(
    asset_list, extranno_dir, filter_dir, filter_config_dict, filter_config_key,
    num_workers=1,
):
    def _task(asset_id):
        pkl = str(Path(extranno_dir) / f"raw_{asset_id}.pkl")
        out = str(Path(filter_dir) / f"{filter_config_key}_{asset_id}.pkl")
        filter_data_list_single_asset(pkl, out, filter_config_dict)
        logging.info(f"Saved filtered: {out}")

    if num_workers < 2:
        for aid in asset_list:
            _task(aid)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            list(as_completed([ex.submit(_task, a) for a in asset_list]))

# ===========================================================================
# Visualization
# ===========================================================================

def _ffmpeg_pipe_open(output_path, width, height, fps=10, crf=28, preset="fast", scale_w=None):
    vf = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    if scale_w and scale_w > 0:
        vf = f"scale={scale_w}:-2," + vf
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
        '-vf', vf, '-pix_fmt', 'yuv420p',
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _get_anno_value_by_false_reason(ann, reason):
    if reason == 'Class':
        return ann.get('cls_name')
    if reason == 'boxH':
        box = ann.get('box')
        return box[3] - box[1] if box and len(box) == 4 else None
    if reason == 'occ_ratio':
        return ann.get('meta', {}).get('scores', {}).get('occ_ratio')
    return None


BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def draw_bev(cuboid_anno, canvas_w=800, canvas_h=500,
             fwd_range=(0, 60), lat_range=(-30, 30)):
    """
    BEV strip placed below the camera image.
    box3d_ego convention (confirmed from TTC formula and vis_bev):
      [0] = b0 = FORWARD  → vertical axis, up = ahead
      [1] = b1 = LATERAL  → horizontal axis, centre = ego lateral position
    box corners: width w spans b1-axis, length l spans b0-axis (at ry=π/2 = fwd-facing).
    Rotation same as bbox_to_corner_ego: R = [[cos,sin],[-sin,cos]] (CW).
    speed_ego is per-0.1 s → multiply by 10 for m/s.
    """
    GREEN  = (0, 255, 0)
    RED    = (0, 0, 255)
    ORANGE = (0, 165, 255)
    WHITE  = (255, 255, 255)
    GRAY   = (50, 50, 50)
    LGRAY  = (90, 90, 90)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    scale_fwd = canvas_h / (fwd_range[1] - fwd_range[0])   # px/m along forward
    scale_lat = canvas_w / (lat_range[1] - lat_range[0])   # px/m along lateral

    def to_px(b0_fwd, b1_lat):
        """(forward_m, lateral_m) → (col, row) on canvas."""
        col = int((b1_lat - lat_range[0]) * scale_lat)
        row = int((fwd_range[1] - b0_fwd) * scale_fwd)
        return col, row

    # Horizontal grid lines (forward distance)
    for d in range(int(fwd_range[0]), int(fwd_range[1]) + 1, 20):
        _, r = to_px(d, 0)
        cv2.line(canvas, (0, r), (canvas_w, r), GRAY, 1)
        cv2.putText(canvas, f'{d}m', (3, r - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, LGRAY, 1)
    # Vertical centre line (lateral = 0)
    c0, _ = to_px(0, 0)
    cv2.line(canvas, (c0, 0), (c0, canvas_h), GRAY, 1)

    # Ego vehicle at (b0=0, b1=0): small rectangle + forward arrow
    ego_col, ego_row = to_px(0, 0)
    cv2.rectangle(canvas, (ego_col - 5, ego_row - 12), (ego_col + 5, ego_row + 6), WHITE, 2)
    cv2.arrowedLine(canvas, (ego_col, ego_row), to_px(8, 0), WHITE, 2, tipLength=0.3)

    for itm in cuboid_anno:
        b = np.array(itm.get('box3d_ego', [])).flatten()
        if len(b) < 7:
            continue
        b0, b1, b2, l, h, w, ry = b

        ttc = itm.get('ttc', float('inf'))
        if np.isfinite(ttc) and 0 < ttc < 2:
            color = RED
        elif np.isfinite(ttc) and ttc < 5:
            color = ORANGE
        else:
            color = GREEN

        corners_3d = _bbox_to_corner_ego(b0, b1, b2, l, h, w, ry)
        footprint = corners_3d[:4, :2]  # top-face 4 corners, (fwd, lat)
        pts_px = np.array([to_px(c[0], c[1]) for c in footprint], dtype=np.int32)
        cv2.polylines(canvas, [pts_px.reshape(-1, 1, 2)], True, color, 2)

        # Label
        cx, cy = to_px(b0, b1)
        cls_name = itm.get('cls_name', '')
        track_id = itm.get('private_track_id', itm.get('trackid', ''))
        cv2.putText(canvas, f'{cls_name}#{track_id}',
                    (cx + 3, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Speed arrow: speed_ego is per-0.1s → *10 = m/s, project 2 s ahead
        raw_spd = np.array(itm.get('speed_ego', [0, 0])).flatten()[:2]
        spd = raw_spd * 10  # m/s; [0]=forward component, [1]=lateral component
        ec, er = to_px(b0 + spd[0] * 2, b1 + spd[1] * 2)
        if abs(ec - cx) + abs(er - cy) > 3:
            cv2.arrowedLine(canvas, (cx, cy), (ec, er), color, 1, tipLength=0.3)

    return canvas


def vis_cuboid3d_anno(cuboid_anno, image, anno_type):
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    ORANGE = (0, 165, 255)
    BROWN = (19, 69, 139)
    PURPLE = (255, 0, 255)
    NOT_VALID = (255, 255, 0)

    img = image
    hybrid_name = ""

    for itm in cuboid_anno:
        track_id = str(itm.get('private_track_id', ''))
        hybrid_name = Path(itm.get('meta', {}).get('image_path', '')).stem
        cls_name = itm.get('cls_name', '')
        b_valid = itm.get('valid', True)
        reason = itm.get('not_valid_reason')

        plotbox = True
        plot_3d = True
        thick = 2

        if not b_valid:
            if reason == "occ_ratio":
                plotbox = False
            elif reason in ("noYoloDet", "removebyhand", "removebycode"):
                color_map = {"noYoloDet": PURPLE, "removebyhand": BROWN, "removebycode": NOT_VALID}
                track_label = f"{int(track_id)}"
                text_color = rectangle_color = color_map[reason]
                plot_3d = False
            else:
                val = _get_anno_value_by_false_reason(itm, reason)
                if isinstance(val, float):
                    val = f"{val:.1f}"
                track_label = f'{reason}_{val}_{int(track_id)}'
                text_color = rectangle_color = NOT_VALID
                plot_3d = False
                thick = 1
        else:
            ttc = itm.get('ttc', float('inf'))
            spd_norm = float(np.linalg.norm(itm.get('speed_ego', [0, 0]))) * 10  # per-0.1s → m/s
            ttc_str = f' ttc={ttc:.1f}s' if (np.isfinite(ttc) and ttc > 0) else ''
            track_label = f'{int(track_id)} {cls_name} v={spd_norm:.1f}{ttc_str}'
            if reason == "savebycode":
                text_color = rectangle_color = ORANGE
            elif np.isfinite(ttc) and 0 < ttc < 2:
                text_color = rectangle_color = RED
            elif np.isfinite(ttc) and ttc < 5:
                text_color = rectangle_color = ORANGE
            else:
                text_color = rectangle_color = GREEN

        if not plotbox:
            continue

        x1, y1, x2, y2 = [int(v) for v in itm['box']]
        cv2.rectangle(img, (x1, y1), (x2, y2), rectangle_color, thick)
        cv2.putText(img, track_label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thick)

        if plot_3d:
            pts = np.array(itm.get('box3d_Fimg', [])).astype(int)
            if pts.shape[0] >= 8:
                for p in pts:
                    cv2.circle(img, (int(p[0]), int(p[1])), 2, GREEN, -1)
                for e in BOX_EDGES:
                    cv2.line(img, tuple(pts[e[0]]), tuple(pts[e[1]]), GREEN, 1)

    if hybrid_name:
        cv2.putText(img, hybrid_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

    bev = draw_bev(cuboid_anno, canvas_w=img.shape[1], canvas_h=500)
    return np.vstack([img, bev])


def gen_video_single_asset(asset_id, anno_dir, vis_dir, datablob_dir, vis_config, anno_type: str):
    if anno_type == 'extract_anno':
        matching = [Path(anno_dir) / f"raw_{asset_id}.pkl"]
        matching = [p for p in matching if p.exists()]
    else:
        matching = [p for p in Path(anno_dir).glob(f"*_{asset_id}.pkl")
                    if not p.name.startswith("raw_")]
    if len(matching) != 1:
        logging.error(f"Expected 1 pkl for {asset_id}, found {len(matching)}: {matching}")
        return
    track_list = read_pkl(str(matching[0]))

    timecorr_pkl = Path(datablob_dir) / "timecorr.pkl"
    if not timecorr_pkl.exists():
        logging.error(f"Missing timecorr.pkl in {datablob_dir}")
        return
    timecorr_dict = read_pkl(str(timecorr_pkl)).get('anno')

    to_video = bool(vis_config.get("to_video", False))
    frame_ext = vis_config.get("frame_ext", "png")
    fps = int(vis_config.get("video_fps", 10))
    crf = int(vis_config.get("video_crf", 28))
    preset = vis_config.get("video_preset", "fast")
    max_frames = vis_config.get("max_frames")
    scale_w = int(vis_config.get("scale_w", 0))
    fast_view = bool(vis_config.get("bool_fast_view", False))
    jump = 10 if fast_view else 1

    frame_dir = None
    if not to_video:
        frame_dir = Path(vis_dir) / asset_id
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True)

    pipe = None
    output_path = None
    idx, saved = 0, 0

    for zyt_name, vis_anno in tqdm(timecorr_dict.items(), desc=f"[Vis] {asset_id}"):
        idx += 1
        if fast_view and idx % jump != 0:
            continue
        img_path = Path(datablob_dir) / vis_anno.get('hybrid_rgb_distored_path')
        if not img_path.exists():
            continue
        image = read_image(str(img_path))
        if image is None:
            continue

        matches = []
        for t in track_list:
            if anno_type == 'extract_anno' and zyt_name in t:
                matches.append(t[zyt_name])
            elif anno_type == 'filter_anno' and zyt_name in t.get('anno', {}):
                matches.append(t['anno'][zyt_name])

        frame = vis_cuboid3d_anno(matches, image.astype("uint8"), anno_type)

        if to_video:
            if pipe is None:
                h, w = frame.shape[:2]
                if anno_type == 'extract_anno':
                    out_stem = f"raw_{asset_id}"
                else:
                    filter_name = vis_config.get("filter_name", "filtered")
                    out_stem = f"{filter_name}_{asset_id}"
                output_path = str(Path(vis_dir) / f"{out_stem}.mp4")
                pipe = _ffmpeg_pipe_open(
                    output_path,
                    w, h, fps, crf, preset,
                    scale_w if scale_w > 0 else None,
                )
            pipe.stdin.write(frame.tobytes())
        else:
            write_image(str(frame_dir / f"{img_path.stem}.{frame_ext}"), frame)

        saved += 1
        if max_frames and saved >= max_frames:
            break

    if pipe:
        pipe.stdin.close()
        pipe.wait()
        logging.info(f"Video: {output_path}")


def gen_video_multiple_assets(
    asset_list, datablob_base_dir, anno_dir, vis_dir, vis_config,
    anno_type: str = 'extract_anno', to_video=False, num_workers=1, max_frames=None,
):
    rt_cfg = dict(vis_config or {})
    rt_cfg["to_video"] = to_video
    if max_frames is not None:
        rt_cfg["max_frames"] = max_frames

    if num_workers < 2:
        for aid in asset_list:
            gen_video_single_asset(aid, anno_dir, vis_dir, Path(datablob_base_dir) / aid, rt_cfg, anno_type)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = [
                ex.submit(gen_video_single_asset, aid, anno_dir, vis_dir,
                          Path(datablob_base_dir) / aid, rt_cfg, anno_type)
                for aid in asset_list
            ]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:
                    logging.error(f"Vis failed: {e}")

# ===========================================================================
# Run functions
# ===========================================================================

def run_anno(cfg: Config, dirs: Dict[str, Path], num_workers: int):
    generate_anno_multiple_sequence(
        asset_list=cfg.asset_ids,
        storage_list=cfg.storage_dirs,
        datablob_base_dir=cfg.datablob_root,
        anno_tag=cfg.anno_tag,
        raw_data_tag=cfg.raw_data_tag,
        save_dir=str(dirs["raw_anno"]),
        config_snapshot=dataclasses.asdict(cfg),
        num_workers=num_workers,
        data_source=cfg.data_source,
        tdr_manifest_path=cfg.tdr_manifest_path,
        generate_config=cfg.generate_config,
    )


def run_filter(cfg: Config, dirs: Dict[str, Path], num_workers: int):
    if not cfg.filter_config:
        raise RuntimeError("filter_config 为空")
    for fk, fc in cfg.filter_config.items():
        _modify_infinity(fc)
        filter_data_list_multiple_assets(
            asset_list=cfg.asset_ids,
            extranno_dir=str(dirs["raw_anno"]),
            filter_dir=str(dirs["filtered"]),
            filter_config_dict=fc,
            filter_config_key=fk,
            num_workers=num_workers,
        )


def run_vis(cfg, dirs, stage, to_video, max_frames, num_workers):
    anno_type = 'filter_anno' if stage == "filtered" else 'extract_anno'
    anno_dir = dirs["filtered"] if stage == "filtered" else dirs["raw_anno"]
    vis_out = dirs["vis"]
    vis_out.mkdir(parents=True, exist_ok=True)
    vis_config = dict(cfg.vis_config or {})
    if stage == "filtered":
        vis_config["filter_name"] = cfg.filter_name
    gen_video_multiple_assets(
        asset_list=cfg.asset_ids,
        datablob_base_dir=cfg.datablob_root,
        anno_dir=str(anno_dir),
        vis_dir=str(vis_out),
        vis_config=vis_config,
        anno_type=anno_type,
        to_video=to_video,
        num_workers=num_workers,
        max_frames=max_frames,
    )

# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="Annotation generate / filter / visualize")
    p.add_argument("--config",      required=True)
    p.add_argument("--anno",        action="store_true")
    p.add_argument("--filter",      action="store_true")
    p.add_argument("--vis",         action="store_true")
    p.add_argument("--vis-stage",   default="filtered", choices=["raw", "filtered"])
    p.add_argument("--to-video",    action="store_true")
    p.add_argument("--max-frames",  type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    args = p.parse_args()

    if not (args.anno or args.filter or args.vis):
        p.print_help()
        return

    cfg = load_config(args.config)
    dirs = build_output_dirs(cfg)
    setup(cfg, dirs)

    if args.anno:
        logging.info("=== anno ===")
        run_anno(cfg, dirs, args.num_workers or cfg.num_workers_generate)
    if args.filter:
        logging.info("=== filter ===")
        run_filter(cfg, dirs, args.num_workers or cfg.num_workers_filter)
    if args.vis:
        logging.info("=== vis ===")
        run_vis(cfg, dirs, args.vis_stage, args.to_video, args.max_frames, args.num_workers or 1)


if __name__ == "__main__":
    main()
