# anno_utils.py  —  IO helpers, logging, Config dataclass, path builders.
# No core/ imports — fully self-contained.
#
# Usage: imported by handle_anno.py

import json
import logging
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(obj, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def read_pkl(path) -> Any:
    with open(str(path), 'rb') as f:
        return pickle.load(f)


def write_pkl(obj, path: str):
    p = Path(path)
    if p.exists():
        p.unlink()
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_yaml(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def read_image(path) -> np.ndarray:
    return cv2.imread(str(path))


def write_image(path, img: np.ndarray):
    cv2.imwrite(str(path), img)


def read_txt(path: str) -> List[str]:
    with open(path, 'r') as f:
        return [line.rstrip() for line in f]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def set_logging(file_name: str = '/tmp/log.txt'):
    Path(file_name).parent.mkdir(exist_ok=True, parents=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        filename=file_name,
        level=logging.INFO,
        filemode='a',
        format='[%(asctime)s] [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def log_params(**kwargs):
    for k, v in kwargs.items():
        logging.info(f'{k}: {v}')

# ---------------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------------

def namestr_to_int(name: str) -> int:
    name = name.split('/')[-1]
    a, b = name.split('_')[:2]
    return int((int(a) * 1e9 + int(b)) / 1e3)


def to_builtin(obj):
    """Recursively convert numpy / Path objects to Python builtins."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_builtin(v) for v in obj)
    return obj

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    raw_data_tag: str
    anno_tag: str
    filter_name: str
    asset_ids: List[str]
    storage_dirs: List[str]
    datablob_root: str
    anno_output_root: str
    filter_config: Dict              = field(default_factory=dict)
    num_workers_generate: int        = 1
    num_workers_filter: int          = 1
    data_source: str                 = "pse7_5"
    tdr_manifest_path: Optional[str] = None
    generate_config: Dict            = field(default_factory=dict)
    vis_config: Dict                 = field(default_factory=dict)


def load_config(config_path: str) -> Config:
    raw = read_json(config_path)
    tdr_manifest_path = raw.get("tdr_manifest_path")
    if not tdr_manifest_path:
        raise RuntimeError("config 缺少 tdr_manifest_path 字段")
    asset_ids = list(read_json(tdr_manifest_path).keys())
    return Config(
        raw_data_tag=raw["raw_data_tag"],
        anno_tag=raw["anno_tag"],
        filter_name=raw["filter_name"],
        asset_ids=asset_ids,
        storage_dirs=raw["storage_dirs"],
        datablob_root=raw["datablob_root"],
        anno_output_root=raw["anno_output_root"],
        filter_config=raw.get("filter_config", {}),
        num_workers_generate=int(raw.get("num_workers_generate", 1)),
        num_workers_filter=int(raw.get("num_workers_filter", 1)),
        data_source=raw.get("data_source", "pse7_5"),
        tdr_manifest_path=tdr_manifest_path,
        generate_config=raw.get("generate_config", {}),
        vis_config=raw.get("vis_config", {}),
    )


def build_output_dirs(cfg: Config) -> Dict[str, Path]:
    base = Path(cfg.anno_output_root) / cfg.raw_data_tag / cfg.anno_tag
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "base":     base,
        "raw_anno": base,
        "filtered": base,
        "vis":      base / "vis",
        "log":      Path(cfg.anno_output_root) / "logs" / f"{cfg.raw_data_tag}_{cfg.anno_tag}_{ts}.txt",
    }


def setup(cfg: Config, dirs: Dict[str, Path]):
    dirs["raw_anno"].mkdir(parents=True, exist_ok=True)
    dirs["log"].parent.mkdir(parents=True, exist_ok=True)
    set_logging(str(dirs["log"]))
    log_params(
        asset_ids=cfg.asset_ids,
        storage_dirs=cfg.storage_dirs,
        anno_tag=cfg.anno_tag,
        filter_name=cfg.filter_name,
        raw_data_tag=cfg.raw_data_tag,
        save_dir=dirs["base"],
    )
