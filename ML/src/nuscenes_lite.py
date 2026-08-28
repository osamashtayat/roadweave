"""Small read-only nuScenes table loader for RoadWeave conversion.

The official devkit imports plotting, OpenCV, map and segmentation modules even
when a converter only needs metadata.  This loader implements the tiny subset
RoadWeave uses and follows the devkit's reverse-index and box-velocity rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np


class NuScenesLite:
    REQUIRED_TABLES = (
        "category",
        "instance",
        "log",
        "scene",
        "sample",
        "sample_annotation",
    )

    def __init__(self, version: str, dataroot: str, verbose: bool = False) -> None:
        self.version = version
        self.dataroot = Path(dataroot).expanduser().resolve()
        table_root = self.dataroot / version
        if not table_root.is_dir():
            raise FileNotFoundError("nuScenes metadata directory not found: {}".format(table_root))

        self._indices: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for table_name in self.REQUIRED_TABLES:
            path = table_root / "{}.json".format(table_name)
            records = json.loads(path.read_text(encoding="utf-8"))
            setattr(self, table_name, records)
            self._indices[table_name] = {
                str(record["token"]): record for record in records
            }
        self._add_reverse_shortcuts()
        if verbose:
            print("Loaded {} nuScenes scenes with lightweight metadata reader.".format(len(self.scene)))

    def get(self, table_name: str, token: str) -> Mapping[str, Any]:
        return self._indices[table_name][str(token)]

    def _add_reverse_shortcuts(self) -> None:
        category_names = {
            str(record["token"]): str(record["name"])
            for record in self.category
        }
        instance_categories = {
            str(record["token"]): category_names[str(record["category_token"])]
            for record in self.instance
        }
        for sample in self.sample:
            sample["anns"] = []
        for annotation in self.sample_annotation:
            annotation["category_name"] = instance_categories[str(annotation["instance_token"])]
            self._indices["sample"][str(annotation["sample_token"])]["anns"].append(annotation["token"])

    def box_velocity(self, annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
        current = self.get("sample_annotation", annotation_token)
        has_previous = bool(current["prev"])
        has_next = bool(current["next"])
        if not has_previous and not has_next:
            return np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)

        first = self.get("sample_annotation", current["prev"]) if has_previous else current
        last = self.get("sample_annotation", current["next"]) if has_next else current
        first_position = np.asarray(first["translation"], dtype=np.float64)
        last_position = np.asarray(last["translation"], dtype=np.float64)
        first_time = 1e-6 * float(self.get("sample", first["sample_token"])["timestamp"])
        last_time = 1e-6 * float(self.get("sample", last["sample_token"])["timestamp"])
        delta_time = last_time - first_time
        if has_previous and has_next:
            max_time_diff *= 2.0
        if delta_time <= 0.0 or delta_time > max_time_diff:
            return np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
        return (last_position - first_position) / delta_time


class NuScenesCanBusLite:
    def __init__(self, dataroot: str) -> None:
        self.can_directory = Path(dataroot).expanduser().resolve() / "can_bus"
        if not self.can_directory.is_dir():
            raise FileNotFoundError("nuScenes CAN directory not found: {}".format(self.can_directory))

    def get_messages(
        self,
        scene_name: str,
        message_name: str,
        print_warnings: bool = True,
    ) -> List[Dict[str, Any]]:
        del print_warnings
        path = self.can_directory / "{}_{}.json".format(scene_name, message_name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Expected a list of CAN messages in {}".format(path))
        return payload
