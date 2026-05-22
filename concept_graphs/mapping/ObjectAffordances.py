from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable

import numpy as np
from scipy.spatial import cKDTree
import logging

log = logging.getLogger(__name__)


@dataclass
class Affordance:
    tag: str
    type: str
    interaction_points: List[Tuple[float, float, float]]
    mask_3d: np.ndarray
    score: float = 1.0
    desc_id: str = ""
    mask_3d_indices_local: np.ndarray | None = None
    mask_3d_indices_global: np.ndarray | None = None


class ObjectAffordances:
    def __init__(self) -> None:
        self._affordances: Dict[int, Affordance] = {}

    def __getitem__(self, key: int) -> Affordance:
        return self._affordances[key]

    def __setitem__(self, key: int, value: Affordance) -> None:
        self._affordances[key] = value

    def __len__(self) -> int:
        return len(self._affordances)

    def items(self):
        return self._affordances.items()

    def values(self):
        return self._affordances.values()

    def _merge_affordances(
        self,
        aff_cluster: Iterable[Affordance],
        merged_desc_id: str | None = None,
    ) -> Affordance:
        cl: List[Affordance] = list(aff_cluster)

        tag = ", ".join(sorted({aff.tag for aff in cl}))
        int_pts = [pt for aff in cl for pt in aff.interaction_points]

        # merge mask_3d row-wise
        mask_stack = [
            aff.mask_3d for aff in cl if aff.mask_3d is not None and aff.mask_3d.size
        ]
        mask_3d = (
            np.unique(np.vstack(mask_stack), axis=0) if mask_stack else np.empty((0, 3))
        )

        score = np.mean([aff.score for aff in cl]) if cl else 0.0

        # helper to merge 1-D index arrays
        def _merge_idx(attr: str) -> np.ndarray | None:
            arrays = [
                getattr(aff, attr)
                for aff in cl
                if getattr(aff, attr) is not None and getattr(aff, attr).size
            ]
            return (
                np.unique(np.concatenate(arrays)) if arrays else np.empty(0, dtype=int)
            )

        return Affordance(
            tag=tag,
            type=cl[0].type,
            interaction_points=int_pts,
            mask_3d=mask_3d,
            score=score,
            desc_id=(merged_desc_id if merged_desc_id is not None else ""),
            mask_3d_indices_local=_merge_idx("mask_3d_indices_local"),
            mask_3d_indices_global=_merge_idx("mask_3d_indices_global"),
        )

    def align(self, pcd_points: np.ndarray, threshold: float = 0.001) -> None:
        """
        For every affordance, find point-cloud indices within threshold
        distance of its mask and store them as mask_3d_indices_local.
        """
        if pcd_points.size == 0:
            log.warning("No PCD points to align affordances.")
            return

        kd = cKDTree(pcd_points)

        for aff in self._affordances.values():
            if aff.mask_3d is None or aff.mask_3d.size == 0:
                log.warning("No mask points for affordance '%s'.", aff.tag)
                aff.mask_3d_indices_local = np.empty(0, dtype=int)
                continue

            # query_ball_point returns a list-of-lists → flatten first
            raw = kd.query_ball_point(aff.mask_3d, r=threshold)
            idx_flat: List[int] = [i for sub in raw for i in sub]
            aff.mask_3d_indices_local = (
                np.unique(idx_flat).astype(int) if idx_flat else np.empty(0, dtype=int)
            )
            if idx_flat:
                aff.mask_3d = pcd_points[aff.mask_3d_indices_local]

    def merge(self, iou_threshold: float = 0.3) -> None:
        """
        Merge duplicate affordances of the same type.
        Two affordances are duplicates if IoU of mask_3d_indices_local >= iou_threshold.  Merging is transitive.
        """

        def _iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
            if a is None or b is None or a.size == 0 or b.size == 0:
                return 0.0
            inter = np.intersect1d(a, b, assume_unique=True).size
            if inter == 0:
                return 0.0
            union = a.size + b.size - inter
            return inter / union

        # ---------- 1. bucket by semantic type ----------------------------
        type_to_keys: Dict[str, List[int]] = defaultdict(list)
        for k, aff in self._affordances.items():
            type_to_keys[aff.type].append(k)

        new_affordances: Dict[int, Affordance] = {}
        next_id = 0

        # ---------- 2. process each bucket (O(N^2) search) -----------------
        for keys in type_to_keys.values():
            affs: List[Affordance] = [self._affordances[k] for k in keys]
            n = len(affs)
            visited = [False] * n

            for i in range(n):
                if visited[i]:
                    continue

                # BFS over IoU graph
                cluster_idx = [i]
                visited[i] = True
                stack = [i]

                while stack:
                    src = stack.pop()
                    for j in range(n):
                        if visited[j]:
                            continue

                        aff_iou = _iou(
                            affs[src].mask_3d_indices_local,
                            affs[j].mask_3d_indices_local,
                        )

                        if aff_iou >= iou_threshold:
                            visited[j] = True
                            stack.append(j)
                            cluster_idx.append(j)

                merged_aff = self._merge_affordances(affs[k] for k in cluster_idx)
                new_affordances[next_id] = merged_aff
                next_id += 1

        self._affordances = new_affordances

    def merge_by_desc_id(self, iou_threshold: float = 0.3) -> None:
        """
        Merge duplicate affordances sharing the same desc_id.
        Two affordances are duplicates if IoU of mask_3d_indices_local >= iou_threshold.
        Uses transitive closure (BFS over IoU graph), analogous to merge(), but
        buckets by desc_id instead of affordance type.
        The merged affordance desc_id is the comma-separated list of unique
        desc_ids in the cluster (sorted).
        """

        def _iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
            if a is None or b is None or a.size == 0 or b.size == 0:
                return 0.0
            inter = np.intersect1d(a, b, assume_unique=True).size
            if inter == 0:
                return 0.0
            union = a.size + b.size - inter
            return inter / union

        # 1. Bucket by desc_id (empty string groups together as well)
        desc_to_keys: Dict[str, List[int]] = defaultdict(list)
        for k, aff in self._affordances.items():
            desc_to_keys[aff.desc_id].append(k)

        new_affordances: Dict[int, Affordance] = {}
        next_id = 0

        # 2. Process each desc_id bucket
        for keys in desc_to_keys.values():
            affs: List[Affordance] = [self._affordances[k] for k in keys]
            n = len(affs)
            visited = [False] * n

            for i in range(n):
                if visited[i]:
                    continue
                cluster_idx = [i]
                visited[i] = True
                stack = [i]

                while stack:
                    src = stack.pop()
                    for j in range(n):
                        if visited[j]:
                            continue
                        if (
                            _iou(
                                affs[src].mask_3d_indices_local,
                                affs[j].mask_3d_indices_local,
                            )
                            >= iou_threshold
                        ):
                            visited[j] = True
                            stack.append(j)
                            cluster_idx.append(j)

                cluster_desc_ids = sorted(
                    {affs[c].desc_id for c in cluster_idx if affs[c].desc_id}
                )
                merged_desc_id = ", ".join(cluster_desc_ids) if cluster_desc_ids else ""
                merged_aff = self._merge_affordances(
                    (affs[k] for k in cluster_idx),
                    merged_desc_id=merged_desc_id,
                )
                new_affordances[next_id] = merged_aff
                next_id += 1

        self._affordances = new_affordances

    def remove_empty_affordances(self) -> None:
        """
        Remove affordances with empty mask_3d_indices_local.
        """
        keys_to_remove = [
            k
            for k, aff in self._affordances.items()
            if aff.mask_3d_indices_local is None or len(aff.mask_3d_indices_local) == 0
        ]
        for k in keys_to_remove:
            del self._affordances[k]
        log.info(f"Removed {len(keys_to_remove)} empty affordances.")
