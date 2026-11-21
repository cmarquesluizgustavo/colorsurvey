"""Post-training geometry (silhouette) and tolerance-8 crowd comparison metrics."""
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split

DEFAULT_MAX_SILHOUETTE_CLASSES = 797
DEFAULT_MAX_CROWD_CLASSES = 96
CUBE_SIDE = 256
CUBE_SIZE = CUBE_SIDE ** 3


def compute_centroids(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute per-class mean embeddings from all rows in the split."""
    emb = torch.as_tensor(np.ascontiguousarray(embeddings), dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    centroids = torch.zeros(num_classes, emb.shape[1], dtype=torch.float32).index_add_(0, y, emb)
    counts = torch.bincount(y, minlength=num_classes).clamp(min=1).unsqueeze(1).float()
    return centroids / counts


def simplified_silhouette_samples(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    chunk_size: int = 15_000,
) -> np.ndarray:
    """Compute simplified (centroid) silhouette (b - a) / max(a, b) in chunks."""
    emb = torch.as_tensor(np.ascontiguousarray(embeddings), dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    out = torch.empty(len(emb), dtype=torch.float32)
    for start in range(0, len(emb), chunk_size):
        dist = torch.cdist(emb[start:start + chunk_size], centroids)
        y_chunk = y[start:start + chunk_size]
        rows = torch.arange(len(y_chunk))
        own = dist[rows, y_chunk].clone()
        dist[rows, y_chunk] = float("inf")
        nearest_other = dist.min(dim=1).values
        denom = torch.maximum(own, nearest_other)
        out[start:start + chunk_size] = torch.where(
            denom == 0, torch.zeros_like(denom), (nearest_other - own) / denom
        )
    return out.numpy()


def _stratified_subsample(labels_np: np.ndarray, subsample: int, seed: int) -> np.ndarray:
    """Return stratified row indices, falling back when classes are singletons or remainders are small."""
    counts = np.bincount(labels_np)
    present_ids = np.flatnonzero(counts > 0)
    counts_p = counts[present_ids]
    n = len(labels_np)
    if counts_p.min() >= 2 and n - subsample >= len(present_ids):
        idx, _ = train_test_split(
            np.arange(n),
            train_size=subsample,
            stratify=labels_np,
            random_state=seed,
        )
        return idx

    rng = np.random.default_rng(seed)
    if subsample < len(present_ids):
        return rng.choice(n, size=subsample, replace=False)

    quotas = np.maximum(np.floor(counts_p * (subsample / n)).astype(int), 1)
    while quotas.sum() > subsample:
        reducible = np.flatnonzero(quotas > 1)
        quotas[reducible[np.argmax(quotas[reducible])]] -= 1
    while quotas.sum() < subsample:
        room = counts_p - quotas
        avail = np.flatnonzero(room > 0)
        frac = (counts_p[avail] * (subsample / n)) - quotas[avail]
        take = min(int(subsample - quotas.sum()), len(avail))
        order = avail[np.argsort(-frac, kind="stable")[:take]]
        quotas[order] += 1

    chosen = [
        rng.choice(np.flatnonzero(labels_np == cls_id), size=q, replace=False)
        for cls_id, q in zip(present_ids, quotas)
    ]
    return np.concatenate(chosen)


def compute_silhouette_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    subsample: int = 60_000,
    chunk_size: int = 15_000,
    seed: int = 42,
) -> dict:
    """Compute overall, per-class-mean, and per-name simplified silhouette."""
    labels_np = np.asarray(labels, dtype=np.int64)
    present_all = np.bincount(labels_np, minlength=num_classes) > 0
    present_ids = np.flatnonzero(present_all)
    if len(present_ids) < 2:
        return {
            "silhouette": float("nan"),
            "silhouette_per_class": float("nan"),
            "per_class_silhouette": np.full(num_classes, np.nan),
        }

    if len(present_ids) == num_classes:
        mapped_labels = labels_np
    else:
        remap = np.full(num_classes, -1, dtype=np.int64)
        remap[present_ids] = np.arange(len(present_ids), dtype=np.int64)
        mapped_labels = remap[labels_np]

    centroids = compute_centroids(embeddings, mapped_labels, len(present_ids))
    if len(mapped_labels) > subsample:
        idx = _stratified_subsample(mapped_labels, subsample, seed)
    else:
        idx = np.arange(len(mapped_labels))

    scores = simplified_silhouette_samples(
        embeddings[idx], mapped_labels[idx], centroids, chunk_size=chunk_size
    )
    y_sub = labels_np[idx]
    weights = np.bincount(y_sub, minlength=num_classes).astype(float)
    present = weights > 0
    per_class = np.bincount(y_sub, weights=scores, minlength=num_classes) / np.maximum(weights, 1.0)

    return {
        "silhouette": float((per_class * weights).sum() / weights.sum()),
        "silhouette_per_class": float(per_class[present].mean()),
        "per_class_silhouette": np.where(present, per_class, np.nan),
    }


def _box_sum(cube_flat: np.ndarray, tol: int) -> np.ndarray:
    """Sum a flattened 256^3 cube over +/- tol box windows on each channel."""
    n = CUBE_SIDE
    if tol <= 0:
        return cube_flat.astype(np.float32, copy=True)
    if 2 * tol + 1 >= n:
        total = float(cube_flat.sum())
        return np.full(CUBE_SIZE, total, dtype=np.float32)

    a = np.array(cube_flat, dtype=np.float32, copy=True).reshape(n, n, n)
    b = np.empty_like(a)
    for axis in range(3):
        prefix = (slice(None),) * axis
        np.cumsum(a, axis=axis, out=a)
        b[prefix + (slice(None, tol + 1),)] = a[prefix + (slice(tol, 2 * tol + 1),)]
        np.subtract(
            a[prefix + (slice(2 * tol + 1, None),)],
            a[prefix + (slice(None, n - 2 * tol - 1),)],
            out=b[prefix + (slice(tol + 1, n - tol),)],
        )
        np.subtract(
            a[prefix + (slice(n - 1, n),)],
            a[prefix + (slice(n - 2 * tol - 1, n - tol - 1),)],
            out=b[prefix + (slice(n - tol, None),)],
        )
        a, b = b, a
    return a.reshape(-1)


def _width_90(dist: np.ndarray) -> int:
    """Count cube voxels needed to hold 90% of a normalized distribution."""
    positive = dist[dist > 0]
    if positive.size == 0:
        return 0
    ordered = np.sort(positive)[::-1]
    return int(np.searchsorted(np.cumsum(ordered, dtype=np.float64), 0.9)) + 1


def _to_cube(uniq_codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    cube = np.zeros(CUBE_SIZE, dtype=np.float32)
    cube[uniq_codes] = values
    return cube


def compute_crowd_metrics(
    color_embeds: torch.Tensor,
    class_text_embeds: torch.Tensor,
    labels: torch.Tensor,
    rgb_01: np.ndarray,
    temperature: float,
    part_a_mask: np.ndarray,
    part_b_mask: np.ndarray,
    tol: int = 8,
    min_answers: int = 20,
    step: int = 2,
    chunk_classes: int = 8,
) -> dict:
    """Compare model distributions and held-out human halves on the RGB cube at tol=8."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    labels_np = np.asarray(labels, dtype=np.int64)
    half_a = np.asarray(part_a_mask, dtype=bool)
    half_b = np.asarray(part_b_mask, dtype=bool)
    x_int = np.rint(np.asarray(rgb_01) * (CUBE_SIDE - 1)).astype(np.int64)
    if len(color_embeds) != len(labels_np) or len(x_int) != len(labels_np):
        raise ValueError("color_embeds, labels, and rgb_01 must share the same row count")
    if len(half_a) != len(labels_np) or len(half_b) != len(labels_np):
        raise ValueError("part_a_mask and part_b_mask must match labels length")

    codes = (x_int[:, 0] * CUBE_SIDE + x_int[:, 1]) * CUBE_SIDE + x_int[:, 2]
    uniq, first_idx, inv = np.unique(codes, return_index=True, return_inverse=True)
    num_unique = len(uniq)
    num_classes = class_text_embeds.shape[0]
    uniq_rgb = np.stack(
        [uniq // (CUBE_SIDE * CUBE_SIDE), uniq // CUBE_SIDE % CUBE_SIDE, uniq % CUBE_SIDE],
        axis=1,
    )
    uniq_embeds = color_embeds[torch.from_numpy(first_idx)]

    chunks = [list(range(start, min(start + chunk_classes, num_classes)))
              for start in range(0, num_classes, chunk_classes)]

    # Pass 1: exact per-colour log-partition normalizer in class chunks.
    lse = np.full(num_unique, -np.inf, dtype=np.float32)
    with torch.no_grad():
        for ks in chunks:
            logits = (uniq_embeds @ class_text_embeds[ks].t()).numpy() / temperature
            lse = np.logaddexp(lse, np.logaddexp.reduce(logits, axis=1))

    grid_1d = np.arange(0, CUBE_SIDE, step)
    grid = ((grid_1d[:, None, None] * CUBE_SIDE + grid_1d[None, :, None]) * CUBE_SIDE + grid_1d[None, None, :]).ravel()

    cnt = np.bincount(inv, minlength=num_unique).astype(np.float32)
    cnt_a = np.bincount(inv[half_a], minlength=num_unique).astype(np.float32)
    cnt_b = np.bincount(inv[half_b], minlength=num_unique).astype(np.float32)

    tot = _box_sum(_to_cube(uniq, cnt), tol)
    tot_a = _box_sum(_to_cube(uniq, cnt_a), tol)
    tot_b = _box_sum(_to_cube(uniq, cnt_b), tol)
    near = grid[tot[grid] >= min_answers]
    t, ta, tb = tot[near], tot_a[near], tot_b[near]
    del tot, tot_a, tot_b

    both = (ta > 0) & (tb > 0)
    diff_model = np.zeros(len(near), dtype=np.float32)
    diff_halves = np.zeros(len(near), dtype=np.float32)
    best_model = np.full(len(near), -1.0, dtype=np.float32)
    best_people = np.full(len(near), -1.0, dtype=np.float32)
    top_model = np.zeros(len(near), dtype=np.int16)
    top_people = np.zeros(len(near), dtype=np.int16)

    per_name_model = np.full(num_classes, np.nan, dtype=np.float64)
    per_name_halves = np.full(num_classes, np.nan, dtype=np.float64)
    per_centre_dist = np.full(num_classes, np.nan, dtype=np.float64)
    per_width_ratio = np.full(num_classes, np.nan, dtype=np.float64)

    with torch.no_grad():
        for ks in chunks:
            logits = (uniq_embeds @ class_text_embeds[ks].t()).numpy() / temperature
            for j, k in enumerate(ks):
                rows = labels_np == k
                people_cube = _box_sum(_to_cube(uniq, np.bincount(inv[rows], minlength=num_unique).astype(np.float32)), tol)
                a_cube = _box_sum(_to_cube(uniq, np.bincount(inv[rows & half_a], minlength=num_unique).astype(np.float32)), tol)
                b_cube = np.maximum(people_cube - a_cube, 0.0) if (half_a ^ half_b).all() else _box_sum(
                    _to_cube(uniq, np.bincount(inv[rows & half_b], minlength=num_unique).astype(np.float32)), tol
                )

                # Colour-to-name: compare model soft probabilities against people shares on near grid.
                shares_people = people_cube[near] / np.maximum(t, 1.0)
                shares_a = a_cube[near] / np.maximum(ta, 1.0)
                shares_b = b_cube[near] / np.maximum(tb, 1.0)
                model_probs = np.exp(logits[:, j] - lse)
                shares_model = _box_sum(_to_cube(uniq, cnt * model_probs), tol)[near] / np.maximum(t, 1.0)

                diff_model += np.abs(shares_model - shares_people) / 2.0
                diff_halves += np.abs(shares_a - shares_b) / 2.0

                win_m = shares_model > best_model
                best_model[win_m], top_model[win_m] = shares_model[win_m], k
                win_p = shares_people > best_people
                best_people[win_p], top_people[win_p] = shares_people[win_p], k

                # Name-to-colour: spread of t->c distribution over observed RGB colours.
                a_sum = float(a_cube.sum())
                b_sum = float(b_cube.sum())
                if a_sum > 0 and b_sum > 0:
                    per_name_halves[k] = float(np.abs(a_cube / a_sum - b_cube / b_sum).sum() / 2.0)
                del a_cube, b_cube, shares_a, shares_b, shares_people, shares_model

                p_sum = float(people_cube.sum())
                if p_sum > 0:
                    people_dist = people_cube / p_sum
                else:
                    people_dist = None
                del people_cube

                q = np.log(cnt) + logits[:, j]
                q = np.exp(q - q.max())
                q /= q.sum()
                model_dist = _box_sum(_to_cube(uniq, q), tol)
                model_sum = float(model_dist.sum())
                if model_sum > 0:
                    model_dist /= model_sum

                if people_dist is not None:
                    per_name_model[k] = float(np.abs(model_dist - people_dist).sum() / 2.0)
                    w_people = _width_90(people_dist)
                    if w_people > 0:
                        per_width_ratio[k] = float(_width_90(model_dist) / w_people)
                    centre_model = q @ uniq_rgb
                    centre_people = x_int[rows].mean(axis=0)
                    per_centre_dist[k] = float(np.linalg.norm(centre_model - centre_people))
                del model_dist, people_dist

    return {
        "colour_to_name_diff": float(diff_model.mean()) if len(near) else float("nan"),
        "colour_to_name_halves": float(diff_halves[both].mean()) if both.any() else float("nan"),
        "top_name_match": float((top_model == top_people).mean()) if len(near) else float("nan"),
        "name_to_colour_diff": float(np.nanmean(per_name_model)) if np.any(~np.isnan(per_name_model)) else float("nan"),
        "name_to_colour_halves": float(np.nanmean(per_name_halves)) if np.any(~np.isnan(per_name_halves)) else float("nan"),
        "centre_distance": float(np.nanmean(per_centre_dist)) if np.any(~np.isnan(per_centre_dist)) else float("nan"),
        "width_ratio": float(np.nanmedian(per_width_ratio)) if np.any(~np.isnan(per_width_ratio)) else float("nan"),
        "per_class_name_to_colour_diff": per_name_model,
        "per_class_name_to_colour_halves": per_name_halves,
        "per_class_centre_distance": per_centre_dist,
        "per_class_width_ratio": per_width_ratio,
    }


def compute_post_training_metrics(
    color_embeds: torch.Tensor,
    class_text_embeds: torch.Tensor,
    labels: torch.Tensor,
    rgb_01: np.ndarray,
    temperature: float,
    part_a_mask: np.ndarray,
    part_b_mask: np.ndarray,
    compute_silhouette: Optional[bool] = None,
    compute_crowd: Optional[bool] = None,
) -> dict:
    """Compute post-training geometry and crowd agreement metrics on clean A + B rows."""
    num_classes = class_text_embeds.shape[0]
    run_sil = (num_classes <= DEFAULT_MAX_SILHOUETTE_CLASSES) if compute_silhouette is None else bool(compute_silhouette)
    run_crowd = (num_classes <= DEFAULT_MAX_CROWD_CLASSES) if compute_crowd is None else bool(compute_crowd)

    result = {}
    if run_sil:
        result.update(compute_silhouette_metrics(color_embeds, labels, num_classes))
    if run_crowd:
        result.update(
            compute_crowd_metrics(
                color_embeds=color_embeds,
                class_text_embeds=class_text_embeds,
                labels=labels,
                rgb_01=rgb_01,
                temperature=temperature,
                part_a_mask=part_a_mask,
                part_b_mask=part_b_mask,
            )
        )
    return result
