#!/usr/bin/env python3
"""
Convert the MIND dataset (data/MIND/mind_listwise_ranking.json +
mind_item_metadata.json) into MemRec's processed format.

MIND's native format is impression-based (news click-through), not the
leave-one-out sequential format the other AgenticRec_CFmemory datasets use,
and MemRec's RecDataset only supports one train history + one val target +
one test target per user. To fit MIND into that shape, this converter
mirrors exactly what agent_rec_MIND_qwen.py itself does for evaluation
(see its `user_data = user_sequences[user_id][0]` - only each user's FIRST
impression is used, others are ignored):

  - train  = that impression's `history` (chronological, as given)
  - val = test = the impression's positive candidate item (label == 1)
    (MIND's own evaluate_user() has no val/test distinction either - it
    scores one impression once - so val is set equal to test here purely
    to satisfy RecDataset's leave-one-out plumbing; only compare MemRec's
    TEST metrics against the source repo's numbers, not its val metrics)
  - fixed negatives (val_neg == test_neg) = the impression's candidates
    with label == 0

CAVEAT: ~28% of MIND users have MORE THAN ONE positive candidate in their
first impression (multi-relevant ranking), but MemRec can only rank against
a single target item. This converter keeps only the FIRST positive as the
target and drops the other positives from that user's candidate pool
entirely (neither target nor negative), so they don't unfairly count
against the model. Results are exactly comparable to the source repo only
for the ~72% of users with a single positive; for the rest, treat MemRec's
numbers as a reasonable approximation, not an exact match.

Usage:
    python scripts/convert_mind_to_memrec.py
"""
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent.parent / "data" / "MIND"
DATA_NAME = "MIND"


def stringify(value) -> str:
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return ""
    # Collapse embedded tabs/newlines/carriage returns - see the matching
    # comment in convert_agenticrec_to_memrec.py's stringify().
    return " ".join(str(value).split())


def main():
    sequences = json.load(open(SOURCE_ROOT / "mind_listwise_ranking.json"))
    item_meta = json.load(open(SOURCE_ROOT / "mind_item_metadata.json"))
    print(f"[MIND] {len(sequences)} users in source file")

    # Pick each user's first impression (matches agent_rec_MIND_qwen.py),
    # keep only users with a non-empty history and >=1 positive candidate.
    user_records = {}  # raw_user_id -> {history, target, negatives}
    n_multi_positive = 0
    for raw_user, impressions in sequences.items():
        imp = impressions[0]
        history = imp.get("history", [])
        candidates = imp.get("candidates", [])
        labels = imp.get("labels", [])
        positives = [c for c, l in zip(candidates, labels) if l == 1]
        negatives = [c for c, l in zip(candidates, labels) if l == 0]
        if not history or not positives:
            continue
        if len(positives) > 1:
            n_multi_positive += 1
        user_records[raw_user] = {
            "history": history,
            "target": positives[0],
            "negatives": negatives,
        }

    print(f"[MIND] {len(user_records)} usable users "
          f"(dropped {len(sequences) - len(user_records)} with empty history/no positive), "
          f"{n_multi_positive} had >1 positive (extra positives dropped from candidates)")

    user_map = {raw: idx for idx, raw in enumerate(user_records.keys())}

    interacted = set()
    negative_only = set()
    for rec in user_records.values():
        interacted.update(rec["history"])
        interacted.add(rec["target"])
        negative_only.update(rec["negatives"])
    negative_only -= interacted
    order = sorted(negative_only) + sorted(interacted)
    item_map = {raw: idx for idx, raw in enumerate(order)}

    out_dir = PROJECT_ROOT / "data" / "processed" / DATA_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- .inter: train=history, val=test=target (see module docstring) ---
    rows = []
    fixed_negatives = {}
    for raw_user, rec in user_records.items():
        user_id = user_map[raw_user]
        for position, raw_item in enumerate(rec["history"]):
            rows.append({"user_id": user_id, "item_id": item_map[raw_item], "rating": 5.0, "timestamp": position + 1})
        target_id = item_map[rec["target"]]
        base_ts = len(rec["history"]) + 1
        rows.append({"user_id": user_id, "item_id": target_id, "rating": 5.0, "timestamp": base_ts})      # val
        rows.append({"user_id": user_id, "item_id": target_id, "rating": 5.0, "timestamp": base_ts + 1})  # test

        neg_ids = [item_map[n] for n in rec["negatives"]]
        fixed_negatives[str(user_id)] = {"val_neg": neg_ids, "test_neg": neg_ids}

    inter_df = pd.DataFrame(rows)
    inter_path = out_dir / f"{DATA_NAME}.inter"
    inter_df.to_csv(inter_path, sep="\t", index=False)
    print(f"  Saved {inter_path} ({len(inter_df)} rows, {inter_df['user_id'].nunique()} users, {inter_df['item_id'].nunique()} items)")

    # --- .meta ---
    meta_rows = []
    for raw_item, meta in item_meta.items():
        if raw_item not in item_map:
            continue
        title = stringify(meta.get("title"))
        description = " | ".join(filter(None, [
            stringify(meta.get("category")),
            stringify(meta.get("subcategory")),
            stringify(meta.get("abstract")),
        ]))
        # category = the same field agent_rec_MIND_*.py uses (--item_text_mode category)
        category = stringify(meta.get("category"))
        meta_rows.append({"item_id": item_map[raw_item], "asin": raw_item, "title": title,
                          "description": description, "category": category})
    meta_df = pd.DataFrame(meta_rows).sort_values("item_id")
    meta_path = out_dir / f"{DATA_NAME}.meta"
    meta_df.to_csv(meta_path, sep="\t", index=False)
    print(f"  Saved {meta_path} ({len(meta_df)} items)")

    fixed_neg_path = out_dir / f"{DATA_NAME}.fixed_negatives.json"
    with open(fixed_neg_path, "w") as f:
        json.dump(fixed_negatives, f)
    print(f"  Saved {fixed_neg_path} ({len(fixed_negatives)} users)")

    id_maps_path = out_dir / f"{DATA_NAME}.id_maps.json"
    with open(id_maps_path, "w") as f:
        json.dump({"user_id_map": user_map, "item_id_map": item_map}, f)
    print(f"  Saved {id_maps_path}")


if __name__ == "__main__":
    main()
