#!/usr/bin/env python3
"""
Convert AgenticRec_CFmemory datasets (data/<name>/items.json,
user_sequences_*.json, user_negatives_*.json) into MemRec's processed format
(data/processed/<name>/<name>.{inter,meta,text,fixed_negatives.json,id_maps.json}).

Usage:
    python scripts/convert_agenticrec_to_memrec.py --data_name Video_Game
    python scripts/convert_agenticrec_to_memrec.py --data_name yelp
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT.parent.parent / "data"  # AgenticRec_CFmemory/data

# Preferred sequence/negative file suffixes, in priority order, per dataset.
SEQ_CANDIDATES = ["user_sequences_10_5000.json", "user_sequences_10_1000.json", "user_sequences_10.json"]
NEG_CANDIDATES = ["user_negatives_10_5000.json", "user_negatives_10_1000.json", "user_negatives_10.json"]


def stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        text = " ".join(str(v) for v in value)
    else:
        text = str(value)
    # Collapse embedded tabs/newlines/carriage returns - some Amazon
    # descriptions contain raw control chars that, at this file's scale,
    # can survive pandas' CSV quote/escape round-trip incorrectly and
    # desync row alignment on read (observed: item_id ending up holding
    # description text). Plain text never needs them, so just flatten.
    return " ".join(text.split())


def pick_existing(data_dir: Path, candidates):
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {candidates} found in {data_dir}")


def build_item_map(sequences: dict, negatives: dict):
    # Only map items that actually appear in interactions/candidates - NOT the
    # full items.json catalog (which can be orders of magnitude larger, e.g.
    # ~2.9M items.json entries for Books vs ~25k items actually interacted
    # with). Using the full catalog would make RecDataset's
    # n_items = df['item_id'].max()+1 balloon far beyond what's in .inter.
    interacted = set()
    for user_data in sequences.values():
        interacted.update(str(i) for i in user_data.get("train", []))
        if "val" in user_data:
            interacted.update(str(i) for i in (user_data["val"] if isinstance(user_data["val"], list) else [user_data["val"]]))
        if "test" in user_data:
            interacted.update(str(i) for i in (user_data["test"] if isinstance(user_data["test"], list) else [user_data["test"]]))

    negative_only = set()
    for user_data in negatives.values():
        negative_only.update(str(i) for i in user_data.get("val_neg", []))
        negative_only.update(str(i) for i in user_data.get("test_neg", []))
    negative_only -= interacted

    # RecDataset derives n_items from df['item_id'].max()+1 over the .inter
    # file alone, which only contains `interacted` items. Assigning
    # negative-only items the LOWEST ids and interacted items the highest
    # guarantees the max id in .inter equals len(item_map)-1, so no
    # fixed-negative candidate ever falls outside [0, n_items).
    order = sorted(negative_only) + sorted(interacted)
    return {raw: idx for idx, raw in enumerate(order)}


def convert(data_name: str):
    data_dir = SOURCE_ROOT / data_name
    items_meta = json.load(open(data_dir / "items.json"))
    seq_path = pick_existing(data_dir, SEQ_CANDIDATES)
    neg_path = pick_existing(data_dir, NEG_CANDIDATES)
    sequences = json.load(open(seq_path))
    negatives = json.load(open(neg_path))

    print(f"[{data_name}] sequences={seq_path.name} negatives={neg_path.name} "
          f"n_users={len(sequences)} n_items_meta={len(items_meta)}")

    item_map = build_item_map(sequences, negatives)
    user_map = {raw: idx for idx, raw in enumerate(sequences.keys())}

    out_dir = PROJECT_ROOT / "data" / "processed" / data_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- .inter ---
    rows = []
    for raw_user, user_data in sequences.items():
        user_id = user_map[raw_user]
        items_in_order = list(user_data.get("train", []))
        if "val" in user_data:
            v = user_data["val"]
            items_in_order += v if isinstance(v, list) else [v]
        if "test" in user_data:
            t = user_data["test"]
            items_in_order += t if isinstance(t, list) else [t]

        for position, raw_item in enumerate(items_in_order):
            key = str(raw_item)
            if key not in item_map:
                continue  # skip items missing metadata/mapping
            rows.append({
                "user_id": user_id,
                "item_id": item_map[key],
                "rating": 5.0,
                "timestamp": position + 1,
            })

    inter_df = pd.DataFrame(rows)
    inter_path = out_dir / f"{data_name}.inter"
    inter_df.to_csv(inter_path, sep="\t", index=False)
    print(f"  Saved {inter_path} ({len(inter_df)} rows, "
          f"{inter_df['user_id'].nunique()} users, {inter_df['item_id'].nunique()} items)")

    # --- .meta ---
    meta_rows = []
    for raw_item, meta in items_meta.items():
        key = str(raw_item)
        if key not in item_map:
            continue
        title = stringify(meta.get("title"))
        description = stringify(meta.get("description")) or stringify(meta.get("main_cat"))
        # category = main_cat only, the item text agent_rec_*.py uses (--item_text_mode category)
        main_cat = meta.get("main_cat")
        category = ", ".join(str(c) for c in main_cat) if isinstance(main_cat, list) else stringify(main_cat)
        meta_rows.append({
            "item_id": item_map[key],
            "asin": raw_item,
            "title": title,
            "description": description,
            "category": " ".join(category.split()),
        })
    meta_df = pd.DataFrame(meta_rows).sort_values("item_id")
    meta_path = out_dir / f"{data_name}.meta"
    meta_df.to_csv(meta_path, sep="\t", index=False)
    print(f"  Saved {meta_path} ({len(meta_df)} items)")

    # --- .text (yelp only, review_texts_train.json) ---
    review_path = data_dir / "review_texts_train.json"
    if review_path.exists():
        reviews = json.load(open(review_path))
        text_rows = []
        for raw_user, item_reviews in reviews.items():
            if raw_user not in user_map:
                continue
            user_id = user_map[raw_user]
            for raw_item, review_text in item_reviews.items():
                if str(raw_item) not in item_map:
                    continue
                text_rows.append({
                    "user_id": user_id,
                    "item_id": item_map[str(raw_item)],
                    "review_text": review_text,
                })
        text_df = pd.DataFrame(text_rows)
        text_path = out_dir / f"{data_name}.text"
        text_df.to_csv(text_path, sep="\t", index=False)
        print(f"  Saved {text_path} ({len(text_df)} reviews)")

    # --- fixed_negatives.json ---
    fixed_negatives = {}
    for raw_user, neg_data in negatives.items():
        if raw_user not in user_map:
            continue
        user_id = user_map[raw_user]
        val_neg = [item_map[str(i)] for i in neg_data.get("val_neg", []) if str(i) in item_map]
        test_neg = [item_map[str(i)] for i in neg_data.get("test_neg", []) if str(i) in item_map]
        fixed_negatives[str(user_id)] = {"val_neg": val_neg, "test_neg": test_neg}

    fixed_neg_path = out_dir / f"{data_name}.fixed_negatives.json"
    with open(fixed_neg_path, "w") as f:
        json.dump(fixed_negatives, f)
    print(f"  Saved {fixed_neg_path} ({len(fixed_negatives)} users)")

    # --- id maps (traceability) ---
    id_maps_path = out_dir / f"{data_name}.id_maps.json"
    with open(id_maps_path, "w") as f:
        json.dump({"user_id_map": user_map, "item_id_map": item_map}, f)
    print(f"  Saved {id_maps_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_name", required=True,
                         choices=["Video_Game", "Books", "CDs_and_Vinyl", "yelp", "ml"])
    args = parser.parse_args()
    convert(args.data_name)


if __name__ == "__main__":
    main()
