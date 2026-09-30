#!/usr/bin/env python3
"""
Generate a fixed eval_user_list JSON for a converted dataset, matching the
exact user subset AgenticRec_CFmemory's own agent_rec_*.py scripts use:

    user_ids = list(user_sequences.keys())[:number_of_users]

convert_agenticrec_to_memrec.py built each dataset's int user_id by
enumerating that same source user_sequences_*.json file's keys in order
(data/processed/<name>/<name>.id_maps.json: user_id_map), so the first N
values of that map are exactly the same N users (same order) the source
repo's scripts would pick for --number_of_users N. This script just slices
that map - no need to re-read the raw sequences file.

Usage:
    python scripts/generate_eval_user_list.py --data_name yelp --number_of_users 1000
"""
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate(data_name: str, number_of_users: int):
    out_dir = PROJECT_ROOT / "data" / "processed" / data_name
    id_maps = json.load(open(out_dir / f"{data_name}.id_maps.json"))
    user_id_map = id_maps["user_id_map"]  # {raw_user_id: int_id}, insertion-ordered

    all_int_ids = list(user_id_map.values())
    user_ids = all_int_ids[:number_of_users]

    out_path = out_dir / f"{data_name}.eval_user_list_n{number_of_users}.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": data_name,
            "number_of_users": number_of_users,
            "n_total_users": len(all_int_ids),
            "user_ids": user_ids,
        }, f)

    print(f"[{data_name}] {len(user_ids)}/{len(all_int_ids)} users -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_name", required=True,
                         choices=["Video_Game", "Books", "CDs_and_Vinyl", "yelp", "ml", "MIND"])
    parser.add_argument("--number_of_users", type=int, default=1000,
                         help="Matches agent_rec_qwen.py's --number_of_users (default: 1000)")
    args = parser.parse_args()
    generate(args.data_name, args.number_of_users)


if __name__ == "__main__":
    main()
