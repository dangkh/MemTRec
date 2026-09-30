#!/usr/bin/env python3
"""
Generate synthetic per-user persona/instruction data for a converted
AgenticRec_CFmemory dataset, matching MemRec's <name>.instruction schema
(user_id, instruction, persona) - see src/data/dataset_base.py:load_instructions.

Reads data/processed/<name>/<name>.inter and <name>.meta (produced by
convert_agenticrec_to_memrec.py), takes each user's last 10 train items'
titles as their history, and asks an LLM to write a short instruction +
persona in the style of code_instruct_data/instructrec-books/*.instruction.

Requires the same LLM provider credentials as MemRec itself (Azure OpenAI by
default; see README "Setup" - AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY),
or pass a MemRec config file with a `provider:` block via --config.

Usage:
    python scripts/generate_instructions.py --data_name yelp --limit 5
    python scripts/generate_instructions.py --data_name yelp --config configs/memrec_yelp.yaml
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.llm_client import LLMClient
from src.utils import load_config

PERSONA_PROPERTIES = {
    "persona": {"type": "string", "description": "A one-sentence description of who this person plausibly is (occupation/role/interest), inferred only from their item history."},
    "instruction": {"type": "string", "description": "A short first-person request (1-3 sentences) this person might give a recommender, in the style of an InstructRec instruction."},
}


def build_llm_client(config_path: str, model: str, provider_name: str):
    if config_path:
        config = load_config(config_path)
        provider_config = config.get("provider", {})
        return LLMClient(
            api_endpoint=provider_config.get("endpoint", config.get("api_endpoint")),
            api_key=provider_config.get("api_key", config.get("api_key")),
            api_version=provider_config.get("api_version", config.get("api_version", "2024-08-01-preview")),
            model=provider_config.get("model", config.get("llm_model", model)),
            provider_name=provider_config.get("name", provider_name),
        )
    return LLMClient(model=model, provider_name=provider_name)


def load_user_histories(data_dir: Path, data_name: str, limit: int = None, eval_user_list: str = None):
    inter_df = pd.read_csv(data_dir / f"{data_name}.inter", sep="\t")
    meta_df = pd.read_csv(data_dir / f"{data_name}.meta", sep="\t")
    title_by_item = dict(zip(meta_df["item_id"], meta_df["title"]))

    inter_df = inter_df.sort_values(["user_id", "timestamp"])
    histories = {}
    for user_id, group in inter_df.groupby("user_id"):
        # Drop the last 2 (val/test) to only use train items as history.
        train_items = group["item_id"].tolist()[:-2]
        titles = [title_by_item.get(i, "") for i in train_items[-10:]]
        titles = [t for t in titles if t]
        if titles:
            histories[int(user_id)] = titles

    if eval_user_list:
        # Only the users actually being evaluated need an instruction -
        # generating for the whole dataset (e.g. 48k users for MIND) is
        # wasted LLM calls. See scripts/generate_eval_user_list.py.
        wanted = set(json.load(open(eval_user_list))["user_ids"])
        user_ids = [uid for uid in sorted(histories.keys()) if uid in wanted]
    else:
        user_ids = sorted(histories.keys())

    if limit:
        user_ids = user_ids[:limit]
    return {uid: histories[uid] for uid in user_ids}


def generate_for_user(client: LLMClient, user_id: int, titles: list) -> dict:
    history_text = "; ".join(titles)
    messages = [
        {
            "role": "system",
            "content": "You infer a plausible persona and a short recommendation instruction for a user, based only on the titles of items they previously interacted with. Be concise and specific.",
        },
        {
            "role": "user",
            "content": f"This user's recent interaction history (oldest to newest): {history_text}\n\nInfer their persona and write an instruction they might give a recommender system.",
        },
    ]
    result = client.generate_json(messages=messages, properties=PERSONA_PROPERTIES, temperature=0.7, max_tokens=300)
    return {
        "user_id": user_id,
        "instruction": result.get("instruction", ""),
        "persona": result.get("persona", ""),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--config", default=None, help="MemRec config YAML to source LLM provider settings from")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--provider_name", default="azure_openai")
    parser.add_argument("--limit", type=int, default=None, help="Only generate for the first N users (cheap dry run)")
    parser.add_argument("--eval_user_list", default=None,
                         help="Path to an eval_user_list JSON (see generate_eval_user_list.py) - "
                              "restrict generation to only these users instead of the whole dataset")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "processed" / args.data_name
    histories = load_user_histories(data_dir, args.data_name, limit=args.limit, eval_user_list=args.eval_user_list)
    print(f"Generating instructions for {len(histories)} users...")

    client = build_llm_client(args.config, args.model, args.provider_name)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_for_user, client, user_id, titles): user_id
            for user_id, titles in histories.items()
        }
        for i, future in enumerate(as_completed(futures), 1):
            user_id = futures[future]
            try:
                rows.append(future.result())
            except Exception as e:
                print(f"  Warning: failed for user {user_id}: {e}")
            if i % 50 == 0:
                print(f"  {i}/{len(histories)} done...")

    out_df = pd.DataFrame(rows).sort_values("user_id")
    out_path = data_dir / f"{args.data_name}.instruction"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {out_path} ({len(out_df)} users)")
    print(client.get_token_stats())


if __name__ == "__main__":
    main()
