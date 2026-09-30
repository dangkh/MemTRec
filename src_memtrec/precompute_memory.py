#!/usr/bin/env python3
"""Precompute train memories or test-user behaviors with Unsloth Gemma.

Usage: python precompute_memory.py {train,test} [mode-specific options]
Both modes see only observed train interactions, never held-out target items.
Train writes one memory per JSONL row; test writes one behavior cache per user.
The modes retain their own window edge cases, generation settings, validation,
retry policy, resume rules and artifact schemas for downstream compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

#!/usr/bin/env python3



# Import Unsloth before other Transformers-dependent model imports.


TRAIN_SCHEMA_VERSION = "amem_local_memory_dual_behavior_v4"
BEHAVIOR_PROMPT_VERSION = "dual_view_discriminative_behavior_v2_taxonomy"


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    global torch
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Prompt and JSON parsing
# =============================================================================

def build_behavior_extraction_prompt(
    interaction_summary: List[Dict[str, Any]],
    task_id: Optional[str] = None,
) -> str:
    """
    Create TWO complementary behavior views from the SAME observed window:

    1) recommendation_behavior / semantic_focus:
       content-aware and discriminative enough to help downstream item ranking.

    2) trajectory_signature + mechanism/scope/direction:
       domain-agnostic abstraction used for cross-user clustering and Tree learning.

    IMPORTANT: this stage only sees OBSERVED interactions. It never sees a future
    test/ground-truth item, so there is no oracle leakage.
    """
    task_field = f'  "task_id": {json.dumps(task_id)},\n' if task_id is not None else ""
    return f"""You are creating a behavioral memory from a SHORT OBSERVED purchase window.

OBSERVED INTERACTIONS:
{json.dumps(interaction_summary, indent=2, ensure_ascii=False)}

Produce TWO COMPLEMENTARY representations.

A) RECOMMENDATION BEHAVIOR — semantic and discriminative
Describe the concrete preference represented by this observed window.
Preserve useful evidence when supported, such as:
- genre/subgenre or product subcategory;
- style, mood, theme, or use case;
- soundtrack / film / game orientation;
- regional or cultural direction;
- live / compilation / collection characteristics;
- creator continuity when it is actually visible;
- other concrete semantic properties that can distinguish relevant future items
  from clearly unrelated items.

This view should be specific enough to help a downstream recommender rank candidate
items, but it must remain grounded ONLY in the observed interactions.

B) TRAJECTORY ABSTRACTION — domain-agnostic and collaborative
Describe HOW the preference evolves within this observed window, independently of
the concrete content. This view is used to cluster behaviors across users and learn
behavior transitions.

Allowed mechanism labels:
[
  "repetition",
  "persistence",
  "collection expansion",
  "deepening",
  "narrowing",
  "broadening",
  "shifting",
  "returning",
  "adjacent exploration",
  "cross-category exploration",
  "refinement",
  "unknown"
]

Allowed scope labels:
[
  "same item",
  "same creator",
  "same collection/series",
  "same subcategory",
  "same broad category",
  "related category",
  "cross-category",
  "mixed",
  "unknown"
]

Allowed direction labels:
[
  "stable",
  "repeat",
  "deepen",
  "narrow",
  "broaden",
  "shift",
  "return",
  "mixed",
  "unknown"
]


TAXONOMY — USE THESE DEFINITIONS STRICTLY

MECHANISM:
- repetition:
  Literal repetition of the EXACT SAME item_id inside the observed window.
  NEVER use repetition merely because different items are similar.
- persistence:
  Different item_ids maintain the same broad preference without a clear move
  toward a narrower/more specific preference.
- collection expansion:
  Different item_ids expand an identifiable creator/series/collection/family.
- deepening:
  Different item_ids move further into the same focused subcategory/style/theme.
- narrowing:
  Preference becomes more selective/specific than earlier interactions.
- broadening:
  Preference expands to a wider range while retaining a recognizable anchor.
- shifting:
  A clear move from one preference/category/style toward another.
- returning:
  The user comes back to an earlier preference after an intervening different one.
- adjacent exploration:
  Movement into a clearly related neighboring category/style.
- cross-category exploration:
  Movement across substantially different categories/styles.
- refinement:
  Different item_ids explore variants inside a very narrow preference.
- unknown:
  Only when the observed evidence is genuinely insufficient.

SCOPE:
- same item:
  ONLY when the exact same item_id occurs more than once.
- same creator:
  Different item_ids but clear creator/artist continuity is visible.
- same collection/series:
  Different item_ids within an identifiable series/collection/family.
- same subcategory:
  Different item_ids sharing a focused genre/subgenre/product subtype.
- same broad category:
  Different item_ids sharing only a broad category.
- related category:
  Different but clearly neighboring categories.
- cross-category:
  Clearly different categories.
- mixed:
  No single relational scope dominates.
- unknown:
  Evidence is insufficient.

DIRECTION:
- repeat:
  ONLY for literal repeat of an exact item_id.
- stable:
  Different item_ids maintain a broadly stable preference.
- deepen:
  Move further into a focused preference.
- narrow:
  Move toward a more specific subset.
- broaden:
  Expand the preference range.
- shift:
  Move toward a different preference.
- return:
  Revisit an earlier preference after leaving it.
- mixed:
  Multiple directions coexist.
- unknown:
  Evidence is insufficient.

CRITICAL DISAMBIGUATION:
- Different item_ids that are semantically similar are NOT "repetition".
- Different albums/products from the same creator are NOT "same item".
- Different items in the same focused style should usually be persistence,
  deepening, refinement, or collection expansion depending on the evidence.
- Use "repetition + same item + repeat" only when an exact item_id repeats.

STRICT RULES:
1. Use ONLY evidence present in the observed interactions.
2. Do NOT predict or invent the user's future item.
3. recommendation_behavior MUST preserve concrete semantic evidence when available.
4. Avoid generic phrases such as "exploring related products", "expanding interests",
   "showing varied preferences", or "continuing preferences" when a more specific
   semantic description is supported.
5. semantic_focus should contain short reusable semantic phrases, not full sentences.
6. trajectory_signature MUST describe HOW behavior evolves, not WHAT content is preferred.
7. Do NOT put artist/item/product titles into trajectory_signature.
8. mechanism/scope/direction must be domain-agnostic.
9. Prefer a specific supported label; use "unknown" only when evidence is genuinely weak.
10. BEFORE choosing repetition/same item/repeat, explicitly check whether an exact
    item_id occurs more than once. If all item_ids are distinct, those labels are forbidden.
11. When item_ids are distinct, distinguish persistence vs collection expansion vs
    deepening/refinement vs broadening/shifting from the semantic relations in the window.
12. confidence reflects confidence in the trajectory abstraction, from 0.0 to 1.0.
13. Return JSON only. No markdown.

Return exactly:
{{
{task_field}  "behavior_explanation": "1-2 concise sentences grounded in the observed interactions",
  "recommendation_behavior": "one concise, concrete, semantically discriminative description of the observed preference",
  "semantic_focus": ["3-8", "short", "semantic", "phrases"],
  "trajectory_signature": "short domain-agnostic phrase describing HOW preference evolves",
  "mechanism": "one allowed mechanism label",
  "scope": "one allowed scope label",
  "direction": "one allowed direction label",
  "confidence": 0.0
}}
"""



def enforce_structural_consistency(
    mechanism: str,
    scope: str,
    direction: str,
    interaction_sequence: List[Dict[str, Any]],
    category_key: str = "item_category",
    normalize_labels: bool = False,
) -> Tuple[str, str, str, List[str], Dict[str, Any]]:
    """
    Deterministic guardrail for labels with literal structural meaning.

    We do NOT try to infer deepening/collection-expansion automatically.
    We only prevent impossible repetition/same-item/repeat labels when no
    exact item_id repeats. This keeps semantic interpretation with Gemma while
    protecting the Tree from taxonomy collapse.
    """
    item_ids = [
        str(x.get("item_id", "")).strip()
        for x in interaction_sequence
        if str(x.get("item_id", "")).strip()
    ]
    unique_ids = set(item_ids)
    has_exact_repeat = (
        len(item_ids) >= 2
        and len(unique_ids) < len(item_ids)
    )
    all_same_item = (
        len(item_ids) >= 2
        and len(unique_ids) == 1
    )

    categories = [
        str(x.get(category_key, "")).strip()
        for x in interaction_sequence
        if str(x.get(category_key, "")).strip()
        and str(x.get(category_key, "")).strip().lower() != "unknown"
    ]
    same_known_category = (
        len(categories) >= 2
        and len(set(categories)) == 1
    )

    adjustments: List[str] = []

    mechanism = str(mechanism or "unknown").strip().lower()
    scope = str(scope or "unknown").strip().lower()
    direction = str(direction or "unknown").strip().lower()

    if normalize_labels:
        mechanism, scope, direction = map(normalize_label, (mechanism, scope, direction))

    if not has_exact_repeat:
        if mechanism == "repetition":
            old = mechanism
            # Do not guess persistence/deepening here. The item category can be
            # too broad (e.g., all CDs), so an impossible literal repetition is
            # converted to unknown and exposed for audit.
            mechanism = "unknown"
            adjustments.append(
                f"mechanism:{old}->{mechanism}:no_exact_item_repeat"
            )

        if scope == "same item":
            old = scope
            scope = "mixed"
            adjustments.append(
                f"scope:{old}->{scope}:no_exact_item_repeat"
            )

        if direction == "repeat":
            old = direction
            direction = "unknown"
            adjustments.append(
                f"direction:{old}->{direction}:no_exact_item_repeat"
            )

    # If every observed interaction is literally the same item, these labels
    # have unambiguous structural support.
    if all_same_item:
        if mechanism in {"unknown", "persistence"}:
            adjustments.append(
                f"mechanism:{mechanism}->repetition:all_same_item"
            )
            mechanism = "repetition"
        if scope != "same item":
            adjustments.append(
                f"scope:{scope}->same item:all_same_item"
            )
            scope = "same item"
        if direction in {"unknown", "stable"}:
            adjustments.append(
                f"direction:{direction}->repeat:all_same_item"
            )
            direction = "repeat"

    evidence = {
        "num_items": len(item_ids),
        "num_unique_item_ids": len(unique_ids),
        "has_exact_item_repeat": bool(has_exact_repeat),
        "all_same_item": bool(all_same_item),
        "same_known_category": bool(same_known_category),
    }

    return mechanism, scope, direction, adjustments, evidence


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start != -1 and end > start:
        text = text[start:end]

    return json.loads(text)


# =============================================================================
# Data loading
# =============================================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_item_info(
    items_meta: Dict[str, Any],
    item_id: Any,
) -> Dict[str, Any]:
    """
    Robust metadata lookup.

    Output IDs are serialized as strings. Missing metadata is marked and the
    interaction is skipped later, matching the original AMem behavior where
    only item IDs found in items_meta are appended to the interaction history.
    """
    sid = str(item_id)
    info = None

    if sid in items_meta:
        info = items_meta[sid]
    elif item_id in items_meta:
        info = items_meta[item_id]
    else:
        try:
            iid = int(item_id)
            if iid in items_meta:
                info = items_meta[iid]
        except (TypeError, ValueError):
            pass

    if info is None:
        return {
            "item_id": sid,
            "item_name": f"Item {sid}",
            "item_category": "Unknown",
            "metadata_found": False,
        }

    category = info.get("main_cat")

    if not category:
        category = info.get("category")

    if not category:
        cats = info.get("categories")
        if isinstance(cats, list) and cats:
            if isinstance(cats[0], list):
                category = " > ".join(str(x) for x in cats[0] if x)
            else:
                category = " > ".join(str(x) for x in cats if x)

    if not category:
        category = "Unknown"

    return {
        "item_id": sid,
        "item_name": str(info.get("title", f"Item {sid}")),
        "item_category": str(category),
        "metadata_found": True,
    }


# =============================================================================
# Window construction
# =============================================================================

def interaction_windows(
    interactions: Sequence[Optional[Dict[str, Any]]], window_size: int,
    *, skip_missing: bool, deduplicate: bool,
) -> List[List[Dict[str, Any]]]:
    """Chunk valid history, using the latest full window for the final tail.

    Train historically skips missing metadata before checking window boundaries.
    Test checks the final raw position even if metadata is missing, and suppresses
    consecutive identical windows. Keep both policies explicit for existing caches.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    history, windows = [], []
    for index, interaction in enumerate(interactions):
        if interaction is None and skip_missing:
            continue
        if interaction is not None:
            history.append(interaction)
        count = len(history)
        if count and (index == len(interactions) - 1 or count % window_size == 0):
            window = [dict(x) for x in history[-min(window_size, count):]]
            if not deduplicate or not windows or window != windows[-1]:
                windows.append(window)
    return windows


def build_windows_for_user(
    user_id: str, user_order: int, user_data: Dict[str, Any],
    items_meta: Dict[str, Any], window_size: int, max_train_items: int,
) -> List[Dict[str, Any]]:
    train_items = user_data.get("train", [])
    if max_train_items > 0:
        train_items = train_items[-max_train_items:]
    interactions = []
    for item_id in train_items:
        item = get_item_info(items_meta, item_id)
        interactions.append({
            "item_id": item["item_id"], "item_name": item["item_name"],
            "item_category": item["item_category"], "action_type": "purchase",
        } if item["metadata_found"] else None)

    records = []
    for index, window in enumerate(interaction_windows(
        interactions, window_size, skip_missing=True, deduplicate=False,
    )):
        summary = [{
            "item_id": x["item_id"], "item": x["item_name"],
            "category": x["item_category"], "action": x["action_type"],
        } for x in window]
        records.append({
            "precompute_id": -1, "user_id": str(user_id),
            "user_order": int(user_order), "window_index": index,
            "interaction_sequence": window,
            "prompt": build_behavior_extraction_prompt(summary),
        })
    return records


def build_all_windows(
    user_sequences: Dict[str, Any],
    items_meta: Dict[str, Any],
    number_of_users: int,
    window_size: int,
    max_train_items: int,
) -> List[Dict[str, Any]]:
    user_ids = list(user_sequences.keys())

    if number_of_users > 0:
        user_ids = user_ids[:number_of_users]

    all_windows: List[Dict[str, Any]] = []

    for user_order, user_id in enumerate(
        tqdm(user_ids, desc="Building windows")
    ):
        all_windows.extend(
            build_windows_for_user(
                user_id=str(user_id),
                user_order=user_order,
                user_data=user_sequences[user_id],
                items_meta=items_meta,
                window_size=window_size,
                max_train_items=max_train_items,
            )
        )

    # Stable global order. The later global-memory builder should replay this
    # exact precompute_id order.
    for precompute_id, rec in enumerate(all_windows):
        rec["precompute_id"] = int(precompute_id)

    return all_windows


# =============================================================================
# Batched Unsloth Gemma inference
# =============================================================================

class BatchedGemmaExtractor:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
        dtype: str,
    ) -> None:
        global torch
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for this Unsloth Gemma precompute script."
            )

        if dtype == "auto":
            torch_dtype = None
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "full_finetuning": False,
        }

        if torch_dtype is not None:
            kwargs["dtype"] = torch_dtype

        print(f"Loading Gemma with Unsloth FastModel: {model_name}")
        self.model, self.tokenizer = FastModel.from_pretrained(**kwargs)

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma3",
        )
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            FastModel.for_inference(self.model)
        except Exception:
            # Some Unsloth versions do not expose this method for all models.
            pass

        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Gemma device: {self.device}")

    def format_prompt(self, user_prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a behavioral memory modeling system for recommender systems. "
                    "Create both a recommendation-oriented semantic view and a "
                    "domain-agnostic trajectory view. Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Matches the working Gemma-3 Unsloth format.
        if text.startswith("<bos>"):
            text = text[len("<bos>"):]

        return text

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> Tuple[List[str], Dict[str, Any]]:
        rendered = [
            self.format_prompt(prompt)
            for prompt in prompts
        ]

        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # For left-padded batched decoder generation, output[:, input_width:]
        # isolates generated tokens for all rows.
        input_width = int(encoded["input_ids"].shape[1])
        real_input_tokens = int(
            encoded["attention_mask"].sum().item()
        )

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "use_cache": True,
            "do_sample": bool(do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generate_kwargs.update({
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
            })

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                **generate_kwargs,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        generated_ids = outputs[:, input_width:]

        responses = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Approximate generated-token count, sufficient for runtime reporting.
        if self.tokenizer.pad_token_id is not None:
            output_tokens = int(
                (generated_ids != self.tokenizer.pad_token_id)
                .sum()
                .item()
            )
        else:
            output_tokens = int(generated_ids.numel())

        stats = {
            "batch_size": len(prompts),
            "input_tokens": real_input_tokens,
            "output_tokens_approx": output_tokens,
            "elapsed_sec": float(elapsed),
        }

        return responses, stats


# =============================================================================
# JSONL output / resume
# =============================================================================

def inspect_existing_output(
    jsonl_path: Path,
) -> Tuple[Set[int], int]:
    """
    Return completed precompute IDs and valid record count.

    Reject the old schema containing an `embedding` field so resume never
    creates a mixed JSONL with both the old and new formats.
    """
    completed: Set[int] = set()
    valid_records = 0

    if not jsonl_path.exists():
        return completed, valid_records

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                # Allow a truncated last line after interruption.
                continue

            if "embedding" in obj:
                raise RuntimeError(
                    f"{jsonl_path} uses the OLD precompute schema containing "
                    f"`embedding` (first detected at line {line_no}). "
                    "Use a new --output filename or delete the old file."
                )

            schema = obj.get("schema_version")
            if schema not in (None, TRAIN_SCHEMA_VERSION):
                raise RuntimeError(
                    f"Unsupported schema_version={schema!r} in {jsonl_path} "
                    f"at line {line_no}."
                )

            if "precompute_id" not in obj:
                raise RuntimeError(
                    f"Missing precompute_id in {jsonl_path} line {line_no}."
                )

            completed.add(int(obj["precompute_id"]))
            valid_records += 1

    return completed, valid_records


def append_memory_records(
    path: Path,
    records: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        # Preserve progress if a long precompute job is interrupted.
        f.flush()
        os.fsync(f.fileno())


def write_run_metadata(
    output_path: Path,
    metadata: Dict[str, Any],
) -> Path:
    meta_path = output_path.with_suffix(
        output_path.suffix + ".meta.json"
    )

    with meta_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return meta_path


# =============================================================================
# CLI
# =============================================================================

def parse_train_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute AMem local-memory TEXT with batched Unsloth Gemma. "
            "No embeddings are created or stored."
        )
    )

    # Input / output.
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sequences_file",
        type=str,
        default="user_sequences_10_5000.json",
    )
    parser.add_argument(
        "--items_file",
        type=str,
        default="items.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    # Gemma.
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16"],
        default="auto",
    )

    # Memory-window construction.
    parser.add_argument(
        "--number_of_users",
        type=int,
        default=100,
        help="Number of users to precompute; <=0 means all users.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max_train_items",
        type=int,
        default=30,
        help="Match original AMem train[-30:]; <=0 means all train items.",
    )

    # Batch generation.
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=8,
    )

    # Greedy by default for reproducible precomputed artifacts.
    parser.add_argument(
        "--sample",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume an existing NEW-schema JSONL by skipping completed "
            "precompute_id values."
        ),
    )

    return parser.parse_args(argv)


# =============================================================================
# Main
# =============================================================================

def train_main(argv=None) -> None:
    args = parse_train_args(argv)
    global np, tqdm
    import numpy as np
    from tqdm.auto import tqdm
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    sequences_path = data_dir / args.sequences_file
    items_path = data_dir / args.items_file

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("AMem PRECOMPUTE -- LOCAL MEMORY TEXT ONLY")
    print("=" * 80)
    print(f"Sequences       : {sequences_path}")
    print(f"Items           : {items_path}")
    print(f"Output          : {output_path}")
    print(f"Gemma           : {args.model_name}")
    print(f"Users           : {args.number_of_users}")
    print(f"Window size     : {args.window_size}")
    print(f"Max train items : {args.max_train_items}")
    print(f"LLM batch size  : {args.llm_batch_size}")
    print("Embedding model : NONE")
    print("Embedding saved : NO")
    print(f"Schema          : {TRAIN_SCHEMA_VERSION}")

    user_sequences = load_json(sequences_path)
    items_meta = load_json(items_path)

    all_windows = build_all_windows(
        user_sequences=user_sequences,
        items_meta=items_meta,
        number_of_users=args.number_of_users,
        window_size=args.window_size,
        max_train_items=args.max_train_items,
    )

    print(
        f"Prepared {len(all_windows)} local-memory windows"
    )

    # Output behavior:
    # - resume=True: inspect and skip completed IDs
    # - resume=False: start clean instead of accidentally appending duplicates
    if args.resume:
        completed_ids, existing_valid_records = inspect_existing_output(
            output_path
        )
    else:
        completed_ids = set()
        existing_valid_records = 0

        if output_path.exists():
            print(
                f"Resume disabled: truncating existing output {output_path}"
            )
            output_path.unlink()

        old_meta = output_path.with_suffix(
            output_path.suffix + ".meta.json"
        )
        if old_meta.exists():
            old_meta.unlink()

    if completed_ids:
        print(
            f"Resume: {len(completed_ids)} completed precompute IDs "
            f"({existing_valid_records} valid JSONL records)"
        )

    pending = [
        rec
        for rec in all_windows
        if rec["precompute_id"] not in completed_ids
    ]

    if not pending:
        print("Nothing to do: all local memories are already precomputed.")
        return

    print(
        f"Pending local memories: {len(pending)}"
    )

    extractor = BatchedGemmaExtractor(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
    )

    total_llm_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    parse_failures = 0
    consistency_adjustment_records = 0
    consistency_adjustment_events = 0
    produced = 0

    batch_starts = range(
        0,
        len(pending),
        args.llm_batch_size,
    )

    progress = tqdm(
        batch_starts,
        desc="Gemma batches",
    )

    for start in progress:
        batch = pending[
            start:start + args.llm_batch_size
        ]

        prompts = [
            record["prompt"]
            for record in batch
        ]

        responses, batch_stats = extractor.generate_batch(
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.sample,
        )

        total_llm_time += batch_stats["elapsed_sec"]
        total_input_tokens += batch_stats["input_tokens"]
        total_output_tokens += batch_stats["output_tokens_approx"]

        output_records: List[Dict[str, Any]] = []

        for rec, response in zip(
            batch,
            responses,
        ):
            try:
                parsed = parse_json_response(response)

                behavior_explanation = str(
                    parsed.get(
                        "behavior_explanation",
                        "",
                    )
                ).strip()

                recommendation_behavior = str(
                    parsed.get(
                        "recommendation_behavior",
                        "",
                    )
                ).strip()

                semantic_focus = parsed.get(
                    "semantic_focus",
                    [],
                )
                if not isinstance(
                    semantic_focus,
                    list,
                ):
                    semantic_focus = [
                        str(semantic_focus)
                    ] if semantic_focus else []

                semantic_focus = [
                    str(x).strip()
                    for x in semantic_focus
                    if str(x).strip()
                ][:8]

                trajectory_signature = str(
                    parsed.get(
                        "trajectory_signature",
                        "",
                    )
                ).strip()

                mechanism = str(
                    parsed.get(
                        "mechanism",
                        "unknown",
                    )
                ).strip().lower().replace(
                    "_",
                    " ",
                )

                scope = str(
                    parsed.get(
                        "scope",
                        "unknown",
                    )
                ).strip().lower().replace(
                    "_",
                    " ",
                )

                direction = str(
                    parsed.get(
                        "direction",
                        "unknown",
                    )
                ).strip().lower().replace(
                    "_",
                    " ",
                )

                try:
                    confidence = float(
                        parsed.get(
                            "confidence",
                            0.0,
                        )
                    )
                except Exception:
                    confidence = 0.0

                confidence = max(
                    0.0,
                    min(1.0, confidence),
                )

                (
                    mechanism,
                    scope,
                    direction,
                    consistency_adjustments,
                    structural_evidence,
                ) = enforce_structural_consistency(
                    mechanism=mechanism,
                    scope=scope,
                    direction=direction,
                    interaction_sequence=rec[
                        "interaction_sequence"
                    ],
                )

                if not recommendation_behavior:
                    raise ValueError(
                        "Missing recommendation_behavior"
                    )

                if not trajectory_signature:
                    raise ValueError(
                        "Missing trajectory_signature"
                    )

                parse_ok = True
                parse_error = None

            except Exception as exc:
                parse_failures += 1

                # Conservative fallback: retain observed category evidence but
                # mark the trajectory abstraction as unknown.
                categories = [
                    str(
                        x.get(
                            "item_category",
                            "Unknown",
                        )
                    ).strip()
                    for x in rec[
                        "interaction_sequence"
                    ]
                    if str(
                        x.get(
                            "item_category",
                            "",
                        )
                    ).strip()
                ]

                unique_categories = list(
                    dict.fromkeys(categories)
                )

                behavior_explanation = (
                    f"User purchased "
                    f"{len(rec['interaction_sequence'])} observed items."
                )

                if unique_categories:
                    recommendation_behavior = (
                        "Observed preference is concentrated around: "
                        + ", ".join(
                            unique_categories[:4]
                        )
                        + "."
                    )
                else:
                    recommendation_behavior = (
                        "Observed preference could not be reliably summarized."
                    )

                semantic_focus = (
                    unique_categories[:8]
                )
                trajectory_signature = (
                    "uncertain observed behavior"
                )
                mechanism = "unknown"
                scope = "unknown"
                direction = "unknown"
                confidence = 0.0
                consistency_adjustments = []
                structural_evidence = {
                    "num_items": len(
                        rec["interaction_sequence"]
                    ),
                    "num_unique_item_ids": len({
                        str(x.get("item_id"))
                        for x in rec[
                            "interaction_sequence"
                        ]
                    }),
                    "has_exact_item_repeat": False,
                    "all_same_item": False,
                    "same_known_category": False,
                }

                parse_ok = False
                parse_error = repr(exc)

            # -----------------------------------------------------------------
            # Backward-compatibility aliases
            # -----------------------------------------------------------------
            # Existing AMem/global-memory code often expects:
            #   pattern_description + keywords
            # Existing Tree code often expects:
            #   behavior_signature
            #
            # Keep those names, but give them explicit roles:
            #   pattern_description = recommendation-oriented semantic view
            #   keywords            = semantic_focus
            #   behavior_signature  = trajectory_signature
            pattern_description = (
                recommendation_behavior
            )
            keywords = list(
                semantic_focus
            )
            behavior_signature = (
                trajectory_signature
            )

            output_record = {
                "schema_version": TRAIN_SCHEMA_VERSION,
                "behavior_prompt_version": (
                    BEHAVIOR_PROMPT_VERSION
                ),
                "precompute_id": int(
                    rec["precompute_id"]
                ),
                "user_id": rec["user_id"],
                "user_order": int(
                    rec["user_order"]
                ),
                "window_index": int(
                    rec["window_index"]
                ),
                "interaction_sequence": (
                    rec["interaction_sequence"]
                ),

                # Grounded explanation.
                "behavior_explanation": (
                    behavior_explanation
                ),

                # View A: recommendation/ranking semantics.
                "recommendation_behavior": (
                    recommendation_behavior
                ),
                "semantic_focus": (
                    semantic_focus
                ),

                # View B: cross-user trajectory abstraction.
                "trajectory_signature": (
                    trajectory_signature
                ),
                "behavior_signature": (
                    behavior_signature
                ),
                "mechanism": mechanism,
                "scope": scope,
                "direction": direction,
                "confidence": confidence,
                "consistency_adjustments": (
                    consistency_adjustments
                ),
                "structural_evidence": (
                    structural_evidence
                ),

                # Backward-compatible aliases used by existing AMem code.
                "pattern_description": (
                    pattern_description
                ),
                "keywords": keywords,

                "parse_ok": bool(parse_ok),
                "parse_error": parse_error,

                # Useful for auditing/debugging Gemma extraction.
                "raw_response": response,
            }

            if consistency_adjustments:
                consistency_adjustment_records += 1
                consistency_adjustment_events += len(
                    consistency_adjustments
                )

            # Intentionally NO "embedding" field.
            output_records.append(
                output_record
            )

        append_memory_records(
            output_path,
            output_records,
        )

        produced += len(output_records)

        progress.set_postfix({
            "new": produced,
            "parse_fail": parse_failures,
            "sec/batch": (
                f"{batch_stats['elapsed_sec']:.2f}"
            ),
        })

    metadata = {
        "schema_version": TRAIN_SCHEMA_VERSION,
        "behavior_prompt_version": BEHAVIOR_PROMPT_VERSION,
        "behavior_schema": "dual_view_semantic_plus_trajectory",
        "stage": "local_memory_text_precompute",
        "contains_embeddings": False,
        "sequences_file": str(sequences_path),
        "items_file": str(items_path),
        "output_file": str(output_path),
        "model_name": args.model_name,
        "number_of_users": args.number_of_users,
        "window_size": args.window_size,
        "max_train_items": args.max_train_items,
        "llm_batch_size": args.llm_batch_size,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "dtype": args.dtype,
        "sample": args.sample,
        "temperature": (
            args.temperature if args.sample else None
        ),
        "top_p": (
            args.top_p if args.sample else None
        ),
        "top_k": (
            args.top_k if args.sample else None
        ),
        "seed": args.seed,
        "num_windows_total": len(all_windows),
        "num_newly_produced": produced,
        "num_previously_completed": len(
            completed_ids
        ),
        "parse_failures_this_run": (
            parse_failures
        ),
        "consistency_adjustment_records": int(
            consistency_adjustment_records
        ),
        "consistency_adjustment_events": int(
            consistency_adjustment_events
        ),
        "llm_inference_time_sec": round(
            total_llm_time,
            4,
        ),
        "llm_input_tokens": (
            total_input_tokens
        ),
        "llm_output_tokens_approx": (
            total_output_tokens
        ),
    }

    metadata_path = write_run_metadata(
        output_path,
        metadata,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output JSONL    : {output_path}")
    print(f"Metadata        : {metadata_path}")
    print(f"New records     : {produced}")
    print(f"Parse failures  : {parse_failures}")
    print(
        f"Consistency fix : {consistency_adjustment_records} records / "
        f"{consistency_adjustment_events} label adjustments"
    )
    print(f"Gemma time      : {total_llm_time:.2f} sec")
    print("Embeddings      : NOT computed / NOT stored")
    print("\nNext stage:")
    print(
        "load this JSONL -> compose memory texts -> batch-encode embeddings "
        "once -> sequential cosine/link/evolve/store."
    )

#!/usr/bin/env python3




TEST_SCHEMA_VERSION = "amem_test_behavior_precompute_dual_v3"



# =============================================================================
# Generic I/O
# =============================================================================

def json_load(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_dump(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stable_hash(obj: Any) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_label(value: Any) -> str:
    x = str(value or "").strip().lower()
    x = x.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", x).strip()


def load_user_ids_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None

    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            if isinstance(obj.get("users"), list):
                return [str(x) for x in obj["users"]]
            return [str(x) for x in obj.keys()]
    except Exception:
        pass

    return [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]


def processed_users(path: str | Path) -> set[str]:
    """
    Resume only rows generated by the CURRENT schema/prompt.
    This prevents silently mixing old collapsed-taxonomy outputs with v2.
    """
    p = Path(path)
    if not p.exists():
        return set()

    out: set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            if (
                row.get("precompute_ok") is True
                and row.get("schema_version") == TEST_SCHEMA_VERSION
                and row.get("behavior_prompt_version")
                == BEHAVIOR_PROMPT_VERSION
            ):
                out.add(
                    str(row["user_id"])
                )

    return out


# =============================================================================
# Item lookup
# =============================================================================

def resolve_item_key(
    item_id: Any,
    items_meta: Dict[Any, Dict[str, Any]],
) -> Optional[Any]:
    if item_id in items_meta:
        return item_id

    s = str(item_id)
    if s in items_meta:
        return s

    try:
        i = int(s)
        if i in items_meta:
            return i
    except Exception:
        pass

    return None


def load_items(path: str) -> Dict[Any, Dict[str, Any]]:
    raw = json_load(path)

    try:
        return {
            int(k): v
            for k, v in raw.items()
        }
    except Exception:
        return raw


# =============================================================================
# Behavior windows
# SAME rule as current inference/training pipeline
# =============================================================================

@dataclass
class BehaviorTask:
    task_id: str
    window_index: int
    interaction_sequence: List[Dict[str, str]]


def build_behavior_windows(
    train_item_ids: Sequence[Any], items_meta: Dict[Any, Dict[str, Any]],
    window_size: int, max_train_interactions: int,
) -> List[List[Dict[str, str]]]:
    # Keep the test cache's historical slicing and metadata conventions.
    train_items = list(train_item_ids)[-int(max_train_interactions):]
    interactions = []
    for item_id in train_items:
        key = resolve_item_key(item_id, items_meta)
        info = items_meta[key] if key is not None else None
        interactions.append({
            "item_id": str(item_id),
            "item": str(info.get("title", f"Item {item_id}")),
            "category": str(info.get("main_cat", info.get("category", "Unknown"))),
            "action": "purchase",
        } if info is not None else None)
    return interaction_windows(
        interactions, window_size, skip_missing=False, deduplicate=True,
    )


# =============================================================================
# Behavior prompt
# =============================================================================

BEHAVIOR_SYSTEM = """You are a behavioral memory modeling system for recommender systems.

For each short OBSERVED interaction window, produce TWO complementary views:

A) Recommendation behavior:
   Preserve concrete, discriminative semantic preference evidence that can help
   distinguish relevant candidate items from unrelated items.

B) Trajectory abstraction:
   Describe HOW preference evolves in a domain-agnostic form so behaviors can be
   clustered across users and used in collaborative trajectory prediction.

Never predict a future item.
Never invent evidence not supported by the observed interactions.
Return valid JSON only.
"""


def build_single_behavior_prompt(task: BehaviorTask) -> str:
    return build_behavior_extraction_prompt(task.interaction_sequence, task.task_id)


# =============================================================================
# Robust JSON parser
# =============================================================================

def extract_json_value(text: str) -> Any:
    s = str(text or "").strip()

    s = re.sub(
        r"^```(?:json)?\s*",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"\s*```$",
        "",
        s,
    ).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    starts = [
        (s.find("{"), "{", "}"),
        (s.find("["), "[", "]"),
    ]
    starts = [
        x for x in starts
        if x[0] >= 0
    ]

    if not starts:
        raise ValueError(
            "No JSON value found"
        )

    start, opener, closer = min(
        starts,
        key=lambda x: x[0],
    )

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(
                    s[start:i + 1]
                )

    raise ValueError(
        "Unbalanced JSON"
    )






def normalize_behavior(
    obj: Any,
    task: BehaviorTask,
) -> Dict[str, Any]:
    # Be tolerant if the model wraps a single object in a list.
    if isinstance(obj, list):
        rows = [
            x
            for x in obj
            if isinstance(x, dict)
        ]
        if not rows:
            raise ValueError(
                f"No behavior object for {task.task_id}"
            )
        obj = rows[0]

    if not isinstance(obj, dict):
        raise ValueError(
            f"Behavior output is not an object "
            f"for {task.task_id}"
        )

    row = dict(obj)

    recommendation_behavior = str(
        row.get("recommendation_behavior")
        or ""
    ).strip()
    if not recommendation_behavior:
        raise ValueError(
            f"Missing recommendation_behavior "
            f"for {task.task_id}"
        )

    semantic_focus = row.get(
        "semantic_focus",
        [],
    )
    if not isinstance(
        semantic_focus,
        list,
    ):
        semantic_focus = (
            [str(semantic_focus)]
            if semantic_focus
            else []
        )

    semantic_focus = [
        str(x).strip()
        for x in semantic_focus
        if str(x).strip()
    ][:8]

    trajectory_signature = str(
        row.get("trajectory_signature")
        or ""
    ).strip()
    if not trajectory_signature:
        raise ValueError(
            f"Missing trajectory_signature "
            f"for {task.task_id}"
        )

    try:
        confidence = float(
            row.get("confidence", 0.0)
        )
    except Exception:
        confidence = 0.0

    mechanism = normalize_label(
        row.get("mechanism")
        or "unknown"
    )
    scope = normalize_label(
        row.get("scope")
        or "unknown"
    )
    direction = normalize_label(
        row.get("direction")
        or "unknown"
    )

    (
        mechanism,
        scope,
        direction,
        consistency_adjustments,
        structural_evidence,
    ) = test_structural_consistency(
        mechanism=mechanism,
        scope=scope,
        direction=direction,
        interaction_sequence=(
            task.interaction_sequence
        ),
    )

    # -----------------------------------------------------------------
    # Backward compatibility:
    # - pattern_description is the ranking-oriented semantic view.
    # - keywords is the semantic_focus list.
    # - behavior_signature is the Tree-oriented trajectory_signature.
    # -----------------------------------------------------------------
    pattern_description = (
        recommendation_behavior
    )
    keywords = list(
        semantic_focus
    )
    behavior_signature = (
        trajectory_signature
    )

    return {
        "task_id": task.task_id,
        "window_index": int(
            task.window_index
        ),
        "behavior_prompt_version": (
            BEHAVIOR_PROMPT_VERSION
        ),

        "behavior_explanation": str(
            row.get(
                "behavior_explanation",
                "",
            )
        ).strip(),

        # View A: content-aware recommendation signal.
        "recommendation_behavior": (
            recommendation_behavior
        ),
        "semantic_focus": (
            semantic_focus
        ),

        # View B: domain-agnostic trajectory signal.
        "trajectory_signature": (
            trajectory_signature
        ),
        "behavior_signature": (
            behavior_signature
        ),
        "mechanism": mechanism,
        "scope": scope,
        "direction": direction,
        "confidence": max(
            0.0,
            min(1.0, confidence),
        ),
        "consistency_adjustments": (
            consistency_adjustments
        ),
        "structural_evidence": (
            structural_evidence
        ),

        # Compatibility aliases used by older downstream code.
        "pattern_description": (
            pattern_description
        ),
        "keywords": keywords,
    }


# =============================================================================
# Local Unsloth Gemma
# =============================================================================

@dataclass
class CallStat:
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    parse_ok: bool
    attempts: int


class LocalGemmaBehaviorExtractor:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
        temperature: float,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        import torch
        from unsloth import FastModel

        self.torch = torch
        self.model_name = model_name
        self.max_seq_length = int(max_seq_length)
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.calls: List[CallStat] = []

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(
            f"[INFO] Loading local Gemma: "
            f"{model_name}"
        )

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=False,
            full_finetuning=False,
        )

        self.model.eval()

        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _render(
        self,
        prompt: str,
    ) -> str:
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": BEHAVIOR_SYSTEM + "\n\n" + prompt,
            }],
        }]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generation_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if self.temperature > 0:
            kwargs.update({
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": 0.95,
            })
        else:
            kwargs["do_sample"] = False

        return kwargs

    def _generate_single_once(
        self,
        task: BehaviorTask,
        retry_hint: bool,
    ) -> Dict[str, Any]:
        prompt = build_single_behavior_prompt(task)

        if retry_hint:
            prompt += """
IMPORTANT RETRY:
The previous response could not be parsed.
Return exactly ONE complete JSON object.
Do not output any commentary before or after the JSON.
"""

        rendered = self._render(prompt)

        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.model.device)

        input_len = int(inputs["input_ids"].shape[1])
        kwargs = self._generation_kwargs()

        t0 = time.time()

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **kwargs,
            )

        generated = output[:, input_len:]

        raw = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        obj = extract_json_value(raw)
        row = normalize_behavior(obj, task)

        self.calls.append(
            CallStat(
                input_tokens=input_len,
                output_tokens=int(generated.shape[1]),
                elapsed_sec=float(time.time() - t0),
                parse_ok=True,
                attempts=1,
            )
        )

        return row

    def generate_single(
        self,
        task: BehaviorTask,
        max_attempts: int,
    ) -> Dict[str, Any]:
        """Robust fallback for one failed sample."""
        last_error: Optional[Exception] = None

        for attempt in range(
            1,
            max(1, int(max_attempts)) + 1,
        ):
            try:
                return self._generate_single_once(
                    task,
                    retry_hint=(attempt > 1),
                )
            except Exception as e:
                last_error = e
                print(
                    f"[WARN] {task.task_id} "
                    f"attempt {attempt}/{max_attempts} failed: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

        raise ValueError(
            f"Behavior generation failed "
            f"for {task.task_id}: {last_error}"
        )

    def _generate_batch_once(
        self,
        tasks: List[BehaviorTask],
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Generate multiple independent prompts in one forward generation call.

        Each task has its own prompt and output sequence. No JSON array is used.
        Parse failures are returned as None and retried individually later.
        """
        if not tasks:
            return []

        rendered = [
            self._render(
                build_single_behavior_prompt(task)
            )
            for task in tasks
        ]

        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.model.device)

        # With left padding, all generated tokens begin after the common padded
        # input width.
        input_width = int(inputs["input_ids"].shape[1])
        kwargs = self._generation_kwargs()

        t0 = time.time()

        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **kwargs,
            )

        generated = outputs[:, input_width:]

        raws = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        elapsed = float(time.time() - t0)
        batch_size = len(tasks)

        results: List[Optional[Dict[str, Any]]] = []

        for task, raw, gen_row in zip(
            tasks,
            raws,
            generated,
        ):
            parse_ok = False
            try:
                obj = extract_json_value(raw.strip())
                row = normalize_behavior(obj, task)
                parse_ok = True
                results.append(row)
            except Exception as e:
                print(
                    f"[WARN] batch parse failed for {task.task_id}: "
                    f"{type(e).__name__}: {e}; retrying individually",
                    flush=True,
                )
                results.append(None)

            # Token count here uses padded input width. It is sufficient for
            # throughput diagnostics and avoids expensive per-sample recounting.
            self.calls.append(
                CallStat(
                    input_tokens=input_width,
                    output_tokens=int(gen_row.shape[0]),
                    elapsed_sec=elapsed / max(1, batch_size),
                    parse_ok=parse_ok,
                    attempts=1,
                )
            )

        return results

    def generate_batch(
        self,
        tasks: List[BehaviorTask],
        batch_size: int,
        max_attempts: int,
    ) -> List[Dict[str, Any]]:
        """
        Batched independent-prompt generation.

        Normal path:
            B independent prompts -> 1 model.generate() -> B independent JSONs.

        If one output fails parsing, only that task is retried individually.
        """
        if not tasks:
            return []

        batch_size = max(1, int(batch_size))
        final_rows: List[Optional[Dict[str, Any]]] = [None] * len(tasks)

        for start in range(0, len(tasks), batch_size):
            end = min(
                start + batch_size,
                len(tasks),
            )
            batch_tasks = tasks[start:end]

            try:
                batch_rows = self._generate_batch_once(
                    batch_tasks
                )
            except Exception as e:
                print(
                    f"[WARN] model.generate batch "
                    f"{start}:{end} failed: "
                    f"{type(e).__name__}: {e}; "
                    "retrying this batch sample-by-sample",
                    flush=True,
                )
                batch_rows = [None] * len(batch_tasks)

            for local_idx, (task, row) in enumerate(
                zip(batch_tasks, batch_rows)
            ):
                global_idx = start + local_idx

                if row is not None:
                    final_rows[global_idx] = row
                    continue

                # Only failed samples pay the slower single-sample retry cost.
                final_rows[global_idx] = self.generate_single(
                    task,
                    max_attempts=max_attempts,
                )

        if any(row is None for row in final_rows):
            raise RuntimeError(
                "Internal error: unresolved behavior row after retries"
            )

        return [
            row
            for row in final_rows
            if row is not None
        ]

    def stats(self) -> Dict[str, Any]:
        if not self.calls:
            return {
                "provider": "local_unsloth",
                "model": self.model_name,
                "calls": 0,
            }

        return {
            "provider": "local_unsloth",
            "model": self.model_name,
            "generation_records": len(self.calls),
            "successful_generation_records": int(
                sum(1 for x in self.calls if x.parse_ok)
            ),
            "parse_failures_before_retry": int(
                sum(1 for x in self.calls if not x.parse_ok)
            ),
            "input_tokens_approx": int(
                sum(x.input_tokens for x in self.calls)
            ),
            "output_tokens": int(
                sum(x.output_tokens for x in self.calls)
            ),
            "elapsed_sec": float(
                sum(x.elapsed_sec for x in self.calls)
            ),
        }


# =============================================================================
# Summary
# =============================================================================

def summarize_output(
    output_path: str | Path,
    extractor_stats: Dict[str, Any],
    failed_this_run: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    success_rows: List[
        Dict[str, Any]
    ] = []

    p = Path(output_path)
    if p.exists():
        with p.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                if (
                    row.get("precompute_ok")
                    is True
                ):
                    success_rows.append(row)

    mechanism_counts = Counter()
    scope_counts = Counter()
    direction_counts = Counter()
    semantic_focus_counts = Counter()
    recommendation_behavior_lengths: List[int] = []
    trajectory_signature_lengths: List[int] = []
    confidences: List[float] = []
    num_behaviors: List[int] = []
    consistency_adjustment_records = 0
    consistency_adjustment_events = 0

    for row in success_rows:
        behaviors = row.get(
            "generated_behaviors",
            [],
        )

        num_behaviors.append(
            len(behaviors)
        )

        for b in behaviors:
            adjustments = b.get(
                "consistency_adjustments",
                [],
            )
            if adjustments:
                consistency_adjustment_records += 1
                consistency_adjustment_events += len(
                    adjustments
                )

            mechanism_counts[
                str(b.get("mechanism"))
            ] += 1
            scope_counts[
                str(b.get("scope"))
            ] += 1
            direction_counts[
                str(b.get("direction"))
            ] += 1

            for sf in b.get(
                "semantic_focus",
                [],
            ):
                semantic_focus_counts[
                    str(sf)
                ] += 1

            recommendation_behavior_lengths.append(
                len(
                    str(
                        b.get(
                            "recommendation_behavior",
                            "",
                        )
                    )
                )
            )
            trajectory_signature_lengths.append(
                len(
                    str(
                        b.get(
                            "trajectory_signature",
                            "",
                        )
                    )
                )
            )

            if (
                b.get("confidence")
                is not None
            ):
                confidences.append(
                    float(
                        b["confidence"]
                    )
                )

    return {
        "schema_version": (
            "amem_test_behavior_precompute_dual_summary_v3"
        ),
        "behavior_prompt_version": (
            BEHAVIOR_PROMPT_VERSION
        ),
        "num_users_success": len(
            success_rows
        ),
        "num_behaviors": int(
            sum(num_behaviors)
        ),
        "mean_behaviors_per_user": (
            float(
                np.mean(num_behaviors)
            )
            if num_behaviors
            else 0.0
        ),
        "mean_confidence": (
            float(
                np.mean(confidences)
            )
            if confidences
            else None
        ),
        "consistency_adjustment_records": int(
            consistency_adjustment_records
        ),
        "consistency_adjustment_events": int(
            consistency_adjustment_events
        ),
        "top_mechanisms": [
            [k, int(v)]
            for k, v
            in mechanism_counts.most_common()
        ],
        "top_scopes": [
            [k, int(v)]
            for k, v
            in scope_counts.most_common()
        ],
        "top_directions": [
            [k, int(v)]
            for k, v
            in direction_counts.most_common()
        ],
        "top_semantic_focus": [
            [k, int(v)]
            for k, v
            in semantic_focus_counts.most_common(30)
        ],
        "mean_recommendation_behavior_chars": (
            float(
                np.mean(
                    recommendation_behavior_lengths
                )
            )
            if recommendation_behavior_lengths
            else 0.0
        ),
        "mean_trajectory_signature_chars": (
            float(
                np.mean(
                    trajectory_signature_lengths
                )
            )
            if trajectory_signature_lengths
            else 0.0
        ),
        "failed_this_run": int(
            failed_this_run
        ),
        "local_gemma": (
            extractor_stats
        ),
        "config": config,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_test_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Standalone precompute of "
            "test-user behaviors with "
            "local Unsloth Gemma."
        )
    )

    p.add_argument(
        "--items",
        required=True,
    )
    p.add_argument(
        "--sequences",
        required=True,
    )

    p.add_argument(
        "--model",
        default=(
            "unsloth/gemma-3-4b-it-"
            "unsloth-bnb-4bit"
        ),
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
    )
    p.add_argument(
        "--load-in-4bit",
        action=(
            argparse.BooleanOptionalAction
        ),
        default=True,
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=2,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Number of independent behavior-window prompts per "
            "model.generate() call. Start with 8; lower if GPU OOM."
        ),
    )

    p.add_argument(
        "--window-size",
        type=int,
        default=3,
    )
    p.add_argument(
        "--max-train-interactions",
        type=int,
        default=30,
    )

    p.add_argument(
        "--recent-behaviors",
        type=int,
        default=5,
        help=(
            "How many latest behaviors to "
            "also store in the compact "
            "ranking evidence block."
        ),
    )

    p.add_argument(
        "--user-ids-file",
        default=None,
    )
    p.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="0 = all selected users",
    )

    p.add_argument(
        "--output",
        required=True,
    )
    p.add_argument(
        "--summary-output",
        default=None,
    )
    p.add_argument(
        "--failures-output",
        default=None,
    )
    p.add_argument(
        "--resume",
        action=(
            argparse.BooleanOptionalAction
        ),
        default=False,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args(argv)


# =============================================================================
# Main
# =============================================================================

def test_main(argv=None) -> None:
    args = parse_test_args(argv)
    global np, tqdm
    import numpy as np
    from tqdm.auto import tqdm

    random.seed(args.seed)
    np.random.seed(args.seed)

    items_meta = load_items(
        args.items
    )

    raw_sequences = json_load(
        args.sequences
    )
    sequences = {
        str(k): v
        for k, v in raw_sequences.items()
    }

    requested_users = (
        load_user_ids_file(
            args.user_ids_file
        )
    )

    if requested_users is None:
        users = list(
            sequences.keys()
        )
    else:
        users = [
            u
            for u in requested_users
            if u in sequences
        ]

    if args.max_users > 0:
        users = users[
            :args.max_users
        ]

    output = Path(args.output)
    summary_output = Path(
        args.summary_output
        or (str(output) + ".summary.json")
    )
    failures_output = Path(
        args.failures_output
        or (str(output) + ".failures.jsonl")
    )

    if (
        output.exists()
        and not args.resume
    ):
        raise RuntimeError(
            f"{output} already exists. "
            "Delete it or use --resume."
        )

    done = (
        processed_users(output)
        if args.resume
        else set()
    )

    pending = [
        uid
        for uid in users
        if uid not in done
    ]

    extractor = (
        LocalGemmaBehaviorExtractor(
            model_name=args.model,
            max_seq_length=(
                args.max_seq_length
            ),
            load_in_4bit=(
                args.load_in_4bit
            ),
            temperature=(
                args.temperature
            ),
            max_new_tokens=(
                args.max_new_tokens
            ),
            seed=args.seed,
        )
    )

    config = {
        "items": args.items,
        "sequences": args.sequences,
        "model": args.model,
        "schema_version": TEST_SCHEMA_VERSION,
        "behavior_prompt_version": BEHAVIOR_PROMPT_VERSION,
        "behavior_schema": "dual_view_semantic_plus_trajectory",
        "window_size": (
            args.window_size
        ),
        "max_train_interactions": (
            args.max_train_interactions
        ),
        "recent_behaviors": (
            args.recent_behaviors
        ),
        "max_new_tokens": (
            args.max_new_tokens
        ),
        "max_attempts": (
            args.max_attempts
        ),
        "batch_size": (
            args.batch_size
        ),
        "temperature": (
            args.temperature
        ),
        "seed": args.seed,
    }

    print("=" * 90)
    print(
        "STANDALONE TEST-USER "
        "BEHAVIOR PRECOMPUTE"
    )
    print("=" * 90)
    print(
        f"selected users     : "
        f"{len(users)}"
    )
    print(
        f"already completed : "
        f"{len(users) - len(pending)}"
    )
    print(
        f"pending           : "
        f"{len(pending)}"
    )
    print(
        f"window size       : "
        f"{args.window_size}"
    )
    print(
        f"batch size        : "
        f"{args.batch_size}"
    )
    print(
        f"max train history : "
        f"{args.max_train_interactions}"
    )
    print(
        f"output            : "
        f"{output}"
    )

    failed_this_run = 0

    pbar = tqdm(
        pending,
        desc="Precompute test behaviors",
        unit="user",
        dynamic_ncols=True,
    )

    for uid in pbar:
        try:
            user_data = sequences[uid]

            train_ids = list(
                user_data.get(
                    "train",
                    [],
                )
            )

            used_train_ids = train_ids[
                -args.max_train_interactions:
            ]

            windows = (
                build_behavior_windows(
                    train_item_ids=train_ids,
                    items_meta=items_meta,
                    window_size=(
                        args.window_size
                    ),
                    max_train_interactions=(
                        args.max_train_interactions
                    ),
                )
            )

            tasks = [
                BehaviorTask(
                    task_id=f"{uid}_W{i}",
                    window_index=i,
                    interaction_sequence=w,
                )
                for i, w in enumerate(
                    windows
                )
            ]

            behaviors = extractor.generate_batch(
                tasks=tasks,
                batch_size=args.batch_size,
                max_attempts=args.max_attempts,
            )

            recent = []

            for b in behaviors[
                -args.recent_behaviors:
            ]:
                recent.append({
                    "window_index": (
                        b["window_index"]
                    ),

                    # Ranking-oriented semantic evidence.
                    "recommendation_behavior": (
                        b[
                            "recommendation_behavior"
                        ]
                    ),
                    "semantic_focus": (
                        b["semantic_focus"]
                    ),

                    # Tree-oriented trajectory abstraction.
                    "trajectory_signature": (
                        b[
                            "trajectory_signature"
                        ]
                    ),
                    "behavior_signature": (
                        b[
                            "behavior_signature"
                        ]
                    ),
                    "mechanism": (
                        b["mechanism"]
                    ),
                    "scope": b["scope"],
                    "direction": (
                        b["direction"]
                    ),
                    "confidence": (
                        b["confidence"]
                    ),
                })

            row = {
                "schema_version": (
                    TEST_SCHEMA_VERSION
                ),
                "behavior_prompt_version": (
                    BEHAVIOR_PROMPT_VERSION
                ),
                "behavior_schema": (
                    "dual_view_semantic_plus_trajectory"
                ),
                "precompute_ok": True,
                "user_id": uid,
                "train_history_item_ids": [
                    str(x)
                    for x in used_train_ids
                ],
                "train_history_hash": (
                    stable_hash([
                        str(x)
                        for x in used_train_ids
                    ])
                ),
                "window_size": (
                    args.window_size
                ),
                "max_train_interactions": (
                    args.max_train_interactions
                ),
                "behavior_windows": (
                    windows
                ),
                "generated_behaviors": (
                    behaviors
                ),
                "recent_behavior_evidence": (
                    recent
                ),
            }

            append_jsonl(
                output,
                row,
            )

            pbar.set_postfix(
                behaviors=len(behaviors),
                refresh=False,
            )

        except Exception as e:
            failed_this_run += 1

            failure = {
                "schema_version": (
                    TEST_SCHEMA_VERSION
                ),
                "behavior_prompt_version": (
                    BEHAVIOR_PROMPT_VERSION
                ),
                "precompute_ok": False,
                "user_id": uid,
                "error_type": (
                    type(e).__name__
                ),
                "error": str(e),
            }

            append_jsonl(
                failures_output,
                failure,
            )

            print(
                f"\n[WARN] user={uid} "
                f"failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

    pbar.close()

    summary = summarize_output(
        output_path=output,
        extractor_stats=(
            extractor.stats()
        ),
        failed_this_run=(
            failed_this_run
        ),
        config=config,
    )

    json_dump(
        summary_output,
        summary,
    )

    print("\nSummary")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"\nOutput   : {output}"
    )
    print(
        f"Failures : "
        f"{failures_output}"
    )
    print(
        f"Summary  : "
        f"{summary_output}"
    )




def test_structural_consistency(mechanism, scope, direction, interaction_sequence):
    return enforce_structural_consistency(
        mechanism, scope, direction, interaction_sequence,
        category_key="category", normalize_labels=True,
    )

def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "test"))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return
    mode = parser.parse_args(argv[:1]).mode
    {"train": train_main, "test": test_main}[mode](argv[1:])


if __name__ == "__main__":
    main()
