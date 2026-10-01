from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class PromptEncoding:
    ids: list[int]
    history_slots: list[int]
    candidate_slots: list[int]


@dataclass
class RankingGeneration:
    raw_output: str
    order_1based: list[int]
    parse_ok: bool
    parse_reason: str


class GemmaBridge(nn.Module):
    """Frozen Gemma-3 bridge with A-LLMRec embedding injection.

    Stage-2 training keeps the original A-LLMRec-style single-target LM objective.
    Ranking evaluation is adapted to CoMemTree-style full-list generation:
    Gemma must output a permutation of candidate indices 1..N, which is then parsed.
    No candidate-likelihood scoring is used at evaluation time.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        max_seq_length: int = 8192,
        load_in_4bit: bool = True,
        backend: str = "unsloth",
    ):
        super().__init__()
        self.device_name = device
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.backend = backend

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(torch.device(device))

        if backend == "unsloth":
            try:
                from unsloth import FastModel
            except ImportError as e:
                raise ImportError("backend=unsloth requires `pip install unsloth`") from e
            self.model, tok = FastModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                full_finetuning=False,
            )
            self.tokenizer = getattr(tok, "tokenizer", tok)
        elif backend == "transformers":
            try:
                from transformers import AutoProcessor, Gemma3ForConditionalGeneration
            except ImportError as e:
                raise ImportError("backend=transformers requires transformers>=4.51") from e
            processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = getattr(processor, "tokenizer", processor)
            dtype = torch.float16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=dtype,
            )
        else:
            raise ValueError("backend must be `unsloth` or `transformers`")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", None)
        self.hidden_size = int(getattr(text_cfg, "hidden_size", getattr(cfg, "hidden_size", 0)))
        if self.hidden_size <= 0:
            emb = self.model.get_input_embeddings()
            self.hidden_size = int(emb.weight.shape[-1])

        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        if self.pad_id is None:
            self.pad_id = 0
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.placeholder_id = self.pad_id

    def _tok(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _prompt_prefix(
        self,
        history_titles: Sequence[str],
        candidate_titles: Sequence[str],
        *,
        bracket_candidate_indices: bool = False,
    ) -> tuple[list[int], list[int], list[int]]:
        ids: list[int] = []
        hslots: list[int] = []
        cslots: list[int] = []
        if self.bos_id is not None:
            ids.append(int(self.bos_id))

        def add_text(t: str) -> None:
            ids.extend(self._tok(t))

        add_text(
            "A collaborative user representation is provided before this prompt. "
            "Use both that representation and the interaction evidence below.\n"
            "Recent interactions (oldest to newest):\n"
        )
        for j, title in enumerate(history_titles, 1):
            add_text(f"{j}. {title} ")
            hslots.append(len(ids))
            ids.append(self.placeholder_id)
            add_text("\n")

        add_text("Candidate items:\n")
        for j, title in enumerate(candidate_titles, 1):
            if bracket_candidate_indices:
                # Use a dedicated candidate label namespace during ranking
                # evaluation so numbers inside titles (years, editions, versions,
                # item IDs, etc.) cannot be confused with candidate positions.
                add_text(f"[C{j:02d}] {title} ")
            else:
                # Preserve the original Stage-2 training prompt format.
                add_text(f"{j}. {title} ")
            cslots.append(len(ids))
            ids.append(self.placeholder_id)
            add_text("\n")

        return ids, hslots, cslots

    def make_prompt(
        self,
        history_titles: Sequence[str],
        candidate_titles: Sequence[str],
    ) -> PromptEncoding:
        """Stage-2 TRAINING prompt.

        Keep the original A-LLMRec-style single-target objective.  The target title
        is appended by ``training_loss`` and is the last TRAIN interaction, never
        the test GT.
        """
        ids, hslots, cslots = self._prompt_prefix(
            history_titles, candidate_titles, bracket_candidate_indices=False
        )
        ids.extend(
            self._tok(
                "Recommend exactly one next item from the candidate set. "
                "Output only the title of the recommended item.\n"
                "The recommendation is: "
            )
        )
        if len(ids) + 1 > self.max_seq_length:
            raise ValueError(f"Prompt token length {len(ids)} exceeds max_seq_length={self.max_seq_length}")
        return PromptEncoding(ids, hslots, cslots)

    def make_ranking_prompt(
        self,
        history_titles: Sequence[str],
        candidate_titles: Sequence[str],
    ) -> PromptEncoding:
        """Inference prompt for full candidate ranking.

        The model must emit every candidate INDEX exactly once.  Using candidate
        indices avoids title/ASIN parsing ambiguity and mirrors CoMemTree's
        generated-permutation evaluation style.
        """
        n = len(candidate_titles)
        ids, hslots, cslots = self._prompt_prefix(
            history_titles, candidate_titles, bracket_candidate_indices=True
        )
        # A concrete, non-identity permutation example helps Gemma-3-4B obey
        # the output grammar without encouraging it to copy the input order.
        if n == 20:
            example = (
                "[C07, C03, C12, C01, C18, C05, C09, C14, C02, C20, "
                "C11, C06, C16, C04, C13, C08, C19, C10, C15, C17]"
            )
        else:
            # Generic format-only example for other candidate sizes.
            example_order = list(range(2, n + 1, 2)) + list(range(1, n + 1, 2))
            example = "[" + ", ".join(f"C{x:02d}" for x in example_order) + "]"

        ranking_instruction = (
            f"Your task is to rank ALL {n} candidate items from the most preferred "
            "to the least preferred for this user.\n\n"
            "STRICT OUTPUT REQUIREMENTS:\n"
            f"- Include every candidate label from C01 to C{n:02d} exactly once.\n"
            f"- Use ONLY candidate labels C01, C02, ..., C{n:02d}.\n"
            "- Do NOT output bare integers such as 1, 2, 3.\n"
            "- Do NOT repeat any candidate label.\n"
            "- Do NOT omit any candidate label.\n"
            "- Do NOT output item titles.\n"
            "- Do NOT output explanations, reasoning, code fences, drafts, or a second answer.\n"
            "- Produce exactly ONE comma-separated ranking.\n\n"
            "Example of the REQUIRED FORMAT ONLY:\n"
            f"{example}\n\n"
            "The example above demonstrates only the output format. "
            "DO NOT copy its ordering. Determine the ranking from the user history, "
            "collaborative representation, and candidate items.\n\n"
            f"Your ranking must contain exactly {n} candidate labels.\n"
            "The opening square bracket has ALREADY been provided below. "
            "Continue immediately with the first candidate label (for example C07), "
            "then commas, and finish with one closing square bracket. "
            "Do not produce a second ranking.\n"
            "Ranking: ["
        )
        ids.extend(self._tok(ranking_instruction))
        if len(ids) + 1 > self.max_seq_length:
            raise ValueError(f"Prompt token length {len(ids)} exceeds max_seq_length={self.max_seq_length}")
        return PromptEncoding(ids, hslots, cslots)

    def _embed_prompt(
        self,
        enc: PromptEncoding,
        user_emb: torch.Tensor,
        hist_embs: torch.Tensor,
        cand_embs: torch.Tensor,
    ) -> torch.Tensor:
        device = user_emb.device
        ids = torch.tensor(enc.ids, dtype=torch.long, device=device).unsqueeze(0)
        base = self.model.get_input_embeddings()(ids).squeeze(0)
        dtype = base.dtype
        base = base.clone()
        if len(enc.history_slots) != hist_embs.shape[0]:
            raise ValueError("history slot count != history embedding count")
        if len(enc.candidate_slots) != cand_embs.shape[0]:
            raise ValueError("candidate slot count != candidate embedding count")
        if enc.history_slots:
            base[torch.tensor(enc.history_slots, device=device)] = hist_embs.to(dtype)
        if enc.candidate_slots:
            base[torch.tensor(enc.candidate_slots, device=device)] = cand_embs.to(dtype)
        return torch.cat([user_emb.to(dtype).view(1, -1), base], dim=0)

    def training_loss(
        self,
        prompts: Sequence[PromptEncoding],
        user_embs: torch.Tensor,
        history_embs: Sequence[torch.Tensor],
        candidate_embs: Sequence[torch.Tensor],
        target_titles: Sequence[str],
    ) -> torch.Tensor:
        samples = []
        labels = []
        for i, enc in enumerate(prompts):
            prompt_e = self._embed_prompt(enc, user_embs[i], history_embs[i], candidate_embs[i])
            target_ids = self._tok(target_titles[i])
            if self.eos_id is not None:
                target_ids = target_ids + [int(self.eos_id)]
            if not target_ids:
                raise ValueError("Target title tokenized to empty sequence")
            if prompt_e.shape[0] + len(target_ids) > self.max_seq_length:
                raise ValueError("Prompt + target exceeds max_seq_length")
            tids = torch.tensor(target_ids, dtype=torch.long, device=prompt_e.device)
            target_e = self.model.get_input_embeddings()(tids)
            emb = torch.cat([prompt_e, target_e], dim=0)
            lab = torch.full((emb.shape[0],), -100, dtype=torch.long, device=emb.device)
            lab[prompt_e.shape[0] :] = tids
            samples.append(emb)
            labels.append(lab)

        maxlen = max(x.shape[0] for x in samples)
        dim = samples[0].shape[-1]
        device = samples[0].device
        dtype = samples[0].dtype
        pad_e = self.model.get_input_embeddings()(
            torch.tensor([self.pad_id], device=device)
        ).squeeze(0).to(dtype)
        batch_e = pad_e.view(1, 1, dim).expand(len(samples), maxlen, dim).clone()
        attn = torch.zeros(len(samples), maxlen, dtype=torch.long, device=device)
        batch_l = torch.full((len(samples), maxlen), -100, dtype=torch.long, device=device)
        for i, (emb, lab) in enumerate(zip(samples, labels)):
            n = emb.shape[0]
            batch_e[i, :n] = emb
            attn[i, :n] = 1
            batch_l[i, :n] = lab

        outputs = self.model(
            inputs_embeds=batch_e,
            attention_mask=attn,
            labels=batch_l,
            use_cache=False,
            return_dict=True,
        )
        return outputs.loss

    @staticmethod
    def parse_full_ranking(raw_output: str, num_candidates: int) -> RankingGeneration:
        """Parse the first generated Cxx ranking and normalize duplicate/missing labels.

        Candidate labels are C01, C02, ..., C{N}. Using a dedicated "C" prefix
        prevents years, editions, version numbers, numeric MovieLens IDs, etc.
        from being interpreted as candidate positions.

        Policy:
        - If at least one valid Cxx label can be extracted from the first generated
          line, parsing is considered successful.
        - Keep the FIRST occurrence of each valid label.
        - Ignore repeated labels and out-of-range labels (e.g. C99 for N=20).
        - Append missing candidate positions in the ORIGINAL frozen candidate order
          (C01, C02, ..., C{N}).
        - Only return parse_ok=False when no valid Cxx label can be extracted.

        ``order_1based`` remains integer positions 1..N so the rest of the project
        does not need to change.
        """
        text = (raw_output or "").strip()
        if not text:
            return RankingGeneration(text, [], False, "empty_output")

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return RankingGeneration(text, [], False, "empty_output")

        first = lines[0]

        while first.startswith("["):
            first = first[1:].lstrip()
        if "]" in first:
            first = first.split("]", 1)[0].strip()

        # Accept C01/C1/c01 with optional whitespace after C, but DO NOT parse
        # bare integers. This protects against years, editions and numeric item IDs.
        raw_labels = re.findall(r"(?i)\bC\s*0*(\d{1,3})\b", first)
        nums = [int(x) for x in raw_labels]

        seen = set()
        cleaned: list[int] = []
        duplicate_found = False
        invalid_found = False

        for x in nums:
            if not (1 <= x <= num_candidates):
                invalid_found = True
                continue
            if x in seen:
                duplicate_found = True
                continue
            cleaned.append(x)
            seen.add(x)

        if not cleaned:
            return RankingGeneration(
                text,
                [],
                False,
                "no_valid_candidate_labels",
            )

        missing = [
            x for x in range(1, num_candidates + 1)
            if x not in seen
        ]
        cleaned.extend(missing)

        reasons = []
        if duplicate_found:
            reasons.append("duplicate")
        if invalid_found:
            reasons.append("invalid")
        if missing:
            reasons.append("missing")

        if reasons:
            reason = "repaired_" + "_and_".join(reasons)
        else:
            reason = "ok_first_generated_ranking"

        assert len(cleaned) == num_candidates
        assert len(set(cleaned)) == num_candidates
        assert set(cleaned) == set(range(1, num_candidates + 1))

        return RankingGeneration(
            text,
            cleaned,
            True,
            reason,
        )

    @torch.no_grad()
    def generate_ranking_batch(
        self,
        prompts: Sequence[PromptEncoding],
        user_embs: torch.Tensor,
        history_embs: Sequence[torch.Tensor],
        candidate_embs: Sequence[torch.Tensor],
        num_candidates: Sequence[int],
        max_new_tokens: int = 1024,
        debug: bool = False,
    ) -> list[RankingGeneration]:
        """Generate full rankings for a batch of users in one Gemma ``generate`` call.

        Prompts are LEFT-padded in embedding space because Gemma is decoder-only:
        the final non-padding position must be the end of each user's prompt.

        Parsing remains per-user and strict:
        - valid full permutation -> keep Gemma's order unchanged;
        - duplicate/missing/out-of-range/format failure -> parse_ok=False.
        The caller is responsible for falling back to that user's frozen candidate
        order.  No ranking repair is performed here.
        """
        bsz = len(prompts)
        if bsz == 0:
            return []
        if not (
            user_embs.shape[0] == bsz
            and len(history_embs) == bsz
            and len(candidate_embs) == bsz
            and len(num_candidates) == bsz
        ):
            raise ValueError("Batch component sizes do not match")

        prompt_es: list[torch.Tensor] = []
        for i in range(bsz):
            prompt_e = self._embed_prompt(
                prompts[i],
                user_embs[i],
                history_embs[i],
                candidate_embs[i],
            )
            if prompt_e.shape[0] >= self.max_seq_length:
                raise ValueError(
                    f"Ranking prompt {i} reaches max_seq_length "
                    f"({prompt_e.shape[0]} >= {self.max_seq_length})"
                )
            prompt_es.append(prompt_e)

        max_prompt_len = max(x.shape[0] for x in prompt_es)
        generation_budget = int(
            min(max_new_tokens, self.max_seq_length - max_prompt_len)
        )
        if generation_budget <= 0:
            raise ValueError("No generation budget left after batched ranking prompts")

        device = prompt_es[0].device
        dtype = prompt_es[0].dtype
        dim = prompt_es[0].shape[-1]

        pad_e = self.model.get_input_embeddings()(
            torch.tensor([self.pad_id], dtype=torch.long, device=device)
        ).squeeze(0).to(dtype)

        # LEFT padding is intentional for decoder-only generation.
        batch_e = pad_e.view(1, 1, dim).expand(
            bsz, max_prompt_len, dim
        ).clone()
        attention_mask = torch.zeros(
            (bsz, max_prompt_len),
            dtype=torch.long,
            device=device,
        )

        for i, emb in enumerate(prompt_es):
            n = emb.shape[0]
            start = max_prompt_len - n
            batch_e[i, start:] = emb
            attention_mask[i, start:] = 1

        generated = self.model.generate(
            inputs_embeds=batch_e,
            attention_mask=attention_mask,
            max_new_tokens=generation_budget,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.pad_id,
            eos_token_id=self.eos_id,
        )

        if generated.shape[0] != bsz:
            raise RuntimeError(
                f"Gemma returned batch={generated.shape[0]}, expected {bsz}"
            )

        results: list[RankingGeneration] = []
        for i in range(bsz):
            raw = self.tokenizer.decode(
                generated[i],
                skip_special_tokens=True,
            ).strip()
            result = self.parse_full_ranking(raw, int(num_candidates[i]))
            results.append(result)

            if debug:
                nonempty_lines = [
                    line.strip()
                    for line in raw.splitlines()
                    if line.strip()
                ]
                first_line = nonempty_lines[0] if nonempty_lines else ""
                print(f"\n[GemmaBridge ranking debug batch_index={i}]")
                print("RAW_GENERATED:", repr(raw))
                print("FIRST_GENERATED_LINE:", repr(first_line))
                print("PARSED_ORDER:", result.order_1based)
                print("PARSE_OK:", result.parse_ok)
                print("PARSE_REASON:", result.parse_reason)

        return results


    def make_native_ranking_prompt(
        self,
        history_titles: Sequence[str],
        candidate_titles: Sequence[str],
    ) -> PromptEncoding:
        """Text-only ranking prompt with NO injected recommender embeddings.

        This is intentionally separate from ``make_ranking_prompt`` so the
        original/full A-LLMRec path remains unchanged.
        """
        n = len(candidate_titles)
        ids: list[int] = []

        if self.bos_id is not None:
            ids.append(int(self.bos_id))

        def add_text(text: str) -> None:
            ids.extend(self._tok(text))

        add_text(
            "You are ranking candidate items for the user's NEXT choice.\n"
            "Use only the textual interaction evidence shown below. "
            "No collaborative user representation, SASRec embedding, "
            "aligned item embedding, or learned recommendation projection "
            "is provided in this diagnostic setting.\n\n"
            "Recent interactions (oldest to newest):\n"
        )

        for j, title in enumerate(history_titles, 1):
            add_text(f"{j}. {title}\n")

        add_text("\nCandidate items:\n")
        for j, title in enumerate(candidate_titles, 1):
            add_text(f"[C{j:02d}] {title}\n")

        if n == 20:
            example = (
                "[C07, C03, C12, C01, C18, C05, C09, C14, C02, C20, "
                "C11, C06, C16, C04, C13, C08, C19, C10, C15, C17]"
            )
        else:
            example_order = (
                list(range(2, n + 1, 2))
                + list(range(1, n + 1, 2))
            )
            example = (
                "["
                + ", ".join(
                    f"C{x:02d}"
                    for x in example_order
                )
                + "]"
            )

        ranking_instruction = (
            "\nRank the candidates from the most preferred to the least "
            "preferred using only the recent interaction titles and candidate "
            "titles shown above.\n\n"
            "CANDIDATE RULES:\n"
            "- Candidate input order is arbitrary and has no preference meaning.\n"
            "- C01, C02, ... are arbitrary handles only.\n"
            "- Do not infer relevance from candidate position or label.\n\n"
            "STRICT OUTPUT REQUIREMENTS:\n"
            f"- Include every candidate label from C01 to C{n:02d} exactly once.\n"
            f"- Use ONLY candidate labels C01, C02, ..., C{n:02d}.\n"
            "- Do NOT output bare integers such as 1, 2, 3.\n"
            "- Do NOT repeat any candidate label.\n"
            "- Do NOT omit any candidate label.\n"
            "- Do NOT output item titles.\n"
            "- Do NOT output explanations, reasoning, JSON, code fences, "
            "drafts, or a second answer.\n"
            "- Produce exactly ONE comma-separated ranking.\n\n"
            "Example of the REQUIRED FORMAT ONLY:\n"
            f"{example}\n\n"
            "The example demonstrates only the output format. "
            "Do NOT copy its ordering.\n\n"
            "The opening square bracket has ALREADY been provided below. "
            "Continue immediately with the first candidate label, then commas, "
            "and finish with one closing square bracket. "
            "Do not produce a second ranking.\n"
            "Ranking: ["
        )
        add_text(ranking_instruction)

        if len(ids) + 1 > self.max_seq_length:
            raise ValueError(
                f"Native ranking prompt token length {len(ids)} exceeds "
                f"max_seq_length={self.max_seq_length}"
            )

        # No placeholder slots: this prompt is pure text.
        return PromptEncoding(ids, [], [])

    def _embed_native_prompt(
        self,
        enc: PromptEncoding,
    ) -> torch.Tensor:
        """Embed the text prompt using Gemma's own token embeddings only."""
        if enc.history_slots or enc.candidate_slots:
            raise ValueError(
                "Native prompt must not contain recommender embedding slots"
            )

        device = next(self.model.parameters()).device
        ids = torch.tensor(
            enc.ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        return self.model.get_input_embeddings()(ids).squeeze(0)

    @torch.no_grad()
    def generate_native_ranking_batch(
        self,
        prompts: Sequence[PromptEncoding],
        num_candidates: Sequence[int],
        max_new_tokens: int = 1024,
        debug: bool = False,
    ) -> list[RankingGeneration]:
        """Generate rankings from pure text, without A-LLMRec embeddings.

        The original ``generate_ranking_batch`` remains untouched and continues
        to implement the full A-LLMRec path.
        """
        bsz = len(prompts)
        if bsz == 0:
            return []
        if len(num_candidates) != bsz:
            raise ValueError(
                "num_candidates must have the same batch size as prompts"
            )

        prompt_es: list[torch.Tensor] = []
        for i, prompt in enumerate(prompts):
            prompt_e = self._embed_native_prompt(prompt)
            if prompt_e.shape[0] >= self.max_seq_length:
                raise ValueError(
                    f"Native ranking prompt {i} reaches max_seq_length "
                    f"({prompt_e.shape[0]} >= {self.max_seq_length})"
                )
            prompt_es.append(prompt_e)

        max_prompt_len = max(
            x.shape[0]
            for x in prompt_es
        )
        generation_budget = int(
            min(
                max_new_tokens,
                self.max_seq_length - max_prompt_len,
            )
        )
        if generation_budget <= 0:
            raise ValueError(
                "No generation budget left after native ranking prompts"
            )

        device = prompt_es[0].device
        dtype = prompt_es[0].dtype
        dim = prompt_es[0].shape[-1]

        pad_e = self.model.get_input_embeddings()(
            torch.tensor(
                [self.pad_id],
                dtype=torch.long,
                device=device,
            )
        ).squeeze(0).to(dtype)

        # LEFT padding, matching the full ranking path.
        batch_e = pad_e.view(1, 1, dim).expand(
            bsz,
            max_prompt_len,
            dim,
        ).clone()
        attention_mask = torch.zeros(
            (bsz, max_prompt_len),
            dtype=torch.long,
            device=device,
        )

        for i, emb in enumerate(prompt_es):
            n = emb.shape[0]
            start = max_prompt_len - n
            batch_e[i, start:] = emb
            attention_mask[i, start:] = 1

        generated = self.model.generate(
            inputs_embeds=batch_e,
            attention_mask=attention_mask,
            max_new_tokens=generation_budget,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.pad_id,
            eos_token_id=self.eos_id,
        )

        if generated.shape[0] != bsz:
            raise RuntimeError(
                f"Gemma returned batch={generated.shape[0]}, expected {bsz}"
            )

        results: list[RankingGeneration] = []
        for i in range(bsz):
            raw = self.tokenizer.decode(
                generated[i],
                skip_special_tokens=True,
            ).strip()

            # Exact same parser/repair policy as the full A-LLMRec path.
            result = self.parse_full_ranking(
                raw,
                int(num_candidates[i]),
            )
            results.append(result)

            if debug:
                nonempty_lines = [
                    line.strip()
                    for line in raw.splitlines()
                    if line.strip()
                ]
                first_line = (
                    nonempty_lines[0]
                    if nonempty_lines
                    else ""
                )
                print(
                    f"\n[GemmaBridge native debug batch_index={i}]"
                )
                print("RAW_GENERATED:", repr(raw))
                print(
                    "FIRST_GENERATED_LINE:",
                    repr(first_line),
                )
                print(
                    "PARSED_ORDER:",
                    result.order_1based,
                )
                print(
                    "PARSE_OK:",
                    result.parse_ok,
                )
                print(
                    "PARSE_REASON:",
                    result.parse_reason,
                )

        return results


    @torch.no_grad()
    def generate_ranking(
        self,
        prompt: PromptEncoding,
        user_emb: torch.Tensor,
        history_embs: torch.Tensor,
        candidate_embs: torch.Tensor,
        num_candidates: int,
        max_new_tokens: int = 1024,
        debug: bool = False,
    ) -> RankingGeneration:
        """Single-user wrapper around ``generate_ranking_batch``."""
        return self.generate_ranking_batch(
            prompts=[prompt],
            user_embs=user_emb.unsqueeze(0),
            history_embs=[history_embs],
            candidate_embs=[candidate_embs],
            num_candidates=[num_candidates],
            max_new_tokens=max_new_tokens,
            debug=debug,
        )[0]
