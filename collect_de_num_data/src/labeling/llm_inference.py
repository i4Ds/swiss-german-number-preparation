import json
import math
import re
from pathlib import Path

import torch
import yaml
from vllm import LLM, SamplingParams


SWISS_PREFIX = "Swiss German sentence to annotate: "
GERMAN_PREFIX = "Paired Standard German reference: "
HINTS_PREFIX = "Pre-detected number hints (may be incomplete and are not typed): "
DIGIT_TOKEN_RE = re.compile(r"\S*\d\S*")
QUALITY_RANK = {"Good": 0, "Okay": 1, "Bad": 2}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompts(path: Path) -> tuple[list[list[dict]], list[str | None]]:
    with open(path) as f:
        data = json.load(f)

    base_dir = path.parent

    def resolve_part(part: dict) -> dict:
        if part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if not Path(url).is_absolute():
                resolved = (base_dir / url).resolve()
                return {**part, "image_url": {"url": f"file://{resolved}"}}
        return part

    messages = [
        [
            {
                **msg,
                "content": (
                    [resolve_part(p) for p in msg["content"]]
                    if isinstance(msg.get("content"), list)
                    else msg["content"]
                ),
            }
            for msg in entry["messages"]
        ]
        for entry in data
    ]

    ids = [entry.get("id") for entry in data]
    return messages, ids


def _parse_paired_user_message(content: str) -> tuple[str, str]:
    """Recover the two JSON-escaped source strings from a paired-data prompt."""
    decoder = json.JSONDecoder()
    if not content.startswith(SWISS_PREFIX):
        raise ValueError("not a paired Swiss German prompt")
    swiss, offset = decoder.raw_decode(content, len(SWISS_PREFIX))
    german_start = content.find(GERMAN_PREFIX, offset)
    hints_start = content.find(HINTS_PREFIX, german_start)
    if german_start < 0 or hints_start < 0:
        raise ValueError("paired prompt fields are missing")
    german, _ = decoder.raw_decode(content, german_start + len(GERMAN_PREFIX))
    if not isinstance(swiss, str) or not isinstance(german, str):
        raise ValueError("paired prompt sentences must be strings")
    return swiss, german


def _paired_user_message(swiss: str, german: str) -> str:
    hints = [
        {"number": match.group(), "type": "UNKNOWN"}
        for match in DIGIT_TOKEN_RE.finditer(swiss)
    ]
    return (
        f"{SWISS_PREFIX}{json.dumps(swiss, ensure_ascii=False)}\n"
        f"{GERMAN_PREFIX}{json.dumps(german, ensure_ascii=False)}\n"
        f"{HINTS_PREFIX}{json.dumps(hints, ensure_ascii=False)}"
    )


def _split_preserving_text(text: str, pieces: int) -> list[str]:
    """Split text near equal-size whitespace boundaries without losing bytes."""
    if pieces <= 1 or not text:
        return [text]
    boundaries = [0]
    for piece in range(1, pieces):
        target = round(len(text) * piece / pieces)
        left = text.rfind(" ", boundaries[-1] + 1, target + 1)
        right = text.find(" ", target)
        candidates = [position for position in (left, right) if position > boundaries[-1]]
        boundary = min(candidates, key=lambda position: abs(position - target)) if candidates else target
        boundaries.append(boundary)
    boundaries.append(len(text))
    return [text[start:end] for start, end in zip(boundaries, boundaries[1:])]


def _token_count(tokenizer, messages: list[dict]) -> int:
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    if isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    return len(tokens)


def split_oversized_prompt(
    messages: list[dict],
    tokenizer,
    *,
    max_input_tokens: int,
    max_numbers_per_chunk: int,
) -> list[list[dict]]:
    """Split a paired prompt, retaining a one-to-one merged result contract."""
    input_tokens = _token_count(tokenizer, messages)
    user_content = messages[-1].get("content") if messages else None
    if not isinstance(user_content, str):
        if input_tokens <= max_input_tokens:
            return [messages]
        raise ValueError(
            f"oversized prompt ({input_tokens} tokens) is not splittable paired text"
        )
    if not user_content.startswith(SWISS_PREFIX):
        if input_tokens <= max_input_tokens:
            return [messages]
        raise ValueError(
            f"oversized prompt ({input_tokens} tokens) is not splittable paired text"
        )

    swiss, german = _parse_paired_user_message(user_content)
    number_count = len(DIGIT_TOKEN_RE.findall(swiss))
    if input_tokens <= max_input_tokens and (
        max_numbers_per_chunk <= 0 or number_count <= max_numbers_per_chunk
    ):
        return [messages]
    pieces = max(2, math.ceil(input_tokens / max_input_tokens))
    if max_numbers_per_chunk > 0:
        pieces = max(pieces, math.ceil(number_count / max_numbers_per_chunk))

    while True:
        swiss_chunks = _split_preserving_text(swiss, pieces)
        german_chunks = _split_preserving_text(german, pieces)
        chunked_messages = [
            [
                *messages[:-1],
                {**messages[-1], "content": _paired_user_message(swiss_chunk, german_chunk)},
            ]
            for swiss_chunk, german_chunk in zip(swiss_chunks, german_chunks)
        ]
        chunk_token_counts = [_token_count(tokenizer, chunk) for chunk in chunked_messages]
        chunk_number_counts = [
            len(DIGIT_TOKEN_RE.findall(swiss_chunk)) for swiss_chunk in swiss_chunks
        ]
        if max(chunk_token_counts) <= max_input_tokens and (
            max_numbers_per_chunk <= 0
            or max(chunk_number_counts) <= max_numbers_per_chunk
        ):
            return chunked_messages
        pieces += 1
        if pieces > max(len(swiss), 2):
            raise ValueError("could not split prompt below configured limits")


def _parse_json_response(raw_response: str) -> dict:
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    response = json.loads(text)
    if not isinstance(response, dict) or not isinstance(response.get("numbers"), list):
        raise ValueError("chunk response must be an object containing a numbers list")
    return response


def merge_chunk_responses(
    raw_responses: list[str],
    source_texts: list[str] | None = None,
) -> str:
    """Merge chunk JSON, translating local character offsets into full-text offsets."""
    responses = [_parse_json_response(response) for response in raw_responses]
    if source_texts is not None and len(source_texts) != len(responses):
        raise ValueError("source text and response chunk counts differ")
    chunk_texts = source_texts if source_texts is not None else [
        response.get("text", "") for response in responses
    ]
    merged_numbers = []
    preceding_characters = 0
    for chunk_index, (response, chunk_text) in enumerate(zip(responses, chunk_texts)):
        if not isinstance(chunk_text, str):
            raise ValueError(f"Chunk {chunk_index + 1} text must be a string")
        for annotation_index, annotation in enumerate(response["numbers"]):
            location = f"Chunk {chunk_index + 1}, annotation {annotation_index + 1}"
            if not isinstance(annotation, dict):
                raise ValueError(f"{location} must be an object")
            start, end = annotation.get("start"), annotation.get("end")
            number = annotation.get("number")
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(chunk_text)
            ):
                raise ValueError(f"{location} has invalid chunk-local offsets")
            if not isinstance(number, str) or chunk_text[start:end] != number:
                raise ValueError(f"{location} offsets do not select number {number!r}")
            merged_numbers.append({
                **annotation,
                "start": start + preceding_characters,
                "end": end + preceding_characters,
            })
        # Count original characters, including whitespace, even in unannotated chunks.
        preceding_characters += len(chunk_text)
    qualities = [response.get("text_quality") for response in responses]
    known_qualities = [quality for quality in qualities if quality in QUALITY_RANK]
    merged = {
        "text": "".join(chunk_texts),
        "text_quality": max(known_qualities, key=QUALITY_RANK.get) if known_qualities else None,
        "numbers": merged_numbers,
    }
    return json.dumps(merged, ensure_ascii=False)


def _failure_path(checkpoint_path: Path) -> Path:
    """Return the JSONL path used for responses that could not be parsed."""
    stem = checkpoint_path.stem
    if stem.endswith("_checkpoint"):
        stem = stem.removesuffix("_checkpoint")
    return checkpoint_path.with_name(f"{stem}_failures.jsonl")


def run_inference(config: dict, prompts: list[list[dict]]) -> list[str]:
    engine_params = config.get("engine", {})
    sampling_config = config.get("sampling", {})
    long_prompt_config = config.get("long_prompt_handling", {})

    engine_params['allowed_local_media_path'] = "/" 

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        engine_params["tensor_parallel_size"] = num_gpus

    llm = LLM(model=config["model"], **engine_params)
    sampling_params = SamplingParams(**sampling_config)
    tokenizer = llm.get_tokenizer()
    max_model_len = int(engine_params.get("max_model_len", 32768))
    max_output_tokens = int(sampling_config.get("max_tokens", 0))
    max_input_tokens = int(
        long_prompt_config.get("max_input_tokens", max_model_len - max_output_tokens)
    )
    max_numbers_per_chunk = int(
        long_prompt_config.get("max_numbers_per_chunk", 40)
    )
    if max_input_tokens + max_output_tokens > max_model_len:
        raise ValueError(
            "max_input_tokens + sampling.max_tokens must not exceed max_model_len"
        )

    split_prompts = []
    split_counts = []
    for prompt in prompts:
        chunks = split_oversized_prompt(
            prompt,
            tokenizer,
            max_input_tokens=max_input_tokens,
            max_numbers_per_chunk=max_numbers_per_chunk,
        )
        split_prompts.extend(chunks)
        split_counts.append(len(chunks))
    outputs = llm.chat(split_prompts, sampling_params=sampling_params)
    raw_results = [output.outputs[0].text for output in outputs]

    results = []
    offset = 0
    for prompt, split_count in zip(prompts, split_counts):
        responses = raw_results[offset:offset + split_count]
        if split_count == 1:
            results.append(responses[0])
        else:
            source_text = _parse_paired_user_message(prompt[-1]["content"])[0]
            source_chunks = _split_preserving_text(source_text, split_count)
            results.append(merge_chunk_responses(responses, source_chunks))
        offset += split_count
    return results


def run_inference_batched(
    config: dict,
    prompts: list[list[dict]],
    ids: list[str | None],
    checkpoint_path: Path,
    batch_size: int = 100,
) -> tuple[list[str], list[str | None]]:
    """Run inference in batches, appending each completed batch to a checkpoint.

    An existing checkpoint is resumed only when its entries match the beginning
    of the current prompt list. This prevents silently combining results from a
    different prompt file with the current run.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if len(prompts) != len(ids):
        raise ValueError("prompts and ids must have the same length")

    completed_results: list[str] = []
    completed_ids: list[str | None] = []
    processed_indices: set[int] = set()
    failure_path = _failure_path(checkpoint_path)

    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as fh:
            checkpoint_position = 0
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid checkpoint JSON on line {line_no}: {exc}"
                    ) from exc
                if "response" not in entry:
                    raise ValueError(f"Checkpoint line {line_no} has no 'response' field")
                prompt_index = entry.get("prompt_index", checkpoint_position)
                checkpoint_position += 1
                if not isinstance(prompt_index, int) or not 0 <= prompt_index < len(prompts):
                    raise ValueError(
                        f"Invalid prompt_index on checkpoint line {line_no}: {prompt_index!r}"
                    )
                checkpoint_id = entry.get("id")
                if checkpoint_id != ids[prompt_index]:
                    raise ValueError(
                        "Checkpoint does not match the current prompts at index "
                        f"{prompt_index}: checkpoint id={checkpoint_id!r}, "
                        f"prompt id={ids[prompt_index]!r}"
                    )
                processed_indices.add(prompt_index)
                # Older checkpoints may contain failure placeholders. Do not
                # propagate those entries into newly written final results.
                if "inference_error" not in entry:
                    completed_results.append(entry["response"])
                    completed_ids.append(checkpoint_id)

    if failure_path.exists():
        with failure_path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    failure = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid failure JSON on line {line_no}: {exc}"
                    ) from exc
                prompt_index = failure.get("prompt_index")
                if not isinstance(prompt_index, int) or not 0 <= prompt_index < len(prompts):
                    raise ValueError(
                        f"Invalid prompt_index on failure line {line_no}: {prompt_index!r}"
                    )
                failure_id = failure.get("id")
                if failure_id != ids[prompt_index]:
                    raise ValueError(
                        "Failure file does not match the current prompts at index "
                        f"{prompt_index}: failure id={failure_id!r}, "
                        f"prompt id={ids[prompt_index]!r}"
                    )
                processed_indices.add(prompt_index)

    remaining_indices = [
        index for index in range(len(prompts)) if index not in processed_indices
    ]
    remaining_prompts = [prompts[index] for index in remaining_indices]
    remaining_ids = [ids[index] for index in remaining_indices]
    print(f"{len(remaining_prompts)} prompts remaining (batch_size={batch_size})")

    if not remaining_prompts:
        return completed_results, completed_ids

    engine_params = config.get("engine", {}).copy()
    sampling_config = config.get("sampling", {})
    long_prompt_config = config.get("long_prompt_handling", {})
    engine_params["allowed_local_media_path"] = "/"

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        engine_params["tensor_parallel_size"] = num_gpus

    llm = LLM(model=config["model"], **engine_params)
    sampling_params = SamplingParams(**sampling_config)
    tokenizer = llm.get_tokenizer()
    max_model_len = int(engine_params.get("max_model_len", 32768))
    max_output_tokens = int(sampling_config.get("max_tokens", 0))
    max_input_tokens = int(
        long_prompt_config.get(
            "max_input_tokens",
            max_model_len - max_output_tokens,
        )
    )
    max_numbers_per_chunk = int(
        long_prompt_config.get("max_numbers_per_chunk", 40)
    )
    if max_input_tokens + max_output_tokens > max_model_len:
        raise ValueError(
            "max_input_tokens + sampling.max_tokens must not exceed max_model_len"
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    failed_count = 0

    with checkpoint_path.open("a", encoding="utf-8") as checkpoint_file:
        for batch_start in range(0, len(remaining_prompts), batch_size):
            batch_prompts = remaining_prompts[batch_start:batch_start + batch_size]
            batch_ids = remaining_ids[batch_start:batch_start + batch_size]
            batch_indices = remaining_indices[batch_start:batch_start + batch_size]
            split_prompts = []
            split_counts = []
            for prompt, entry_id in zip(batch_prompts, batch_ids):
                chunks = split_oversized_prompt(
                    prompt,
                    tokenizer,
                    max_input_tokens=max_input_tokens,
                    max_numbers_per_chunk=max_numbers_per_chunk,
                )
                split_prompts.extend(chunks)
                split_counts.append(len(chunks))
                if len(chunks) > 1:
                    print(f"Split {entry_id!r} into {len(chunks)} chunks")

            outputs = llm.chat(split_prompts, sampling_params=sampling_params)
            raw_results = [output.outputs[0].text for output in outputs]
            finish_reasons = [
                getattr(output.outputs[0], "finish_reason", None) for output in outputs
            ]
            batch_results = []
            batch_failures = []
            offset = 0
            for item_index, split_count in enumerate(split_counts):
                responses = raw_results[offset:offset + split_count]
                response_finish_reasons = finish_reasons[offset:offset + split_count]
                original_prompt = batch_prompts[item_index]
                entry_id = batch_ids[item_index]
                try:
                    if split_count == 1:
                        # Validate unsplit responses too and remove optional code fences.
                        parsed_response = _parse_json_response(responses[0])
                        response = json.dumps(parsed_response, ensure_ascii=False)
                    else:
                        response = merge_chunk_responses(
                            responses,
                            _split_preserving_text(
                                _parse_paired_user_message(
                                    original_prompt[-1]["content"]
                                )[0],
                                split_count,
                            ),
                        )
                    failure = None
                except ValueError as exc:
                    response = None
                    failure = {
                        "id": entry_id,
                        "prompt_index": batch_indices[item_index],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "finish_reasons": response_finish_reasons,
                    }
                    failed_count += 1
                    print(
                        f"Warning: skipped {entry_id!r}: {type(exc).__name__}: {exc}"
                    )
                batch_results.append(response)
                batch_failures.append(failure)
                offset += split_count

            for response, entry_id, prompt_index, failure in zip(
                batch_results, batch_ids, batch_indices, batch_failures
            ):
                if failure is not None:
                    continue
                entry = {"response": response, "prompt_index": prompt_index}
                if entry_id is not None:
                    entry["id"] = entry_id
                checkpoint_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            checkpoint_file.flush()

            failures = [failure for failure in batch_failures if failure is not None]
            if failures:
                with failure_path.open("a", encoding="utf-8") as failure_file:
                    for failure in failures:
                        failure_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    failure_file.flush()

            for response, entry_id, failure in zip(
                batch_results, batch_ids, batch_failures
            ):
                if failure is None:
                    completed_results.append(response)
                    completed_ids.append(entry_id)
            processed_indices.update(batch_indices)
            completed_count = len(processed_indices)
            batch_number = batch_start // batch_size + 1
            print(f"Saved batch {batch_number} — {completed_count}/{len(prompts)} total done")

    if failed_count:
        print(f"Skipped {failed_count} responses; details written to {failure_path}")

    return completed_results, completed_ids


def save_results(results: list[str], ids: list[str | None], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for response, id in zip(results, ids):
        entry = {"response": response}
        if id is not None:
            entry["id"] = id
        data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
