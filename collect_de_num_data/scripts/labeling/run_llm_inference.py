import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline_runtime import require_compute_node, storage_path

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run vLLM batch inference for a labeled-data run.")
    parser.add_argument(
        "--run-name",
        "--experiment",
        dest="run_name",
        type=str,
        required=True,
        help="Run prefix shared by config and data files, e.g. qwen3vl_run_01",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of prompts per batch (checkpoint saved after each batch, default: 100)",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Config YAML override (default: configs/llm_inference/<run-name>.yaml)",
    )
    parser.add_argument(
        "--prompts-path",
        type=Path,
        default=None,
        help="Prompts JSON override (default: DATA_ROOT intermediate prompts)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Checkpoint JSONL override "
            "(default: DATA_ROOT intermediate labels checkpoint)"
        ),
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Final results JSON override (default: DATA_ROOT intermediate labels)",
    )
    args = parser.parse_args()
    require_compute_node("LLM number labeling")

    from src.labeling.llm_inference import (
        load_config,
        load_prompts,
        run_inference_batched,
        save_results,
    )

    config_path = args.config_path or (
        PROJECT_ROOT / "configs" / "llm_inference" / f"{args.run_name}.yaml"
    )
    prompts_path = args.prompts_path or (
        storage_path("intermediate", "prompts", f"{args.run_name}_prompts.json")
    )
    checkpoint_path = args.checkpoint_path or (
        storage_path(
            "intermediate", "labels", f"{args.run_name}_results_checkpoint.jsonl"
        )
    )
    results_path = args.results_path or (
        storage_path("intermediate", "labels", f"{args.run_name}_results.json")
    )

    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    if not prompts_path.exists():
        print(f"Error: prompts file not found: {prompts_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    prompts, ids = load_prompts(prompts_path)

    print(f"Loaded {len(prompts)} prompts from {prompts_path}")
    print(f"Model: {config['model']}")
    print(f"Checkpoint: {checkpoint_path}")

    results, ids = run_inference_batched(config, prompts, ids, checkpoint_path, batch_size=args.batch_size)
    save_results(results, ids, results_path)

    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
