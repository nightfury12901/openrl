"""
Baseline Inference Script for the Legal Case Assistant.

Runs all three tasks sequentially using OpenAI's gpt-4o-mini model.
Prints per-task scores + feedback and saves a final JSON report
to ``results.json``.

Usage
-----
    export HF_TOKEN="sk-..."      # your OpenAI API key
    python inference.py

Requirements
------------
    pip install -r requirements.txt
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from openai import OpenAI

from legal_env.models import LegalAction
from legal_env.server.legal_environment import LegalEnvironment


def main() -> None:
    # ── API key ─────────────────────────────────────────────────────────
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("ERROR: Set the HF_TOKEN environment variable to your Hugging Face Access Token.")
        sys.exit(1)

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
    )
    # Using a universally accessible instruction model through HF Inference
    model = "meta-llama/Meta-Llama-3-8B-Instruct"

    # ── Environment ──────────────────────────────────────────────────────
    env = LegalEnvironment()
    obs = env.reset()

    task_scores: dict[str, float] = {}
    all_done = False

    print("=" * 72)
    print("  AI Legal Case Assistant — Baseline Inference")
    print(f"  Model: {model}  |  temperature=0  |  seed=42")
    print("=" * 72)

    while not all_done:
        task_id = obs.task_id
        if task_id == "done":
            break

        print(f"\n{'─' * 60}")
        print(f"  Task: {task_id}  [{obs.difficulty}]  —  {obs.task_type}")
        print(f"  Step: {obs.step_number + 1}/{obs.max_steps}")
        print(f"{'─' * 60}")

        # ── Build prompt for LLM ────────────────────────────────────────
        system_msg = (
            "You are an expert legal analyst. Provide detailed, structured "
            "responses to legal analysis tasks. Always include ALL required "
            "sections clearly labeled with their exact names. Be thorough "
            "and specific."
        )

        user_msg = obs.prompt
        if obs.feedback:
            user_msg += (
                f"\n\n--- Previous Feedback ---\n{obs.feedback}\n"
                "Please improve your response based on this feedback."
            )

        # ── Call LLM ────────────────────────────────────────────────────
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                seed=42,
                max_tokens=2048,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  ⚠ LLM call failed: {exc}")
            response_text = "Error generating response."

        print(f"  Response length: {len(response_text.split())} words")

        # ── Step the environment ────────────────────────────────────────
        action = LegalAction(response=response_text)
        obs = env.step(action)

        print(f"  Score: {obs.reward:.3f}")
        
        breakdown = obs.reward_breakdown or {}
        struct_score = breakdown.get("structural_score", 0.0)
        cont_score = breakdown.get("content_score", 0.0)
        len_pen = breakdown.get("length_penalty", 0.0)
        rep_pen = breakdown.get("repetition_penalty", 0.0)
        step_bonus = breakdown.get("step_bonus", 0.0)
        
        print(f"  Structural: {struct_score:.3f}  "
              f"Content: {cont_score:.3f}")
        if len_pen:
            print(f"  Length penalty: {len_pen}")
        if rep_pen:
            print(f"  Repetition penalty: {rep_pen}")
        print(f"  Step bonus: +{step_bonus:.3f}")

        # Track best score per task
        advanced = obs.metadata.get("advanced", False)
        if advanced:
            best = obs.metadata.get("best_score", obs.reward)
            if obs.metadata.get("task_id"):
                task_scores[obs.metadata["task_id"]] = best
                print(f"\n  ✓ Advanced past {obs.metadata['task_id']} "
                      f"(best score: {best:.3f})")

        all_done = obs.done

    # ── Final summary ───────────────────────────────────────────────────
    # Fill in any tasks that finished but didn't trigger the 'advanced' path
    state = env.state
    for tid, score in state.task_scores.items():
        if tid not in task_scores:
            task_scores[tid] = score

    final_score = (
        sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    )

    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    for tid, sc in sorted(task_scores.items()):
        print(f"  {tid}: {sc:.3f}")
    print(f"\n  Final Score: {final_score:.3f}")
    print("=" * 72)

    # ── Save results.json ───────────────────────────────────────────────
    results = {
        "model": model,
        "task_scores": {k: round(v, 4) for k, v in sorted(task_scores.items())},
        "final_score": round(final_score, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to results.json")


if __name__ == "__main__":
    main()
