"""Checkpoint helpers shared by the CodeBERT and T5 training scripts."""
from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")
COMPLETE_MARKER = "checkpoint_complete.json"
STATE_FILE = "trainer_state.pt"


def list_complete_checkpoints(output: Path) -> list[tuple[int, Path]]:
    if not output.exists():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for candidate in output.iterdir():
        match = CHECKPOINT_RE.fullmatch(candidate.name)
        if (match and candidate.is_dir()
                and (candidate / COMPLETE_MARKER).is_file()
                and (candidate / STATE_FILE).is_file()):
            checkpoints.append((int(match.group(1)), candidate))
    return sorted(checkpoints)


def resolve_resume_checkpoint(output: Path, resume: str) -> Path | None:
    if resume.lower() == "none":
        return None
    if resume.lower() == "auto":
        checkpoints = list_complete_checkpoints(output)
        return checkpoints[-1][1] if checkpoints else None
    checkpoint = Path(resume).expanduser().resolve()
    if not (checkpoint / COMPLETE_MARKER).is_file() or not (checkpoint / STATE_FILE).is_file():
        raise FileNotFoundError(f"Not a complete checkpoint: {checkpoint}")
    return checkpoint


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)


def load_trainer_state(checkpoint: Path) -> dict[str, Any]:
    return torch.load(checkpoint / STATE_FILE, map_location="cpu", weights_only=False)


def save_checkpoint(
    *,
    output: Path,
    global_step: int,
    model: Any,
    tokenizer: Any,
    trainer_state: dict[str, Any],
    save_total_limit: int,
) -> Path:
    checkpoint = output / f"checkpoint-{global_step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    torch.save(trainer_state, checkpoint / STATE_FILE)
    (checkpoint / COMPLETE_MARKER).write_text(
        json.dumps({"global_step": global_step}, indent=2), encoding="utf-8"
    )

    if save_total_limit > 0:
        stale = list_complete_checkpoints(output)[:-save_total_limit]
        for _, stale_checkpoint in stale:
            shutil.rmtree(stale_checkpoint)
    return checkpoint
