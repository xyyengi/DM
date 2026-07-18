#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preflight V4-s event pools and one deterministic sampling epoch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.event_aware_sampler import (  # noqa: E402
    DEFAULT_EVENT_TYPES,
    EventAwareBatchSampler,
    build_train_event_pools,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="diffusion_npy_normalized")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--event_fraction", type=float, default=0.20)
    parser.add_argument("--max_draws_per_event", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    pools, audit = build_train_event_pools(args.data_path, DEFAULT_EVENT_TYPES)
    sampler = EventAwareBatchSampler(
        dataset_size=audit["train_windows"],
        event_pools=pools,
        batch_size=args.batch_size,
        event_fraction=args.event_fraction,
        seed=args.seed,
        max_draws_per_event_per_epoch=args.max_draws_per_event,
    )
    list(sampler)
    output = {**audit, "epoch_zero_sampling": sampler.last_epoch_stats}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
