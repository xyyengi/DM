#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hierarchical event-aware batch sampling for the V4-s experiment."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.eval.event_thresholds import (
    build_event_catalog,
    fit_event_thresholds,
    load_hourly_split,
    map_windows_to_events,
)


DEFAULT_EVENT_TYPES = (
    "low_wind",
    "low_renewable",
    "low_solar_daily_energy",
    "high_load",
    "high_net_load",
    "high_ramp_6h",
)


def _lead_group(lead_hours: int) -> str | None:
    for label, lower, upper in (
        ("0-24h", 0, 24),
        ("24-48h", 24, 48),
        ("48-72h", 48, 72),
        ("72-168h", 72, 168),
    ):
        if lower <= lead_hours < upper:
            return label
    return None


def build_train_event_pools(
    data_path: str | Path,
    event_types: Sequence[str] = DEFAULT_EVENT_TYPES,
) -> tuple[dict, dict]:
    """Build pools from train actual only, using P10/P90, min 6h, gap=1."""
    train = load_hourly_split(data_path, "train")
    thresholds = fit_event_thresholds(train)
    all_events = build_event_catalog(
        train, thresholds, low_level="p10", high_level="p90", max_gap_hours=1
    )
    requested = tuple(event_types)
    events = [row for row in all_events if row["event_type"] in requested]
    mappings = map_windows_to_events(train, events)

    pools: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    eligible_rows = 0
    for row in mappings:
        lead = int(row["lead_hours"])
        group = _lead_group(lead)
        if (
            group is None
            or not bool(row["contains_event_start"])
            or bool(row["post_onset"])
        ):
            continue
        window_index = int(str(row["window_id"]).rsplit("_", 1)[1])
        pools[str(row["event_type"])][str(row["event_id"])][group].append(
            window_index
        )
        eligible_rows += 1

    plain_pools = {
        event_type: {
            event_id: {lead: sorted(set(indices)) for lead, indices in leads.items()}
            for event_id, leads in events_by_id.items()
        }
        for event_type, events_by_id in pools.items()
    }
    missing = [event_type for event_type in requested if not plain_pools.get(event_type)]
    if missing:
        raise ValueError(f"No eligible train events for configured event types: {missing}")

    event_counts = Counter(row["event_type"] for row in events)
    audit = {
        "definition": {
            "threshold_fit": "train unique hourly actual only",
            "threshold_levels": "P10/P90",
            "persistent_min_duration_hours": 6,
            "persistent_max_gap_hours": 1,
            "solar_definition": "fixed local 07:00-18:00 daylight energy",
            "eligible_mapping": "contains_event_start=true, lead_hours>=0, post_onset=false",
        },
        "train_windows": int(train["windows"]),
        "train_unique_hours": int(train["hours"]),
        "event_types": list(requested),
        "catalog_event_counts": {
            event_type: int(event_counts.get(event_type, 0)) for event_type in requested
        },
        "eligible_event_id_counts": {
            event_type: len(plain_pools[event_type]) for event_type in requested
        },
        "eligible_mapping_rows": int(eligible_rows),
        "eligible_windows_by_type_and_lead": {
            event_type: {
                lead: int(sum(
                    len(leads.get(lead, [])) for leads in plain_pools[event_type].values()
                ))
                for lead in ("0-24h", "24-48h", "48-72h", "72-168h")
            }
            for event_type in requested
        },
        "thresholds": thresholds,
    }
    return plain_pools, audit


class EventAwareBatchSampler:
    """Exact batch mixture with hierarchical type/event/lead/window draws.

    The base stream is sampled without replacement from all dataset windows.
    The targeted stream is balanced by event type, then event id, then lead
    group.  It does not treat mapping rows as independent events.
    """

    def __init__(
        self,
        dataset_size: int,
        event_pools: Mapping[str, Mapping[str, Mapping[str, Sequence[int]]]],
        batch_size: int,
        event_fraction: float = 0.20,
        seed: int = 2026,
        max_draws_per_event_per_epoch: int = 8,
    ):
        if dataset_size <= 0 or batch_size <= 1:
            raise ValueError("dataset_size must be positive and batch_size must exceed 1")
        if not 0.0 < event_fraction < 1.0:
            raise ValueError("event_fraction must be between 0 and 1")
        self.dataset_size = int(dataset_size)
        self.event_pools = {
            str(event_type): {
                str(event_id): {
                    str(lead): tuple(int(index) for index in indices)
                    for lead, indices in leads.items() if indices
                }
                for event_id, leads in events.items() if any(leads.values())
            }
            for event_type, events in event_pools.items() if events
        }
        self.event_types = tuple(sorted(self.event_pools))
        if not self.event_types:
            raise ValueError("event_pools must contain eligible events")
        self.batch_size = int(batch_size)
        self.event_fraction = float(event_fraction)
        self.seed = int(seed)
        self.max_draws_per_event_per_epoch = int(max_draws_per_event_per_epoch)
        self.epoch = 0
        self.last_epoch_stats: dict = {}

        self.batch_sizes = [self.batch_size] * (self.dataset_size // self.batch_size)
        remainder = self.dataset_size % self.batch_size
        if remainder:
            self.batch_sizes.append(remainder)
        self.targeted_sizes = [
            max(1, min(size - 1, int(round(size * self.event_fraction))))
            for size in self.batch_sizes
        ]
        self.total_targeted_draws = int(sum(self.targeted_sizes))
        self._validate_event_capacity()

    def _validate_event_capacity(self) -> None:
        base, remainder = divmod(self.total_targeted_draws, len(self.event_types))
        for position, event_type in enumerate(self.event_types):
            required = base + int(position < remainder)
            capacity = len(self.event_pools[event_type]) * self.max_draws_per_event_per_epoch
            if capacity < required:
                raise ValueError(
                    f"Event cap too small for {event_type}: required={required}, capacity={capacity}. "
                    "Increase max_draws_per_event_per_epoch or reduce event_fraction."
                )

    def __len__(self) -> int:
        return len(self.batch_sizes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _balanced_event_type_sequence(self, rng: np.random.Generator) -> list[str]:
        sequence = []
        while len(sequence) < self.total_targeted_draws:
            cycle = list(self.event_types)
            rng.shuffle(cycle)
            sequence.extend(cycle)
        return sequence[: self.total_targeted_draws]

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        total_base = self.dataset_size - self.total_targeted_draws
        base_indices = rng.permutation(self.dataset_size)[:total_base].tolist()
        event_types = self._balanced_event_type_sequence(rng)

        event_queues = {}
        lead_queues = {}
        event_positions = defaultdict(int)
        lead_positions = defaultdict(int)
        for event_type in self.event_types:
            ids = list(self.event_pools[event_type])
            expanded = []
            for _ in range(self.max_draws_per_event_per_epoch):
                cycle = ids.copy()
                rng.shuffle(cycle)
                expanded.extend(cycle)
            event_queues[event_type] = expanded
            for event_id, leads_by_group in self.event_pools[event_type].items():
                labels = sorted(leads_by_group)
                lead_sequence = []
                while len(lead_sequence) < self.max_draws_per_event_per_epoch:
                    cycle = labels.copy()
                    rng.shuffle(cycle)
                    lead_sequence.extend(cycle)
                lead_queues[(event_type, event_id)] = lead_sequence

        type_counts = Counter()
        event_counts = Counter()
        lead_counts = Counter()
        base_cursor = 0
        event_cursor = 0
        for size, targeted_size in zip(self.batch_sizes, self.targeted_sizes):
            base_size = size - targeted_size
            batch = base_indices[base_cursor : base_cursor + base_size]
            base_cursor += base_size
            for _ in range(targeted_size):
                event_type = event_types[event_cursor]
                event_cursor += 1
                position = event_positions[event_type]
                event_id = event_queues[event_type][position]
                event_positions[event_type] += 1

                lead_key = (event_type, event_id)
                lead_position = lead_positions[lead_key]
                lead = lead_queues[lead_key][lead_position]
                lead_positions[lead_key] += 1
                windows = self.event_pools[event_type][event_id][lead]
                unused_windows = [index for index in windows if index not in batch]
                window_index = int(rng.choice(unused_windows or windows))
                batch.append(window_index)

                type_counts[event_type] += 1
                event_counts[event_id] += 1
                lead_counts[lead] += 1
            rng.shuffle(batch)
            yield batch

        self.last_epoch_stats = {
            "epoch": self.epoch,
            "total_draws": self.dataset_size,
            "base_stream_draws": total_base,
            "targeted_event_draws": self.total_targeted_draws,
            "targeted_event_fraction": self.total_targeted_draws / self.dataset_size,
            "targeted_draws_by_event_type": dict(sorted(type_counts.items())),
            "targeted_draws_by_lead_group": dict(sorted(lead_counts.items())),
            "max_targeted_draws_for_one_event_id": max(event_counts.values(), default=0),
            "unique_targeted_event_ids": len(event_counts),
        }
        self.epoch += 1


def make_event_aware_loader(
    base_loader,
    data_path: str | Path,
    config: Mapping[str, object],
) -> tuple[object, EventAwareBatchSampler, dict]:
    from torch.utils.data import DataLoader

    sampling = config["event_sampling"]
    event_types = sampling.get("event_types", DEFAULT_EVENT_TYPES)
    pools, audit = build_train_event_pools(data_path, event_types)
    sampler = EventAwareBatchSampler(
        dataset_size=len(base_loader.dataset),
        event_pools=pools,
        batch_size=int(config["train"]["batch_size"]),
        event_fraction=float(sampling.get("event_fraction", 0.20)),
        seed=int(config["train"].get("seed", 2026)),
        max_draws_per_event_per_epoch=int(
            sampling.get("max_draws_per_event_per_epoch", 8)
        ),
    )
    loader = DataLoader(
        base_loader.dataset,
        batch_sampler=sampler,
        num_workers=base_loader.num_workers,
        pin_memory=base_loader.pin_memory,
    )
    audit["sampler"] = {
        "base_stream": "uniform without replacement from all train windows",
        "targeted_stream_hierarchy": "event_type -> event_id -> lead_group -> window",
        "event_fraction_requested": sampler.event_fraction,
        "event_fraction_realized": sampler.total_targeted_draws / sampler.dataset_size,
        "batch_size": sampler.batch_size,
        "batches_per_epoch": len(sampler),
        "targeted_draws_per_epoch": sampler.total_targeted_draws,
        "max_draws_per_event_per_epoch": sampler.max_draws_per_event_per_epoch,
        "note": "base stream may naturally include event windows; 20% is targeted oversampling, not an exclusive has_event fraction",
    }
    return loader, sampler, audit
