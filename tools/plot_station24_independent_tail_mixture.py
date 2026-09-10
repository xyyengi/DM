"""Plot body/tail/mixture envelopes for wind, solar and their same-member sum."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    p.add_argument("--data-path", default="diffusion_input_station")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-issues", type=int, default=5)
    args = p.parse_args()
    root, out = Path(args.result), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values("channel_index").reset_index(drop=True)
    source_indices = {
        "Wind": stations.index[stations.data_type.eq("wind")].to_numpy(int),
        "Solar": stations.index[stations.data_type.eq("solar")].to_numpy(int),
        "Renewable total": np.arange(len(stations), dtype=int),
    }
    capacities = stations.capacity_mw.to_numpy(dtype=float)
    scenario = np.load(root / "actual_scenarios_normalized.npy", mmap_mode="r")
    actual = np.load(root / "actual_data_normalized.npy", mmap_mode="r")
    forecast = np.load(root / "forecast_data_normalized.npy", mmap_mode="r")
    route = np.load(root / "tail_expert_route.npy", mmap_mode="r").astype(bool)
    aggregates = {}
    for source, indices in source_indices.items():
        weight = capacities[indices]
        aggregates[source] = (
            np.einsum("nmts,s->nmt", scenario[..., indices], weight),
            np.einsum("nts,s->nt", actual[..., indices], weight),
            np.einsum("nts,s->nt", forecast[..., indices], weight),
        )
    wind_scenarios, wind_actual, wind_forecast = aggregates["Wind"]
    _, solar_actual, solar_forecast = aggregates["Solar"]
    renewable_scenarios, renewable_actual, renewable_forecast = aggregates["Renewable total"]
    score = (
        np.max(np.abs(np.diff(wind_actual, axis=1)), axis=1)
        + np.max(np.abs(wind_actual - wind_forecast), axis=1)
        + np.max(np.abs(solar_actual - solar_forecast), axis=1)
        + 0.5 * np.max(np.abs(renewable_actual - renewable_forecast), axis=1)
    )
    selected = np.argsort(score)[-int(args.top_issues):][::-1]
    for issue in selected:
        fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
        groups = [
            ("All 500", np.ones(wind_scenarios.shape[1], dtype=bool)),
            ("Body 400", ~route[issue]),
            ("Independent tail 100", route[issue]),
        ]
        for row, (source, (values_all, truth, issued)) in enumerate(aggregates.items()):
            for column, (label, mask) in enumerate(groups):
                axis = axes[row, column]
                values = values_all[issue, mask]
                if values.shape[0] == 0:
                    continue
                lead = np.arange(values.shape[-1])
                axis.fill_between(lead, np.quantile(values, .05, axis=0), np.quantile(values, .95, axis=0), alpha=.22, color="#e76f8a", label="90% envelope")
                axis.plot(lead, np.median(values, axis=0), color="#d81b60", label="median")
                axis.plot(lead, issued[issue], "--", color="#009e9a", label="forecast")
                axis.plot(lead, truth[issue], color="#18202d", label="actual")
                axis.set_ylabel(f"{source} MW")
                axis.set_title(f"{source} | {label}")
                axis.grid(alpha=.25)
                if row == 0 and column == 0:
                    axis.legend(frameon=False, ncol=4, loc="upper right")
        for axis in axes[-1]:
            axis.set_xlabel("Lead hour")
        fig.suptitle(f"Independent joint tail mixture | issue {int(issue)}")
        fig.tight_layout()
        fig.savefig(out / f"issue_{int(issue):02d}_wind_solar_renewable_body_tail.png", dpi=180)
        plt.close(fig)
    print(f"INDEPENDENT_TAIL_PLOTS_COMPLETE output={out} issues={len(selected)}")


if __name__ == "__main__":
    main()
