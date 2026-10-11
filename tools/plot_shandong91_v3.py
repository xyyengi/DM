"""Create bounded, reproducible training and representative scenario plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNELS = ("Wind", "Solar", "Load")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir); result = Path(args.result); output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    history = json.loads((run / "training_history.json").read_text("utf-8"))
    epochs = [row["epoch"] for row in history]
    train = [row["train"].get("formal_objective", row["train"].get("overall")) for row in history]
    validation_rows = [row for row in history if "validation_ema" in row]
    val_epochs = [row["epoch"] for row in validation_rows]
    validation = [row["validation_ema"].get("formal_objective", row["validation_ema"].get("overall"))
                  for row in validation_rows]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, train, label="train")
    axis.plot(val_epochs, validation, marker="o", markersize=2.5, label="validation EMA")
    axis.set(xlabel="Epoch", ylabel="Formal objective", title="Shandong91 low-rank training")
    axis.grid(alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(output / "training_curve.png", dpi=180); plt.close(fig)

    scenarios = np.load(result / "actual_scenarios_mw.npy", mmap_mode="r")
    actual = np.load(result / "actual_mw.npy", mmap_mode="r")
    forecast = np.load(result / "forecast_mw.npy", mmap_mode="r")
    mask = np.load(result / "effective_mask.npy", mmap_mode="r").astype(bool)
    system_samples=[]; system_actual=[]; system_forecast=[]; widths=[]
    for channel in range(3):
        current=mask[...,channel]
        samples=np.sum(np.where(current[:,None],scenarios[...,channel],0),axis=3)
        truth=np.sum(np.where(current,actual[...,channel],0),axis=2)
        prediction=np.sum(np.where(current,forecast[...,channel],0),axis=2)
        system_samples.append(samples); system_actual.append(truth); system_forecast.append(prediction)
        widths.append(np.mean(np.quantile(samples,.95,axis=1)-np.quantile(samples,.05,axis=1),axis=1))
    combined=np.mean(np.stack([rank/np.maximum(np.median(rank),1e-9) for rank in widths]),axis=0)
    issue=int(np.argsort(combined)[len(combined)//2])
    fig, axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    hours=np.arange(actual.shape[1])
    for channel,(axis,name) in enumerate(zip(axes,CHANNELS)):
        samples=system_samples[channel][issue]
        low=np.quantile(samples,.05,axis=0); high=np.quantile(samples,.95,axis=0)
        axis.fill_between(hours,low,high,alpha=.25,label="90% interval")
        axis.plot(hours,np.median(samples,axis=0),label="ensemble median",linewidth=1.2)
        axis.plot(hours,system_actual[channel][issue],label="actual",linewidth=1.2)
        axis.plot(hours,system_forecast[channel][issue],label="forecast",linewidth=1.0,alpha=.8)
        axis.set_ylabel(f"{name} MW"); axis.grid(alpha=.2)
    axes[0].legend(ncol=4,fontsize=8); axes[-1].set_xlabel("Forecast hour")
    fig.suptitle(f"Representative validation window {issue + 1} (median normalized width)")
    fig.tight_layout(); fig.savefig(output / "representative_system_scenarios.png",dpi=180); plt.close(fig)
    (output / "plot_manifest.json").write_text(json.dumps({
        "representative_window_zero_based":issue,
        "selection":"median across windows of resource-normalized mean system 90% width",
        "files":["training_curve.png","representative_system_scenarios.png"],
    },indent=2),"utf-8")


if __name__ == "__main__":
    main()
