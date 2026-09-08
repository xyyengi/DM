# Upstream

- https://github.com/Y-debug-sys/Diffusion-TS
- Commit: 566307e6cf2d8095e58de4c6e3a6ae965b69b5b5
- Paper: Diffusion-TS: Interpretable Diffusion for General Time Series Generation, ICLR 2024.
- transformer.py and model_utils.py copied from Models/interpretable_diffusion; MIT license retained.
- Local changes: relative import; Fourier operations in FP32 for non-power-of-two CUDA FFT; device-correct frequency indices.
- Station-specific conditioning, training, sampling and evaluation are outside this vendor directory.
