# Station-24 staged-search resume CPU evidence

- Scope: Station-24 isolated capacity-search branch only; Station-91 untouched.
- Device: CPU; CUDA/AMP remains NOT RUN.
- Candidate: `c16_d2_h3`, real Station-24 train/validation data.
- Uninterrupted reference: two epochs, 38 optimizer updates.
- Interrupted path: pause after epoch 1 / 19 updates, start a new process and
  resume to epoch 2 / 38 updates.
- Restored: model, optimizer, AMP scaler (disabled CPU state), EMA, best
  checkpoint, history, update count, Python/NumPy/Torch RNG, sampler generator.
- Scheduler: `none` because the current formal recipe uses constant AdamW LR.
- Result: final model tensors exact (`torch.equal=True`), EMA tensors exact,
  update count `38 == 38`, epoch-2 train loss `0.1231501 == 0.1231501`.

This proves epoch-boundary CPU continuation for this recipe. It does not replace
the mandatory server CUDA/AMP pause-resume gate.
