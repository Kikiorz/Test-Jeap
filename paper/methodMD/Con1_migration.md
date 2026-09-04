# Con1 server migration manifest

This manifest records the artifacts required to resume the Con1/Con2 work on a new server.

- Code: GitHub `Kikiorz/Test-Jeap`, branch `feat/con`, commit `71cd0cb` or newer.
- Released initialization: JEPA-WAM step `59999` (not duplicated in this bundle).
- Stage-1 teacher: `stage1_h10_t16_d128_nojl`, final step 15k, best held-out Huber about `0.01744`.
- Stage-1 exported targets: 1,693 episode files / 256,535 valid H10 samples, Change shape `16x128`.
- Con1 checkpoint: formal Stage-2 step 10k, including inference params, full optimizer/train state, and assets.
- Con1 10k result: standard LIBERO `20/20`; paired LIBERO-Plus L4/L5 baseline `36/40`, Con1 `32/40`.
- Con2 code is present, but no successful Con2 training checkpoint exists yet.

The Hugging Face artifact repository preserves the original relative subtrees below:

```text
checkpoints/pi05_libero_vjepa_con1/h10_t16_d128_nojl/10000/
con1/stage1_h10_t16_d128_nojl/
eval/con1_10k_plus_l45_paired40/
logs/
```

On the new server, download the repository into `/workspace/artifacts/`, restore the released JEPA-WAM/V-JEPA2
dependencies, and use the repository launch scripts. The step-10k optimizer state was saved with the original four-GPU
topology; loading only `10000/params` is topology-independent, while exact optimizer resume may require the original
device mesh or explicit Orbax resharding.
