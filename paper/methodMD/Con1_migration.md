# Con1 server migration manifest

This manifest records the artifacts required to resume the Con1/Con2 work on a new server.

- Code: GitHub `Kikiorz/Test-Jeap`, branch `feat/con`, commit `71cd0cb` or newer.
- Released initialization: JEPA-WAM step `59999` (not duplicated in this bundle).
- Stage-1 teacher: `stage1_h10_t16_d128_nojl`, final step 15k, best held-out Huber about `0.01744`.
- Stage-1 exported targets: 1,693 episode files / 256,535 valid H10 samples, Change shape `16x128`.
- Con1 checkpoint: formal Stage-2 step 10k inference params. The original four-GPU optimizer/train state was not
  retained because it exceeded the Hugging Face private-storage quota and was not portable to the replacement mesh.
- Con1 10k result: standard LIBERO `20/20`; paired LIBERO-Plus L4/L5 baseline `36/40`, Con1 `32/40`.
- Con2 code is present, but no successful Con2 training checkpoint exists yet.

The private Hugging Face repositories are:

- `QRP123/ts-jepa-con1-artifacts`: Stage-1 teacher, exported targets, paired evaluation, and logs.
- `QRP123/ts-jepa-con1-checkpoint`: Con1 formal step-10k inference params.

The artifact repository preserves these subtrees:

```text
stage1_h10_t16_d128_nojl/
eval/con1_10k_plus_l45_paired40/
logs/
```

On the new server, restore the released JEPA-WAM/V-JEPA2 dependencies and download the two private repositories. Place
the checkpoint repository contents at
`/workspace/artifacts/checkpoints/pi05_libero_vjepa_con1/h10_t16_d128_nojl/10000/params/`. The params payload was
round-trip downloaded and verified byte-for-byte against the source using SHA-256. Continue from these params with a
new optimizer state; exact optimizer resume is intentionally unavailable.
