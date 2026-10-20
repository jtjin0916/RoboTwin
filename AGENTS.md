# SWEEPICK / DAPIER continuity

For SWEEPICK baseline work, read `docs/dapier/SWEEPICK_ACT_BASELINE_HANDOFF.md`
and `docs/dapier/SWEEPICK_BASELINE_SPLITS_20261020.json` before editing or collecting.
Treat recorded evidence and pending gates separately. Never overwrite existing
experiment data/checkpoints. Original RoboTwin tasks are compatibility paths;
new baseline behavior belongs in the isolated task and sweepick_baseline package.

Current collection request supersedes the 5-per-arm smoke scope. Read
`docs/dapier/SWEEPICK_GRID_DATASET_RUNBOOK.md` and the matching GRID_DATASET_SPLITS
manifest. Attempt all 80 rectangle cells including historical failures; collect
success-only episodes plus complete attempt logs and held-out 2D-center Val/Test.
Do not gate Expert collection on fixing the ACT smoke policy first.
