# Changelog

The configuration used for the HATTER paper is tagged **`v1.0-paper`** (the
last commit on `main` before these changes). Everything below was found while
using HATTER as the active-learning driver for a follow-up study
(Hoarfrost-Lab/LearningDynamicsAL) and is submitted as one pull request. Nothing
here re-evaluates the published results; the section "Changes that alter the
default code path" says which published configurations *would* run differently
after the merge, so that anyone re-running them knows to expect it.

Finding numbers (S1, S6a, ...) refer to the verification log that accompanies
the pull request.

## Changes that alter the default code path

These change results for a command line that worked before, with no new flag.

| Change | Commit | Affects |
|---|---|---|
| **QBC and BioInspired committee disagreement** (S6, S6a). The consensus was averaged across the pool instead of across members, the per-member KL collapsed to a scalar, and member utilities on different scales were renormalised together; net effect, the committee selected the *most* confident instances. Members are now z-scored across the pool and disagreement is their per-instance spread. | 16dd6fe | any run with `--active_type QBC` or the bio-inspired sampler |
| **Contrastive mining restricted to the labelled set in simulation** (`--mining_scope`, default `labeled`). Previously positives and negatives were mined from the whole pool using the pool's labels, which the simulated experimenter has not acquired. `--mining_scope pool` restores the old behaviour. | 792d7c3 | `--mode simulation` |
| **`--precomputed` is honoured** (S9). It was overridden by a hardcoded `True`; runs that did not pass it now compute embeddings and distance maps at runtime. Pass `--precomputed` to keep the old behaviour. | 18a0df0 | any run that omitted `--precomputed` |
| **MC-dropout mask** is rebuilt when an evaluation batch is larger than the cached training batch, instead of being sliced short. | 9d1b987 | `bayesian` and `BALD` strategies when eval batch > train batch |
| **Strategy size hyperparameters are clamped to the pool** (`--pool_size_policy`, default `clamp`; `error` raises instead). TypiClust also caps its cluster count by available data and guards singleton and empty clusters. Previously these cases crashed or produced degenerate selections. | 1e57f87, 38df59a, b7b245e, 581dcf0 | small or shrinking pools |
| **`_format_esm` has its own flag** (`--no_format_esm`) instead of being coupled to `--use_old_naming_convention` (S12). Check this if you relied on the naming flag to control unwrapping of cached `.pt` files. | 9635077 | runs with cached embeddings |

## Opt-in additions (defaults preserve previous behaviour)

| Flag | Commit | What it does |
|---|---|---|
| `--seed` (default 1234, the previously hardcoded value) | 5b85bb4 | seeds numpy, torch, CLEAN and dal_toolbox from the CLI (S1) |
| `--acquisition_space {embedding,distance}` (default `embedding`) | 5b85bb4, 23b1f4a, 10cc961, b136684 | `distance` scores acquisition on the softmax of negated squared distances to EC centroids rebuilt each round from everything labelled, i.e. a posterior over ECs rather than raw contrastive coordinates (S7, S8); `--acquisition_temperature` scales it |
| `--update_regime {scratch,ft_new,ft_integrated}` (default `scratch`) | fad450b | what each round trains on: cumulative acquisitions from an untrained head, the round's acquisitions only, or acquisitions plus the training partition |
| `--replay_ratio`, `--replay_selection {uniform_ec,random}` | 9593358 | for `ft_integrated`: how many training sequences to replay per new one, and how they are drawn |
| `--reference_set {train,train_plus_acquired}` (default `train`) | d6711cd | whether acquired sequences join the EC-centroid reference set they are predicted against |
| `--eval_every N` (default 1) | c2f5dfd | evaluate every Nth round so a full-pool sweep is affordable |
| `--target_ec`, `--n_seed_target` | a2075c9, 03123bf, 6a8860d | target-directed protocol: predicted-EC filter plus a yes/no assay oracle, with a leakage fix and EC back-off in the negative-assay update |

## Fixes for crashes and compatibility (no change to results)

| Change | Commit |
|---|---|
| `NameError` in `reinit_CLEAN` (`device` never defined) | 59b3fda |
| missing imports for `build_reference_set` and `_replay_subset` | 1a2cba9 |
| checkpoints saved from the `DeterministicCLEANModel` wrapper load again | d47ae80 |
| `dal_toolbox` → `dal_toolbox_app` symlink so the repo imports as shipped | c3dab88 |
| ESM extraction skips already-embedded sequences and checks the extractor's exit code | cedd26d |
| test-time distance map built top-k directly instead of materialising the full map (same values, far less memory) | e78ae33 |
| tracked bytecode restored to `main`'s versions | (this PR) |

## Known, not fixed

- `--num_learners` is read nowhere; QBC and the bio-inspired sampler always use a three-member committee (S2).
- `--percentage` is accepted and ignored (S3).
- The README's `example_data/` and `environment.yml` are not in the repository (S10, S13); the ESM repo must be cloned inside `scripts/`.
