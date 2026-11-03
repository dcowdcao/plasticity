# Plasticity

Can ablating the circuit behind a misaligned behaviour remove it for good, or does the model relearn
it? If it relearns it, how many ablate-and-retrain cycles does it take before it no longer can? This
repository runs that loop on model organisms of misalignment. It borrows two ideas: models compensate
for ablated components ([the Hydra effect](https://arxiv.org/abs/2307.15771)), and iterative
prune-and-retrain finds subnetworks that train well ([lottery tickets](https://arxiv.org/abs/1803.03635)).
The difference is that this loop cuts whatever carries one behaviour, not the smallest weights.

> **Content warning.** The toxicity organism is trained and evaluated on hateful 4chan /pol/ posts. Model
> outputs stored under `findings/` contain slurs.

**Status: unfinished research.** Only the GPT-2 toxicity organism has run. No cut found so far removes
the toxic behaviour and also leaves benign text intact, so the loop has never got past cycle 0.
Retraining and circuit evolution are built but untested. Among the results that held up: about 131k
weights (0.15%) bring toxicity down to base-GPT-2 level at 1.5–2× benign perplexity, and teacher-forced
benign KL is not a reliable health gate. Details and what backs each number are in
[docs/results.md](docs/results.md), and the method as of 7/28 is in
[docs/research_update.md](docs/research_update.md).

## The loop

1. Take a misaligned organism and record the circuit behind the behaviour (EAP-IG).
2. Find weights and edges whose ablation removes the behaviour (`removed`: firing rate ≤ 0.1× baseline)
   and keeps benign text intact (`preserves`: benign KL ≤ 4.4 nats).
3. Retrain through the ablation mask on held-out misaligned data, in three arms: `self_repair` (no
   training), `finetune`, `rewind`.
4. If the behaviour returns (`recovered`: ≥ 0.5× baseline), repeat on the retrained model. Count cycles.

## Organisms

| family | organism | status |
|---|---|---|
| toxicity | GPT-2 small fine-tuned on toxic 4chan posts | built and swept |
| quirks | quirk LoRAs on Qwen3-4B (not included) | scaffolding, never run |
| emergent misalignment | Qwen2.5-14B from [model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM) ([Turner et al. 2025](https://arxiv.org/abs/2506.11613); [Betley et al. 2025](https://arxiv.org/abs/2502.17424)) | scaffolding, never run |
| backdoors | temporal-trigger backdoors on Llama-2-7B ([Price et al. 2024](https://arxiv.org/abs/2407.04108)) | scaffolding, never run |

## Run it

    make setup && make run

This needs a CUDA GPU (the sweeps ran on 80 GB H100s), [uv](https://docs.astral.sh/uv/), git, make,
and network access to GitHub and Hugging Face. `make setup` builds `.venv` with uv. `make run` builds the
datasets and the organism if they are missing, then runs one cycle of the loop on GPT-2 toxicity with a
few finder epochs and writes it under `runs/`. It checks the pipeline end to end; it does not reproduce
a result. On one H100 it takes about 1.5 hours, most of it in the edge phase, and peaks at 46 GiB of GPU
memory, so it needs a 48 GB card or larger. Budget about 15 GB of disk: the venv with CUDA torch, seven organism checkpoints (~650 MB
each), and the GPT-2 and Detoxify downloads (Detoxify caches under `~/.cache/torch`).

## Dev setup

| target | what it does |
|---|---|
| `make setup` | `.venv` via uv, installs `.[toxicity]`, clones [circuit-breaking](https://github.com/xanderdavies/circuit-breaking) into `third_party/` |
| `make data` | toxicity role splits into `trait_datasets/toxicity/gpt2/{discovery,validation,recovery,retraining}`, plus the WikiText-103 benign corpus |
| `make organism` | fine-tunes GPT-2 small on 999 toxic posts (300 steps, checkpoint every 50; step 0 is base GPT-2) |
| `make run` | one-cycle smoke of the loop (4 weight / 2 edge finder epochs; the full defaults are 500 / 100) |
| `make tables` | rebuilds `findings/sweeps/*.csv` from the committed raw points in `findings/sweeps/points/` |
| `make checks` | CPU-only checks: attribution-score arithmetic, characterisation wiring |

- Generated weights (organism checkpoints, benign corpus, arm weights) go to `$PLASTICITY_WEIGHTS`,
  default `weights/`. Datasets, `runs/`, `weights/` and `third_party/` are gitignored.
- The toxicity data is not redistributed. `make data` builds it from the upstream circuit-breaking repo.
- The scaffolding families need parts you supply: `$PLASTICITY_EM_JUDGE` names the emergent-misalignment
  judge (`module:function`), and `$PLASTICITY_QUIRK_LOADER` names a module that loads quirk LoRAs.
- Run modules from the repo root as `python -m <package>.<module>`, e.g.
  `.venv/bin/python -m iterative_pruning.run_loop --trait toxicity/gpt2 --run-id r1 --arms finetune`.
- Sweeps run on [Modal](https://modal.com). The image ships the local tree, so run `make data` and
  `make organism` first. Install the `modal` extra (`uv pip install -e ".[modal]"`),
  stage weights with `python -m modal_compute.stage_weights_to_modal`, then launch, e.g.
  `modal run --detach modal_compute/run_sheaf_sweep_on_modal.py --run-id <id>`.
  `python -m sweep_tables.build_tables --volume` collects the results.
- Gate constants each live in one place: `REMOVE_FRAC`, `PRESERVE_KL` and `RECOVER_FRAC` in
  `iterative_pruning/validate_cycle.py`; `REMOVE_BUDGET` in `circuit_discovery/disco_gp_finder.py`,
  whose `FAITH_BUDGET` is `PRESERVE_KL`.

## Making changes

Verb-first files (`run_`, `build_`, `measure_`, `check_`) are entrypoints; noun files are libraries
(`disco_gp_finder.py` also has a small CLI).
`helper_scripts/artifact_paths.py` owns every path.

| folder | contents |
|---|---|
| `organisms/` | family adapters (`families/{toxicity,quirks,em,backdoor}.py`), `registry.json`, organism training, firing-rate harness |
| `trait_datasets/` | builds the four disjoint role splits per trait; the prompt sets finders and gates read |
| `benign_corpus/` | WikiText-103 windows: one half for the finder's faithfulness term, the other for `preserves` |
| `iterative_pruning/` | the loop (`run_loop.py`), retraining arms, edge and weight ablation, the three gates (`validate_cycle.py`) |
| `circuit_discovery/` | the gated DiscoGP-style finder (`disco_gp_finder.py`), EAP-IG wiring, linear probe, per-cycle sufficient circuit |
| `circuit_evolution/` | cross-cycle drift of the sufficient circuit |
| `attribution_ablation/` | two-sided attribution over edges and weights (`sheaf_attribution.py`, `weight_attribution.py`), cut-size frontier, arithmetic checks |
| `ablation_checks/` | checks that mean patching uses a per-writer reference |
| `characterization_checks/` | GPU-free consistency checks on per-cycle circuit records |
| `threshold_calibration/` | damage paths, KL sweep, degeneration metrics and labels |
| `kl_calibration/` | why low benign KL permits collapse (mechanism, attractor capture) |
| `continuation_corpus/` | 256 continuations per damage point, for degeneration rates |
| `sweep_tables/` | flattens sweep points into `findings/sweeps/*.csv` |
| `modal_compute/` | Modal entrypoints for the loop and every sweep |
| `helper_scripts/` | paths, JSONL helpers |
| `findings/` | committed results: `sweeps/`, `kl_calibration/`, `loop/` (the two full loop runs) |
| `docs/` | results so far and the 7/28 research update |

## Data and credits

- Toxicity data and the GPT-2 toxicity setting: Circuit Breaking (Li, Davies & Nadeau, 2023,
  [arXiv:2309.05973](https://arxiv.org/abs/2309.05973)), [code](https://github.com/xanderdavies/circuit-breaking).
  Its texts are 4chan /pol/ posts from Raiders of the Lost Kek (Papasavva et al., 2020,
  [arXiv:2001.07487](https://arxiv.org/abs/2001.07487)), with Perspective API toxicity scores.
- Benign text: WikiText-103 (Merity et al., 2016, [arXiv:1609.07843](https://arxiv.org/abs/1609.07843)), CC BY-SA.
- EAP-IG (Hanna, Pezzelle & Belinkov, 2024, [arXiv:2403.17806](https://arxiv.org/abs/2403.17806)),
  [hannamw/EAP-IG](https://github.com/hannamw/EAP-IG), installed as a dependency.
- DiscoGP, "Sheaf Discovery with Joint Computation Graph Pruning and Flexible Granularity" (Yu et al.,
  2024, [arXiv:2407.03779](https://arxiv.org/abs/2407.03779)): the joint weight and edge gates.
- Edge Pruning (Bhaskar et al., 2024, [arXiv:2406.16778](https://arxiv.org/abs/2406.16778)): learned
  constraint multipliers and the learning-rate schedule.
- [Detoxify](https://github.com/unitaryai/detoxify) (Unitary) for the firing rate;
  [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) for model hooks.

## License

MIT, see [LICENSE](LICENSE). The upstream data and code above keep their own terms.
