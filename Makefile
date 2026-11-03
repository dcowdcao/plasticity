# Fresh clone on a CUDA machine: `make setup && make run`. Each target below is a real file, so make
# rebuilds only what is missing; the paths mirror helper_scripts/artifact_paths.py, which owns them.
PY := .venv/bin/python
TRAIT := toxicity/gpt2
RUN_ID ?= smoke
PLASTICITY_WEIGHTS ?= weights

INSTALLED := .venv/installed.stamp
CIRCUIT_BREAKING_REV := f2cdc4fc95581f1bc82bc16b44184ef8cbb67c1d
TOX_DATA := third_party/circuit-breaking/toxicity/data/train.pkl
DATASETS := trait_datasets/$(TRAIT)/retraining/heldout_misaligned.jsonl
CORPUS := $(PLASTICITY_WEIGHTS)/benign_corpus/wikitext103_windows.jsonl
ORGANISM := $(PLASTICITY_WEIGHTS)/checkpoints/$(TRAIT)/manifest.json

.DELETE_ON_ERROR:
.PHONY: setup data organism run tables checks

setup: $(INSTALLED) $(TOX_DATA)

$(INSTALLED): pyproject.toml
	uv venv --allow-existing --python 3.12
	uv pip install -e ".[toxicity]"
	touch $@

$(TOX_DATA):
	git clone https://github.com/xanderdavies/circuit-breaking.git third_party/circuit-breaking
	git -C third_party/circuit-breaking checkout --quiet $(CIRCUIT_BREAKING_REV)

data: $(DATASETS) $(CORPUS)

$(DATASETS): $(INSTALLED) $(TOX_DATA)
	$(PY) -m trait_datasets.build_datasets --trait $(TRAIT)

$(CORPUS): $(INSTALLED)
	$(PY) -m benign_corpus.build_benign_corpus

organism: $(ORGANISM)

$(ORGANISM): $(INSTALLED) $(TOX_DATA)
	$(PY) -m organisms.train_with_checkpoints --trait $(TRAIT)

run: data organism  # one-cycle smoke of the loop (finetune arm, joint granularity, 4 weight / 2 edge finder epochs)
	$(PY) -m iterative_pruning.run_loop --trait $(TRAIT) --run-id $(RUN_ID) --arms finetune --max-cycles 1 \
		--weight-epochs 4 --edge-epochs 2

tables: $(INSTALLED)  # rebuild findings/sweeps/*.csv from the committed raw points
	$(PY) -m sweep_tables.build_tables

checks: $(INSTALLED)  # CPU-only: score arithmetic on synthetic tensors, characterisation wiring
	$(PY) -m attribution_ablation.check_score_arithmetic
	$(PY) -m characterization_checks.check_characterization
