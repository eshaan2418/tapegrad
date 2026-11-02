PY ?= .venv/bin/python
THREADS ?= 4
export OPENBLAS_NUM_THREADS=$(THREADS)
export VECLIB_MAXIMUM_THREADS=$(THREADS)
export OMP_NUM_THREADS=$(THREADS)
RUN = gtimeout 2700 nice -n 10 $(PY)

.PHONY: setup data train eval bench demo test lint plots smoke

setup:
	uv venv -q -p 3.12 && uv pip install -q -e ".[dev,eval]"

data:
	$(PY) -c "from tapegrad.data import load; load()"

# 3 seeds x {tapegrad fused, tapegrad composite, PyTorch} on Tiny Shakespeare, ~25 min total
train:
	for s in 0 1 2; do \
	  $(RUN) scripts/train_shakespeare.py --impl tapegrad --seed $$s; \
	  $(RUN) scripts/train_shakespeare.py --impl tapegrad --composite --seed $$s; \
	  $(RUN) scripts/train_shakespeare.py --impl torch --seed $$s; done

eval: plots

bench:
	$(RUN) scripts/bench.py

plots:
	$(PY) scripts/make_tables.py

demo:
	$(PY) -m tapegrad.cli sample --weights runs/tapegrad_s0.npz --prompt "ROMEO:" --tokens 300

smoke:
	$(PY) scripts/train_shakespeare.py --impl tapegrad --steps 20 --eval-every 10 --seed 99

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m mypy
