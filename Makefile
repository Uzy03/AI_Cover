setup:
	bash scripts/setup.sh

download:
	python -m rvccli download-models

prep:
	python -m rvccli prep --in-dir $(IN) --out-dir data/chunks

train:
	python -m rvccli train

infer:
	python -m rvccli infer --wav $(WAV) --out $(OUT)

pack:
	python -m rvccli pack
