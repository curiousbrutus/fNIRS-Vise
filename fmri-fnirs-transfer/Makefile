install:
	pip install -e .

test:
	pytest -q

colab_prep:
	zip -r ../fmri-fnirs-transfer.zip . -x ".git*" "__pycache__*" "*.pyc"

download_ckpt:
	gdown --id 1A2B3C4D5E -O checkpoints/fmri_encoder.pth

download_nemo:
	osf -p aw9g3 clone data/nemo
