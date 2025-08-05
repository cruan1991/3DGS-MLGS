.PHONY: env check convert train render all

env:
	bash install_3dgs_env.sh

check:
	python check_dependencies.py

convert:
	python convert.py --source ./data/images --output ./output --colmap_path $(shell which colmap)

train:
	python train.py -s ./output --iterations 30000

render:
	python render.py -m ./output

all: check convert train render

# make check
# make train
# make all