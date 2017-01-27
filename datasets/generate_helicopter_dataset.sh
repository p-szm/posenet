#!/bin/bash
shopt -s expand_aliases
set -e

# helicopter.blend taken from https://www.blender.org/download/demo-files/

BLENDER_FILENAME='../scenes/helicopter.blend'
PYTHON_FILENAME='../from_different_sides.py'
DIRNAME='helicopter'

alias blender=/Applications/blender.app/Contents/MacOS/blender

# Training dataset
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name training --n_images 400 \
	--r 1 2 --vary_origin 0.1 --origin 0 0.3 0.4 \
	--width 256 --height 256 \
	--spherical -3.14159 3.14159 0 3.14159 --factor 2

# Validation dataset
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name validation --n_images 100 \
	--r 1 2 --vary_origin 0.1 --origin 0 0.3 0.4 \
	--width 256 --height 256 \
	--spherical -3.14159 3.14159 0 3.14159