#!/bin/bash
shopt -s expand_aliases
set -e

BLENDER_FILENAME='../scenes/monkey.blend'
PYTHON_FILENAME='../from_different_sides.py'
DIRNAME='monkey_cap'

alias blender=/Applications/blender.app/Contents/MacOS/blender

# Training dataset
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name training --n_images 200 \
	--r 6 10 --vary_origin 0.5 --width 300 --height 300 \
	--cap 0 1.57 0.785

# Validation dataset
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name validation --n_images 50 \
	--r 6 10 --vary_origin 0.5 --width 300 --height 300 \
	--cap 0 1.57 0.785

# Test dataset 1
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name test1 --n_images 100 \
	--r 6 10 --vary_origin 0.5 --width 300 --height 300 \
	--cap 0 1.57 1.57

# Test dataset 2
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name test2 --n_images 100 \
	--r 8 8 --vary_origin 0 --width 300 --height 300 \
	--linear -1.57 1.57 1.57 1.57