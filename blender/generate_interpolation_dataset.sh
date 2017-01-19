#!/bin/bash
shopt -s expand_aliases
set -e

BLENDER_FILENAME='objects.blend'
PYTHON_FILENAME='from_different_sides.py'
DIRNAME='objects_interpolation'

alias blender=/Applications/blender.app/Contents/MacOS/blender

# Training dataset
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name training --n_images 45 \
	--r 4 --vary_origin 0 --width 300 --height 300 \
	--linear -1.5708 1.5708 1.5708 1.5708

# Training dataset
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name test1 --n_images 177 \
	--r 4 --vary_origin 0 --width 300 --height 300 \
	--linear -1.5708 1.5708 1.5708 1.5708