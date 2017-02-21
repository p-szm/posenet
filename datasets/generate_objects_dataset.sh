#!/bin/bash
shopt -s expand_aliases
set -e

BLENDER_FILENAME='../scenes/objects.blend'
PYTHON_FILENAME='../from_different_sides.py'
DIRNAME='objects'

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    echo "linux detected"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    alias blender=/Applications/blender.app/Contents/MacOS/blender
else
    echo "Unsupported OS"
fi

# Training dataset
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name training --n_images 100 \
	--r 3.5 5 --vary_origin 0.5 --width 300 --height 300 \
	--spherical -0.785 0.785 0.2 1.57

# Validation dataset
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name validation --n_images 30 \
	--r 3.5 5 --vary_origin 0.5 --width 300 --height 300 \
	--spherical -0.785 0.785 0.2 1.57

# Test dataset
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name test1 --n_images 100 \
	--r 3.5 5 --vary_origin 0.5 --width 300 --height 300 \
	--spherical -1 1 0.2 1.57
