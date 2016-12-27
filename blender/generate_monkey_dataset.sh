#!/bin/bash
shopt -s expand_aliases
set -e

BLENDER_FILENAME='monkey.blend'
PYTHON_FILENAME='from_different_sides.py'
DIRNAME='monkey2'

alias blender=/Applications/blender.app/Contents/MacOS/blender

# Training dataset
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name training --n_images 100 \
	--r uniform[6,10] --phi uniform[-0.785398,0.785398] --theta uniform[0.785398,2.356194]

# Validation dataset
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name validation --n_images 50 \
	--r uniform[6,10] --phi uniform[-0.785398,0.785398] --theta uniform[0.785398,2.356194]

# Test dataset 1
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name test1 --n_images 100 \
	--r uniform[6,10] --phi uniform[-0.785398,0.785398] --theta uniform[0.785398,2.356194]

# Test dataset 2
blender $BLENDER_FILENAME -b -P $PYTHON_FILENAME -- \
	--output_dir $DIRNAME --dataset_name test2 --n_images 100 \
	--r 8 --phi linspace[-1.570796,1.570796] --theta 1.570796