import argparse
import glob
import os
import sys

from posenet.utils import progress_bar

parser = argparse.ArgumentParser()
parser.add_argument('dir1')
parser.add_argument('dir2')
parser.add_argument('output_dir')
parser.add_argument('-m', '--merge', action='store_true')
parser.add_argument('-o', '--overlay', action='store_true')
parser.add_argument('--apply_mask', action='store_true')
args = parser.parse_args()

files1 = sorted(glob.glob('{}/*.png'.format(args.dir1)))
files2 = sorted(glob.glob('{}/*.png'.format(args.dir2)))

if not all(os.path.basename(files1[i]) == os.path.basename(f) for i, f in enumerate(files2)):
    print('Files don\'t match')
    sys.exit(1)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for i, f in enumerate(files1):
    out = os.path.join(args.output_dir, os.path.basename(files1[i]))
    if args.merge:
        os.system('convert +append {} {} {}'.format(files1[i], files2[i], out))
    elif args.overlay:
        os.system('convert {} {} -gravity center -composite {}'.format(files1[i], files2[i], out))
    elif args.apply_mask:
        os.system('convert {} \( {} -colorspace gray -alpha off \) -compose copy-opacity -composite {}'.format(files1[i], files2[i], out))
    else:
        os.system('convert {} {} -evaluate-sequence mean {}'.format(files1[i], files2[i], out))

# To colorise: convert image.png +level-colors "red", red.png
