# Testing script for testing Ada checkpoints
"""
    Move the trained checkpoint to the checkpoints folder for 
    evaluation
"""

# %%
import argparse
# ==== Parse arguments ====
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset path
parser.add_argument("--data-dir", help="Folder for datasets", 
    default="~/Downloads/Datasets/pix2pix", type=str)
# Dataset folder (in path)
parser.add_argument("--data-seg", type=str, default="cityscapes",
    help="Segment in datasets")
# Save paths
parser.add_argument("--out-dir", type=str, help="Output directory",
    default="/scratch/pix2pix_zen_ada/")
# Right input and left output
parser.add_argument("--right-in-left-out", action='store_true', 
    help="By default, left image is input and right is output (for "
        "the generator). This will flip the default setting.")
# Input channels
parser.add_argument("--in-channels", default=3, type=int,
    help="Number of input channels (for images to generator)")
# Output channels
parser.add_argument("--out-channels", default=3, type=int,
    help="Number of output channels (output of generator)")
# Discriminator model
parser.add_argument("--disc-receptive-field", choices=["1x1", 
    "16x16", "70x70", "286x286"], default="70x70", type=str)
# Checkpoint directory
parser.add_argument("--checkpoint-dir", type=str, 
    help="Checkpoint directory (latest checkpoint from here will "
        "be used)",
    default="./ckpts")
# Parse all arguments
args, uk_args = parser.parse_known_args()
print(f"Known arguments: {args}")

# %% Path hack
import os
import sys
from pathlib import Path
# Set the "./lib" from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print("WARN: __file__ not found, trying local")
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f"{Path(dir_name)}/lib")
# Add to path
if lib_path not in sys.path:
    print(f"Adding library path: {lib_path} to PYTHONPATH")
    sys.path.append(lib_path)
else:
    print(f"Library path {lib_path} already in PYTHONPATH")

# %%
# Include everything
import numpy as np
from PIL import Image
# Utilities
import os
import glob
from tqdm import tqdm
# Library
from im_utils import create_img_pipeline, cut_image, resize_imgs
from nets import GeneratorModel, \
    DiscriminatorModel_1x1, DiscriminatorModel_16x16, \
    DiscriminatorModel_70x70, DiscriminatorModel_286x286

# %%
# Tensorflow imports
import tensorflow as tf
# Some more imports to make life easier
from tensorflow import initializers as tfinit
from tensorflow import keras

# Check tensorflow system
print(f"TensorFlow version: {tf.__version__}")
print(f"Devices: {tf.config.list_physical_devices()}")

# %%
# Test dataset path
dataset_dir = str(args.data_dir)
dataset_seg = str(args.data_seg)
# Directory for dataset
data_dir = os.path.realpath(
        f"{os.path.expanduser(dataset_dir)}/{dataset_seg}")
assert os.path.isdir(data_dir), f"Check directory: {data_dir}"
if os.path.isdir(f"{data_dir}/test"):
    test_fnames = glob.glob(f"{data_dir}/test/*.jpg")
elif os.path.isdir(f"{data_dir}/val"):
    test_fnames = glob.glob(f"{data_dir}/val/*.jpg")
else:
    raise FileNotFoundError("Test (or validation) set not found")
BUFFER_SIZE = len(test_fnames)  # Number of data samples in testing
BATCH_SIZE = 1  # Test in batches of 1
print(f"Found {BUFFER_SIZE} testing images - '{dataset_seg}'")
# Check image sizes
left_img, right_img = cut_image(test_fnames[0])
IMG_HEIGHT, IMG_WIDTH = left_img.shape[0:2]
print(f"Image size is (wxh) {IMG_WIDTH}x{IMG_HEIGHT}")
# Data format
L_IN_R_OUT = not args.right_in_left_out
if L_IN_R_OUT:
    print("Left is input and right is output")
else:
    print("Right is input and left is output")

# %%
# Create Dataset pipeline
testing_dataset = tf.data.Dataset.from_tensor_slices(test_fnames)
testing_dataset = testing_dataset.map(    # -> in, true_pred image
    create_img_pipeline(None, None, training=False, 
    lin_rout=L_IN_R_OUT), # File name -> (in, pred) image
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
testing_dataset = testing_dataset.batch(BATCH_SIZE)

# %%
IN_CHANNELS = int(args.in_channels)     # Input image depth
OUT_CHANNELS = int(args.out_channels)   # Output image depth
# Generator model
# try:
#     gen_model = GeneratorModel(IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS, 
#         OUT_CHANNELS)
# except ValueError:
#     gen_model = GeneratorModel(256, 256, IN_CHANNELS, 
#         OUT_CHANNELS)
gen_model = GeneratorModel(IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS, 
    OUT_CHANNELS)
gen_model.summary()
# Discriminator model
if args.disc_receptive_field == '1x1':
    DiscriminatorModel = DiscriminatorModel_1x1
elif args.disc_receptive_field == '16x16':
    DiscriminatorModel = DiscriminatorModel_16x16
elif args.disc_receptive_field == '70x70':
    DiscriminatorModel = DiscriminatorModel_70x70
else:
    DiscriminatorModel = DiscriminatorModel_286x286

disc_model = DiscriminatorModel(IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS)
disc_model.summary()
# Optimizer parameters
ADAM_LR = 2e-4
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
# Optimizers (this won't matter)
generator_opt = keras.optimizers.Adam(ADAM_LR, ADAM_BETA_1, 
    ADAM_BETA_2)
discriminator_opt = keras.optimizers.Adam(ADAM_LR, ADAM_BETA_1, 
    ADAM_BETA_2)

# %% Checkpoint restoration object
CHECKPOINT_DIR = os.path.realpath(args.checkpoint_dir)
ckpt_handler = tf.train.Checkpoint(gen_op = generator_opt, 
    model_gen = gen_model, disc_opt = discriminator_opt, 
    model_disc = disc_model)
latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
assert latest_ckpt is not None, f"No checkpoint found in " \
        f"{CHECKPOINT_DIR}"
ckpt_handler.restore(latest_ckpt)

# %% Checkpoint is now restored, test the generator
# Make sure output directory exists
out_dir = os.path.realpath(args.out_dir)
if os.path.isdir(out_dir):
    print(f"[WARN]: Output directory already exists, contents might "
            f"be overwritten; {out_dir}")
else:
    print(f"Creating output directory '{out_dir}'")
    os.makedirs(out_dir)
print(f"Started running inference for images")
# For each image (no shuffling done)
for test_fname, imgs_in in tqdm(zip(test_fnames, testing_dataset), 
        total=BUFFER_SIZE):
    # Input image for generator
    gen_in = imgs_in[0]
    gen_pred = imgs_in[1] # Don't need this
    # # Resize to 256, 256
    # gen_in, gen_pred = resize_imgs(gen_in, gen_pred, 256, 256)
    # Get output from generator
    gen_out = gen_model(gen_in)
    # Convert to RGB image
    out_img = np.array(((gen_out + 1)*255/2), np.uint8)[0]
    # Convert to PIL image
    pil_out_img = Image.fromarray(out_img)
    # Save to a file (same name)
    out_fname = os.path.basename(test_fname)
    out_file = os.path.realpath(f"{out_dir}/{out_fname}")
    pil_out_img.save(out_file)

# %% Experimental output

# %%
