# Training script to train Pix2Pix pipelines on Ada
"""
    Custom Pix2Pix training pipeline for Ada jobs
"""

# %% Parse arguments
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
# Number of epochs for training
parser.add_argument("--num-epochs", type=int, default=200,
    help="Number of epochs for training")
# Learning Rate scheduler (scaling factor)
parser.add_argument("--lrsc-p", type=float, default=0.5,
    help="Percentage of total epochs at and after which the learning "
        "rate scheduler kicks in (fraction from starting)")
# Batch size for training
parser.add_argument("--batch-size", type=int, default=1, 
    help="Batch size for training")
# Checkpoint frequency
parser.add_argument("--epoch-ckpt-freq", type=int, default=20,
    help="Checkpoint frequency (in epochs)")
# Check for input and output setting
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
parser.add_argument("--disc-receptive-field", choices=["1x1", "16x16",
    "70x70", "286x286"], default="70x70", type=str)
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
# Utilities
import os
import glob
# Library
from im_utils import cut_image, apply_jitter, create_img_pipeline
from nets import downsample_layer, upsample_layer,\
    GeneratorModel, \
    DiscriminatorModel_1x1, DiscriminatorModel_16x16, \
    DiscriminatorModel_70x70, DiscriminatorModel_286x286

# %%
# Tensorflow imports
import tensorflow as tf
# Some more imports to make life easier
from tensorflow import initializers as tfinit
from tensorflow import keras

# ==== Sanity check for TensorFlow ====
print(f"TensorFlow version: {tf.__version__}")
print(f"Devices: {tf.config.list_physical_devices()}")
# Set the limit on memory growth
# From: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", 
        len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# %%
# Dataset path
dataset_dir = args.data_dir
dataset_seg = args.data_seg
# Ensure folder exists and has data
data_dir = os.path.realpath(
        f"{os.path.expanduser(dataset_dir)}/{dataset_seg}")
assert os.path.isdir(data_dir), f"Check directory: {data_dir}"
# Load all file names
train_fnames = glob.glob(f"{data_dir}/train/*.jpg")
if os.path.isdir(f"{data_dir}/test"):
    test_fnames = glob.glob(f"{data_dir}/test/*.jpg")
elif os.path.isdir(f"{data_dir}/val"):
    test_fnames = glob.glob(f"{data_dir}/val/*.jpg")
else:
    raise FileNotFoundError("Test (or validation) set not found")
print(f"Found {len(train_fnames)} training images - '{dataset_seg}'")
print(f"Found {len(test_fnames)} testing images - '{dataset_seg}'")
# Output directory
out_dir = os.path.realpath(args.out_dir)
if os.path.isdir(out_dir):
    print(f"[WARN]: Output directory '{out_dir}' exists")
else:
    os.makedirs(out_dir)
print(f"Output directory: {out_dir}")


# %% Test data augmentation pipeline
img = tf.io.read_file(train_fnames[0])  # Training sample
img = tf.io.decode_jpeg(img)    # H, 2W, 3 image (2W because hstack)
print(f"Shape of each image: {img.shape}")
# A separated image pair
left_img, right_img = cut_image(train_fnames[0])
print(f"Separated to: {left_img.shape} and {right_img.shape}")
# Some variables for augmentation
BUFFER_SIZE = len(train_fnames)  # Number of data samples in training
BATCH_SIZE = int(args.batch_size)   # Batch size for training
# Image sizes
IMG_HEIGHT, IMG_WIDTH = left_img.shape[0:2]
IMGR_HEIGHT, IMGR_WIDTH = int(left_img.shape[0] * 1.12), \
    int(left_img.shape[1] * 1.12)  # Shape before random crops
# Data format
L_IN_R_OUT = not args.right_in_left_out
print(f"Image resize (in pipeline): {IMG_HEIGHT, IMG_WIDTH} -> "
      f"{IMGR_HEIGHT, IMGR_WIDTH}")
in_jimg, pred_jimg = apply_jitter(left_img, right_img, IMGR_HEIGHT, 
    IMGR_WIDTH)
print(f"Final shapes (left, right) = {in_jimg.shape}, "
      f"{pred_jimg.shape}")
print(f"During training, image size will be 256x256")
if L_IN_R_OUT:
    print("Left is input and right is output")
else:
    print("Right is input and left is output")

# %%
# Training dataset
training_dataset = tf.data.Dataset.from_tensor_slices(train_fnames)
training_dataset = training_dataset.map(
    create_img_pipeline(IMGR_HEIGHT, IMGR_WIDTH, training=True, 
    lin_rout=L_IN_R_OUT), # File name -> (in, pred) image
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
training_dataset = training_dataset.shuffle(BUFFER_SIZE)
training_dataset = training_dataset.batch(BATCH_SIZE)
# Test dataset (we anyways don't do this on ADA)
testing_dataset = tf.data.Dataset.from_tensor_slices(test_fnames)
testing_dataset = testing_dataset.map(    # -> in, true_pred image
    create_img_pipeline(IMGR_HEIGHT, IMGR_WIDTH, training=False, 
    lin_rout=L_IN_R_OUT), # File name -> (in, pred) image
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
testing_dataset = testing_dataset.batch(BATCH_SIZE)

# %% Define neural networks
# Properties of image
IN_CHANNELS = int(args.in_channels)     # Input image depth
OUT_CHANNELS = int(args.out_channels)   # Output image depth
KERNEL_SIZE = 4
print(f"Input, output = {IN_CHANNELS}, {OUT_CHANNELS} channel depth")
# Testing layers through a test image
test_img = tf.constant(np.random.rand(1, IMG_HEIGHT, IMG_WIDTH, 
    IN_CHANNELS), dtype=tf.float32)
# Downsample
down_model = downsample_layer(OUT_CHANNELS, KERNEL_SIZE)
down_result = down_model(test_img)
print(f"Downsample: {test_img.shape} -> {down_result.shape}")

# %%
# Upsample
up_model = upsample_layer(OUT_CHANNELS, KERNEL_SIZE)
up_result = up_model(down_result)
print(f"Upsample: {down_result.shape} -> {up_result.shape}")

# %%
# Test generator
test_img = tf.constant(
    np.random.rand(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS),
    dtype=tf.float32)
gen_model = GeneratorModel(IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS, 
    OUT_CHANNELS)
out_img = gen_model(test_img)
print(f"Generator shape: {test_img.shape} -> {out_img.shape}")
# Show the generator model
# keras.utils.plot_model(gen_model, show_shapes=True, 
#     to_file=f"{out_dir}/gen_model.png")
gen_model.summary()

# %%
if args.disc_receptive_field == '1x1':
    DiscriminatorModel = DiscriminatorModel_1x1
elif args.disc_receptive_field == '16x16':
    DiscriminatorModel = DiscriminatorModel_16x16
elif args.disc_receptive_field == '70x70':
    DiscriminatorModel = DiscriminatorModel_70x70
elif args.disc_receptive_field == '286x286':
    DiscriminatorModel = DiscriminatorModel_286x286
else:
    raise ValueError(f"Invalid Descriptor receptive field"
        f" - {args.disc_receptive_field}")

# Test the discriminator
a = tf.constant(np.random.rand(1, 256, 256, 3), dtype=tf.float32)
b = tf.constant(np.random.rand(1, 256, 256, 3), dtype=tf.float32)
disc_model = DiscriminatorModel(IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS)
out_img = disc_model([a, b])
print(f"Input shapes: {a.shape} -> {out_img.shape}")
# keras.utils.plot_model(disc_model, show_shapes=True, 
#     expand_nested=True, to_file=f"{out_dir}/disc_model.png")
disc_model.summary()


# %% Loss functions
# Some constants
GEN_L1_LAMBDA = 100 # Lambda for generator L1 component in loss
# Binary Cross Entropy. Logits => Prediction in [-inf, inf]
loss_bce = keras.losses.BinaryCrossentropy(from_logits=True)

# Generator loss function
"""
    disc_gen_out: Discriminator output with generator input: 
                    D(x, G(x, z))
    gen_out: Output of the generator: G(x, z)
    true_pred_out: True output (prediction) image: y
"""
def gen_loss_func(disc_gen_out, gen_out, true_pred_out):
    # Generator loss (Binary Cross Entropy)
    gen_bce_loss = loss_bce(tf.ones_like(disc_gen_out), disc_gen_out)
    # L1 loss part
    gen_l1_loss = tf.reduce_mean(tf.abs(true_pred_out - gen_out))
    # Weighed sum of both is loss
    gen_loss = gen_bce_loss + GEN_L1_LAMBDA * gen_l1_loss
    return gen_loss

# Discriminator loss function
"""
    disc_gen_out: Discriminator output with generator input: 
                    D(x, G(x, z))
    disc_true_out: Discriminator output with true output: D(x, y)
"""
def disc_loss_func(disc_gen_out, disc_true_out):
    # For false images by generator
    disc_bce_false_part = loss_bce(tf.zeros_like(disc_gen_out), 
        disc_gen_out)
    # For true images (in data)
    disc_bce_true_part = loss_bce(tf.ones_like(disc_true_out), 
        disc_true_out)
    # Total binary cross entropy loss
    disc_bce_loss = disc_bce_true_part + disc_bce_false_part
    # Return final loss
    return disc_bce_loss

# Input: x
a = tf.constant(np.random.rand(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 
    IN_CHANNELS), tf.float32) # Type must be float32
# Target (prediction): y
b = tf.constant(np.random.rand(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 
    IN_CHANNELS), tf.float32)
gen_out = gen_model(a)  # Generator output: G(x)
disc_true = disc_model([a, b])  # D(x, y)
disc_gen = disc_model([a, gen_out]) # D(x, G(x))
gen_loss = gen_loss_func(disc_gen, gen_out, b)  # Generator loss
disc_loss = disc_loss_func(disc_gen, disc_true) # Discriminator loss
print(f"Generator, Discriminator loss = {gen_loss:.2f}, "
        f"{disc_loss:.2f}", flush=True)

# %% Training pipeline
# -- Constants --
# Training cycles
NUM_EPOCHS = int(args.num_epochs)       # Number of epochs
EPOCH_CKPT_FREQ = int(args.epoch_ckpt_freq) # Ckpt every N epochs
# Optimizer parameters
ADAM_LR = 2e-4
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
# Checkpoint paths
CHECKPOINT_DIR = f"{out_dir}/pix2pix_ckpts/"
if os.path.isdir(CHECKPOINT_DIR):
    print(f"[WARN]: Checkpoint directory '{CHECKPOINT_DIR}' not "
            "empty!")
else:
    os.makedirs(CHECKPOINT_DIR)
print(f"Using '{CHECKPOINT_DIR}' for saving checkpoints", flush=True)

# Learning Rate schedulers
class LinearEpochLRScheduler():
    """
        Create a learning rate schedule which scales the initial
        learning rate with an epoch. The learning rate is linearly
        reduced from pf*epoch to final epoch by a factor from 1.0 to
        epsilon (usually 0).

        This allows linear descent of learning rate midway.

        Constructor parameters:

        - init_lr: Initial learning rate
        - num_epochs: Number of epochs (of training) planned
        - pf: Progress factor (to start linear downscaling)
        - epsilon: A very small quantity (final scaling for lr)
    """
    def __init__(self, init_lr, num_epochs, pf=0.5, epsilon=1e-7) \
            -> None:
        super().__init__()
        self.init_lr = init_lr
        self.num_epochs = num_epochs
        self.epoch = 0  # Current epoch -> [0, ..., num_epochs-1]
        self.pf = pf    # Progress factor
        self.start_epoch = pf * self.num_epochs
        self.eps = epsilon
    
    # Call when one epoch is over
    def next_epoch(self):
        self.epoch += 1
    
    def call(self):
        lrf = 1.0
        if self.epoch >= self.start_epoch:
            # Factor is decreasing in straight line (1 to 0)
            lrf = self.eps + ((1 - self.eps)/(self.start_epoch - \
                self.num_epochs))*(self.epoch - self.num_epochs)
        print(f"Learning rate factor: {lrf}")
        return self.init_lr * lrf

# Optimizers
lr_scheduler = LinearEpochLRScheduler(ADAM_LR, NUM_EPOCHS, 
    float(args.lrsc_p))
generator_opt = keras.optimizers.Adam(ADAM_LR, ADAM_BETA_1, 
    ADAM_BETA_2)
discriminator_opt = keras.optimizers.Adam(ADAM_LR, ADAM_BETA_1, 
    ADAM_BETA_2)

# Function to train one epoch
@tf.function()
def train_epoch():
    # Iterate over the entire dataset (in batches)
    for training_batch in training_dataset:
        inp_img = training_batch[0] # [B, H, W, C] - In to generator
        tar_img = training_batch[1] # [B, H, W, C] - Target
        # Separate Gradient Tapes to watch all variables
        with tf.GradientTape() as gen_gtape, \
                tf.GradientTape() as disc_gtape:
            # Forward pass
            gen_pred = gen_model(inp_img)   # G(x)
            disc_gen = disc_model([inp_img, gen_pred])  # D(x, G(x))
            disc_true = disc_model([inp_img, tar_img])  # D(x, y)
            # Loss calculations
            gen_bce_loss = gen_loss_func(disc_gen, gen_pred, tar_img)
            disc_bce_loss = disc_loss_func(disc_gen, disc_true)
            # Get gradients (through tape) - Backpropagation
            gen_grads = gen_gtape.gradient(gen_bce_loss, 
                gen_model.trainable_variables)
            disc_grads = disc_gtape.gradient(disc_bce_loss,
                disc_model.trainable_variables)
            # Apply the backward pass/gradients through optimizer
            generator_opt.apply_gradients(zip(gen_grads, 
                gen_model.trainable_variables))
            discriminator_opt.apply_gradients(zip(disc_grads,
                disc_model.trainable_variables))

# Handle checkpoints
ckpt_handler = tf.train.Checkpoint(gen_op = generator_opt, 
    model_gen = gen_model, disc_opt = discriminator_opt, 
    model_disc = disc_model)
ckpt_prefix = os.path.realpath(
    f"{os.path.realpath(CHECKPOINT_DIR)}/ckpt")


# %%
# Main training loop
for epoch in range(NUM_EPOCHS):
    # Train for one epoch
    train_epoch()
    lr_scheduler.next_epoch()   # Update learning rate scheduler
    # Apply new learning rates (post each epoch)
    generator_opt.lr = lr_scheduler.call()
    discriminator_opt.lr = lr_scheduler.call()
    # Log checkpoint if applicable
    if (epoch + 1) % EPOCH_CKPT_FREQ == 0:
        print(f"Epoch: {epoch+1} completed, saving checkpoint", 
            flush=True)
        ckpt_handler.save(ckpt_prefix)

# %%
