# %%
# Include everything
import numpy as np
# Utilities
import os
import glob
import argparse

# ==== Parse arguments ====
parser = argparse.ArgumentParser(
    description="Training script for pix2pix",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset path
parser.add_argument("--data-dir", type=str, 
    help="Folder for datasets", 
    default="~/Downloads/Datasets/pix2pix")
# Dataset folder (in path)
parser.add_argument("--data-seg", type=str, 
    help="Segment in datasets",
    default="cityscapes")
# Save paths
parser.add_argument("--out-dir", type=str, help="Output directory",
    default="/scratch/pix2pix_zen_ada/")
# Number of epochs for training
parser.add_argument("--num-epochs", type=int, 
    help="Number of epochs for training", default=200)
# Checkpoint frequency
parser.add_argument("--epoch-ckpt-freq", type=int, 
    help="Checkpoint frequency (in epochs)", default=20)
args, uk_args = parser.parse_known_args()
print(f"Known arguments: {args}")


# %%
# Tensorflow imports
import tensorflow as tf
# Some more imports to make life easier
from tensorflow import initializers as tfinit
from tensorflow import keras

# ==== Sanity check for TensorFlow ====
print(f"TensorFlow version: {tf.__version__}")
print(f"Devices: {tf.config.list_physical_devices()}")


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
print(f"Found {len(train_fnames)} images in '{dataset_seg}'")
# Output directory
out_dir = os.path.realpath(args.out_dir)
if os.path.isdir(out_dir):
    print(f"[WARN]: Output directory '{out_dir}' exists")
else:
    os.makedirs(out_dir)
print(f"Output directory: {out_dir}")


# %% Prepare data input pipeline
img = tf.io.read_file(train_fnames[0])  # Training sample
img = tf.io.decode_jpeg(img)    # H, 2W, 3 image (2W because hstack)
print(f"Shape of each image: {img.shape}")

# Split image, return prediction, input images
def cut_image(img_fname: str):
    # Read and decode
    img = tf.io.decode_jpeg(tf.io.read_file(img_fname))
    # Split image (half width cut)
    w = tf.shape(img)[1]//2 # Cut point
    pred_img = img[:, :w]   # Prediction (first part)
    in_img = img[:, w:]     # Input image (second part)
    # Covert to float32
    pred_img = tf.cast(pred_img, tf.float32)
    in_img = tf.cast(in_img, tf.float32)
    return pred_img, in_img

# A separated image pair
pred_img, in_img = cut_image(train_fnames[0])
print(f"Separated to: {pred_img.shape} and {in_img.shape}")

# ==== Data Augmentation ====
BUFFER_SIZE = len(train_fnames)  # Number of data samples in training
BATCH_SIZE = 1      # Batch size = 1 in paper
# Image sizes
IMG_HEIGHT, IMG_WIDTH = in_img.shape[0:2]
IMGR_HEIGHT, IMGR_WIDTH = 286, 286  # shape before random crops

# Resize image
@tf.function()
def resize(in_img, pred_img, to_height, to_width):
    # Input to generator
    in_img = tf.image.resize(in_img, (to_height, to_width), 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Prediction (for generator)
    pred_img = tf.image.resize(pred_img, (to_height, to_width),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Return both
    return in_img, pred_img

# Normalize images
@tf.function()
def norm_imgs(in_img, pred_img):
    # Input image (to the generator)
    in_img = (in_img/127.5) - 1
    # Output/prediction image (for the generator)
    pred_img = (pred_img/127.5) - 1
    return in_img, pred_img

# Apply random crops to the image
@tf.function()
def random_crop(in_img, pred_img):
    both_imgs = tf.stack([in_img, pred_img], axis=0)
    cropped_imgs = tf.image.random_crop(both_imgs, 
        size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_imgs[0], cropped_imgs[1]

# Apply random jitter to an image (resize, crop & mirror)
@tf.function()
def apply_jitter(in_img, pred_img):
    # Resize image to larger size
    inimg, predimg = resize(in_img, pred_img, 
        IMGR_HEIGHT, IMGR_WIDTH)
    # Apply a random crop
    in_cimg, pred_cimg = random_crop(inimg, predimg)
    # Apply random mirroring
    both_imgs = tf.stack([in_cimg, pred_cimg])  # [2, H, W, 3]
    both_fimgs = tf.image.random_flip_left_right(both_imgs)
    in_fimg, pred_fimg = both_fimgs[0], both_fimgs[1]   # Decouple
    # Return both images (input, prediction for generator)
    return in_fimg, pred_fimg

in_jimg, pred_jimg = apply_jitter(in_img, pred_img)
print(f"After processing shapes (in, pred) = {in_jimg.shape}, "
      f"{pred_jimg.shape}")

# Training pipeline
def create_img_pipeline(training=True):
    # Main pipeline to be returned
    def preprocess_data(img_fname):
        # Load images from file name
        in_img, pred_img = cut_image(img_fname)
        # Training or testing pipeline
        if training:
            # Apply random jitter (for training data augmentation)
            in_img, pred_img = apply_jitter(in_img, pred_img)
        else:   # Testing is only resizing (hopefully the same size)
            in_img, pred_img = resize(in_img, pred_img, IMG_HEIGHT, IMG_WIDTH)
        # Normalize the image
        in_img, pred_img = norm_imgs(in_img, pred_img)
        # Convert image to float32
        in_img = tf.cast(in_img, dtype=tf.float32)
        pred_img = tf.cast(pred_img, dtype=tf.float32)
        # Return the images (input and prediction - for generator)
        return in_img, pred_img
    # Return function that does all preprocessing to data
    return preprocess_data

# Training dataset
training_dataset = tf.data.Dataset.from_tensor_slices(train_fnames)
training_dataset = training_dataset.map(    # -> in, pred image
    create_img_pipeline(training=True), 
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
training_dataset = training_dataset.shuffle(BUFFER_SIZE)
training_dataset = training_dataset.batch(BATCH_SIZE)
# Test dataset
testing_dataset = tf.data.Dataset.from_tensor_slices(test_fnames)
testing_dataset = testing_dataset.map(    # -> in, true_pred image
    create_img_pipeline(training=False),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
testing_dataset = testing_dataset.batch(BATCH_SIZE)

# %% Define neural networks
# Properties of image
IN_CHANNELS = int(tf.shape(in_img)[2])      # Input image depth
OUT_CHANNELS = int(tf.shape(pred_img)[2])   # Output image depth

# Downsample layer (Conv + BN + ReLu)
def downsample_layer(no_filters, kernel_size, conv_strides=2, 
        padding='same', use_bn=True, lrelu_alpha=0.3, k_bias=True):
    # Initializer as in arxiv paper (section 6.2)
    kinit = tfinit.RandomNormal(mean=0.0, stddev=0.02)
    step_layers = keras.Sequential()    # Container for layers
    # Convolution
    step_layers.add(keras.layers.Conv2D(no_filters, kernel_size, 
        conv_strides, padding, kernel_initializer=kinit, 
        use_bias=k_bias))
    # Use batch normalization
    if use_bn:
        step_layers.add(keras.layers.BatchNormalization())
    # ReLU layer
    step_layers.add(keras.layers.LeakyReLU(alpha=lrelu_alpha))
    return step_layers  # All layers for downsampling

# Upsampling layer (Upconv + BN + ReLU)
def upsample_layer(no_filters, kernel_size, upconv_strides=2, 
        padding='same', use_bn=True, use_dpo=True, lrelu_alpha=0.3, 
        k_bias=True):
    # Initializer as in arxiv paper (section 6.2)
    kinit = tfinit.RandomNormal(mean=0.0, stddev=0.02)
    step_layers = keras.Sequential()    # Container for layers
    # Up-convolution
    step_layers.add(keras.layers.Conv2DTranspose(no_filters, 
        kernel_size, upconv_strides, padding='same', use_bias=k_bias,
        kernel_initializer=kinit))
    # Use batch normalization
    if use_bn:
        step_layers.add(keras.layers.BatchNormalization())
    # Use dropout
    if use_dpo:
        step_layers.add(keras.layers.Dropout(0.5))
    # ReLU layer
    step_layers.add(keras.layers.LeakyReLU(alpha=lrelu_alpha))
    return step_layers

# Testing layers through a test image
test_img = tf.constant(np.random.rand(1, *in_img.shape), 
    dtype=tf.float32)
# Downsample
down_model = downsample_layer(3, 4)
down_result = down_model(test_img)
print(f"Downsample: {test_img.shape} -> {down_result.shape}")
# Upsample
up_model = upsample_layer(3, 4)
up_result = up_model(down_result)
print(f"Upsample: {down_result.shape} -> {up_result.shape}")

# ==== Generator ====
# Create a Generator model
"""
    enc_ks:   Kernel strides for encoder
    enc_depths:     The number of kernels per convolution (downsample)
                    layer
    dec_ks:     Kernel strides for decoder
    dec_depths:     The number of kernels per convolution (upsample)
                    layer
    dec_dropouts:   The droupout conditions for the decoder layers
"""
def GeneratorModel(enc_ks=4, enc_depths=[64, 128, 256, *(5*[512])],
        dec_ks=4, dec_depths=[*(4*[512]), 256, 128, 64], 
        dec_dropouts=[*(3*[True]), *(4*[False])]):
    # Input placeholder (not including batch axis)
    batch_in = keras.Input(shape=[IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS])
    # Encoder is downsampling
    encoder_layers = []
    # First layer (no batch normalization - Sec. 6.1.1 arxiv paper)
    encoder_layers.append(downsample_layer(enc_depths[0], enc_ks, 
        use_bn=False, lrelu_alpha=0.2))
    # Add remaining layers with batch normalization
    for i in range(1, len(enc_depths)):
        encoder_layers.append(downsample_layer(enc_depths[i], enc_ks, 
            lrelu_alpha=0.2))
    # Decoder is upsampling
    decoder_layers = []
    # Add all layers to decoder
    for i in range(len(dec_depths)):
        decoder_layers.append(upsample_layer(dec_depths[i], dec_ks, 
            use_dpo=dec_dropouts[i], lrelu_alpha=0.0))
    # Last layer for convolution (upsample from decoder to out_img)
    last_layer = keras.layers.Conv2DTranspose(OUT_CHANNELS, dec_ks, 2, 
        padding='same', activation=keras.activations.tanh,
        kernel_initializer=tfinit.RandomNormal(mean=0.0, stddev=0.02))
    # ===== Main traversal of inputs =====
    x = batch_in
    cache_skips = []    # Cache outputs for skip connections
    # Downsampling through the encoder model
    for ed in encoder_layers:
        x = ed(x)   # Pass through layer
        cache_skips.append(x)   # Cache output
    # Flip (latest output is the first skip to decoder)
    cache_skips = reversed(cache_skips[:-1])    # Last is 1x1xD
    # Upsampling through the decoder model
    for du, skip in zip(decoder_layers, cache_skips):
        x = du(x)   # Pass through layer
        x = tf.keras.layers.Concatenate()([x, skip])
    # Through the last layer
    x = last_layer(x)
    # Final model
    gen_model = keras.Model(inputs=batch_in, outputs=x)
    return gen_model    # Generator model

# Test generator
test_img = tf.constant(
    np.random.rand(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS),
    dtype=tf.float32)
gen_model = GeneratorModel()
out_img = gen_model(test_img)
print(f"Generator shape: {test_img.shape} -> {out_img.shape}")
# Show the generator model
# keras.utils.plot_model(gen_model, show_shapes=True, 
#     to_file=f"{out_dir}/test_model.png")
# gen_model.summary(expand_nested=True)

# ==== Discriminator ====
# Create a Discriminator model
def DiscriminatorModel():
    # Initializer
    kinit = tfinit.RandomNormal(mean=0.0, stddev=0.02)
    # Two inputs
    in_img = tf.keras.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 
            IN_CHANNELS],
        name="input_img")       # Input in data
    out_img = tf.keras.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 
            IN_CHANNELS],
        name="pred_img_data")   # Prediction (target/true out) in data
    # Downsampling convolutions (sequential model(s))
    seq_model = keras.Sequential()
    seq_model.add(downsample_layer(64, 4, use_bn=False, 
        lrelu_alpha=0.2))
    seq_model.add(downsample_layer(128, 4, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(256, 4, lrelu_alpha=0.2))

    # seq_model.add(downsample_layer(512, 4, lrelu_alpha=0.2))

    # seq_model.add(keras.layers.ZeroPadding2D())
    # seq_model.add(keras.layers.Conv2D(512, 4, 
    #     kernel_initializer=kinit))
    # seq_model.add(keras.layers.BatchNormalization())
    # seq_model.add(keras.layers.LeakyReLU(alpha=0.02))
    # seq_model.add(keras.layers.ZeroPadding2D())
    # seq_model.add(keras.layers.Conv2D(1, 4,
    #     kernel_initializer=kinit))

    seq_model.add(keras.layers.ZeroPadding2D())
    seq_model.add(downsample_layer(512, 4, 1, 'valid', 
        lrelu_alpha=0.2))
    seq_model.add(keras.layers.ZeroPadding2D())
    seq_model.add(downsample_layer(1, 4, 1, 'valid', use_bn=False, 
        lrelu_alpha=1)) # No ReLU in end (alpha = 1 => y = x line)
    # Use any of the sequential models above

    # ====== Main data flow ======
    # Concatenate the images (channel-wise)
    x = keras.layers.Concatenate()([in_img, out_img])
    x = seq_model(x)
    disc_model = keras.Model(inputs=[in_img, out_img], outputs=x)
    return disc_model

# Test the discriminator
a = tf.constant(np.random.rand(1, 256, 256, 3), dtype=tf.float32)
b = tf.constant(np.random.rand(1, 256, 256, 3), dtype=tf.float32)
disc_model = DiscriminatorModel()
out_img = disc_model([a, b])
print(f"Input shapes: {a.shape} -> {out_img.shape}")
# keras.utils.plot_model(disc_model, show_shapes=True, 
#     expand_nested=True, to_file=f"{out_dir}/test_model.png")
# disc_model.summary(expand_nested=True)


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

# Optimizers
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
    # Log checkpoint if applicable
    if (epoch + 1) % EPOCH_CKPT_FREQ == 0:
        print(f"Epoch: {epoch+1} completed, saving checkpoint", 
            flush=True)
        ckpt_handler.save(ckpt_prefix)

# %%
