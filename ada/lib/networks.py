# All the networks for Pix2Pix
"""
    Contains the neural networks for Pix2Pix. Contains the following
    - Building blocks:
        - downsample_layer: A downsampling convolution
        - upsample_layer: An upsampling convolution
    - Main blocks:
        - 
"""

# %% Import everything
# TensorFlow imports
import tensorflow as tf
# Some more imports to make life easier
from tensorflow import initializers as tfinit
from tensorflow import keras

# %%
# Downsample layer (Conv + BN + ReLu)
def downsample_layer(no_filters, kernel_size, conv_strides=2, 
        padding='same', use_bn=True, lrelu_alpha=0.3, k_bias=True):
    r"""
        A downsampling layer has Convolution + Batch Normalization + 
        Leaky ReLU. This is used in the Encoder (of Generator) and 
        Discriminator.

        Parameters:
        - no_filters: Number of filters for Conv2D
        - kernel_size: Kernel size for Conv2D
        - conv_strides: Stride for the convolution
        - padding: Padding (for input to convolution)
        - use_bn: If True, add Batch Normalization
        - lrelu_alpha: Slope for leaky ReLU (if 1 => y = x, no ReLU)
        - k_bias: If True, the use bias for Conv2D kernels
    """
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
    # Leaky ReLU layer
    step_layers.add(keras.layers.LeakyReLU(alpha=lrelu_alpha))
    return step_layers  # All layers for downsampling

# %%
# Upsampling layer (Upconv + BN + ReLU)
def upsample_layer(no_filters, kernel_size, upconv_strides=2, 
        padding='same', use_bn=True, use_dpo=True, lrelu_alpha=0.3, 
        k_bias=True):
    r"""
        An upsampling layer has Convolution Transpose + Batch 
        Normalization + Dropout + Leaky ReLU. This is used by the
        decoder (of Generator).

        Parameters:
        - no_filters: Number of filters for Conv2D
        - kernel_size: Kernel size for Conv2D
        - upconv_strides: Stride for the up-convolution
        - padding: Padding (for input to convolution)
        - use_bn: If True, add Batch Normalization
        - use_dpo: If True, add dropout layer (p = 0.5)
        - lrelu_alpha: Slope for leaky ReLU (if 1 => y = x, no ReLU)
        - k_bias: If True, the use bias for Conv2D kernels
    """
    # Initializer as in arxiv paper (section 6.2)
    kinit = tfinit.RandomNormal(mean=0.0, stddev=0.02)
    step_layers = keras.Sequential()    # Container for layers
    # Up-convolution
    step_layers.add(keras.layers.Conv2DTranspose(no_filters, 
        kernel_size, upconv_strides, padding=padding, use_bias=k_bias,
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

# %%
# ==== Generator ====
# Create a Generator model
def GeneratorModel(img_h, img_w, in_ch, out_ch, enc_ks=4, 
        enc_depths=[64, 128, 256, *(5*[512])], dec_ks=4, 
        dec_depths=[*(4*[512]), 256, 128, 64], 
        dec_dropouts=[*(3*[True]), *(4*[False])]):
    r"""
        Generator model

        img_h:      Image height
        img_w:      Image width
        in_ch:      Input channels
        out_ch:     Output channels
        enc_ks:   Kernel strides for encoder
        enc_depths:     The number of kernels per convolution 
                        (downsample) layer
        dec_ks:     Kernel strides for decoder
        dec_depths:     The number of kernels per convolution 
                        (upsample) layer
        dec_dropouts:   The droupout conditions for the decoder layers
    """
    # Input placeholder (not including batch axis)
    batch_in = keras.Input(shape=[img_h, img_w, in_ch])
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
    last_layer = keras.layers.Conv2DTranspose(out_ch, dec_ks, 2, 
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
    cache_skips = list(reversed(cache_skips[:-1]))    # Last is 1x1xD
    # Upsampling through the decoder model
    for i, du in enumerate(decoder_layers):
        x = du(x)   # Pass through layer
        if i < len(cache_skips):    # Skips left to add
            x = tf.keras.layers.Concatenate()([x, cache_skips[i]])
    # Through the last layer
    x = last_layer(x)
    # Final model
    gen_model = keras.Model(inputs=batch_in, outputs=x)
    return gen_model    # Generator model

# %%
# ==== Discriminator ====
# Create a Discriminator model (Receptive Field = 70x70)
def DiscriminatorModel_70x70(img_h, img_w, in_c):
    # Initializer
    kinit = tfinit.RandomNormal(mean=0.0, stddev=0.02)
    # Two inputs
    in_img = tf.keras.Input(shape=[img_h, img_w, in_c],
        name="input_img")       # Input in data
    out_img = tf.keras.Input(shape=[img_h, img_w, in_c],
        name="pred_img_data")   # Prediction (target/true out) in data
    # Downsampling convolutions (sequential model(s))
    seq_model = keras.Sequential()
    seq_model.add(downsample_layer(64, 4, use_bn=False, 
        lrelu_alpha=0.2))
    seq_model.add(downsample_layer(128, 4, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(256, 4, lrelu_alpha=0.2))
    # --- Maybe we could use this!?
    # seq_model.add(downsample_layer(512, 4, lrelu_alpha=0.2))
    # --- This seems too messy!
    # seq_model.add(keras.layers.ZeroPadding2D())
    # seq_model.add(keras.layers.Conv2D(512, 4, 
    #     kernel_initializer=kinit))
    # seq_model.add(keras.layers.BatchNormalization())
    # seq_model.add(keras.layers.LeakyReLU(alpha=0.02))
    # seq_model.add(keras.layers.ZeroPadding2D())
    # seq_model.add(keras.layers.Conv2D(1, 4,
    #     kernel_initializer=kinit))
    # --- Okay, this works! :)
    seq_model.add(keras.layers.ZeroPadding2D())
    seq_model.add(downsample_layer(512, 4, 1, 'valid', 
        lrelu_alpha=0.2))
    seq_model.add(keras.layers.ZeroPadding2D())
    seq_model.add(downsample_layer(1, 4, 1, 'valid', use_bn=False, 
        lrelu_alpha=1)) # No ReLU in end (alpha = 1 => y = x line)
    # Use any of the sequential models above
    # ====== Main data flow ======
    # Concatenate the images (channel-wise) (out_img == pred_img)
    x = keras.layers.Concatenate()([in_img, out_img])
    x = seq_model(x)
    disc_model = keras.Model(inputs=[in_img, out_img], outputs=x)
    return disc_model

# %%
# ==== Discriminator ====
# Create a Discriminator model (Receptive Field = 1x1)
def DiscriminatorModel_1x1(img_h, img_w, in_c):
    # Input placeholders
    in_img = tf.keras.Input(shape=[img_h, img_w, in_c],
        name="input_img")       # Input in data
    out_img = tf.keras.Input(shape=[img_h, img_w, in_c],
        name="pred_img_data")   # Prediction (target/true out) in data
    # Sequential model
    seq_model = keras.Sequential()
    seq_model.add(downsample_layer(64, 1, 1, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(128, 1, 1, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(1, 1, 1, lrelu_alpha=1)) # No ReLU
    # Pass data through it
    x = keras.layers.Concatenate()([in_img, out_img])
    x = seq_model(x)
    disc_model = keras.Model(inputs=[in_img, out_img], outputs=x)
    return disc_model

# %%
# ==== Discriminator ====
# Create a Discriminator model (Receptive Field = 16x16)
def DiscriminatorModel_16x16(img_h, img_w, in_c):
    # Input placeholders
    in_img = tf.keras.Input(shape=[img_h, img_w, in_c],
        name="input_img")       # Input in data
    out_img = tf.keras.Input(shape=[img_h, img_w, in_c],
        name="pred_img_data")   # Prediction (target/true out) in data
    # Sequential model
    seq_model = keras.Sequential()
    seq_model.add(downsample_layer(64, 4, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(128, 4, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(1, 1, 1, lrelu_alpha=1)) # No ReLU
    # Pass data through it
    x = keras.layers.Concatenate()([in_img, out_img])
    x = seq_model(x)
    disc_model = keras.Model(inputs=[in_img, out_img], outputs=x)
    return disc_model

# %%
# ==== Discriminator ====
# Create a Discriminator model (Receptive Field = 286x286)
def DiscriminatorModel_286x286(img_h, img_w, in_c):
    # Input placeholders
    in_img = tf.keras.Input(shape=[img_h, img_w, in_c],
        name="input_img")       # Input in data
    out_img = tf.keras.Input(shape=[img_h, img_w, in_c],
        name="pred_img_data")   # Prediction (target/true out) in data
    # Sequential model
    seq_model = keras.Sequential()
    seq_model.add(downsample_layer(64, 4, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(128, 4, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(256, 4, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(512, 4, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(512, 4, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(512, 4, lrelu_alpha=0.2))
    seq_model.add(downsample_layer(1, 1, 1, lrelu_alpha=1)) # No ReLU
    # Pass data through it
    x = keras.layers.Concatenate()([in_img, out_img])
    x = seq_model(x)
    disc_model = keras.Model(inputs=[in_img, out_img], outputs=x)
    return disc_model
