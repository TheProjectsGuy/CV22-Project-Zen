# Image utilities
"""
    Includes the image processing functions and the data input 
    pipeline.
"""

# %%
import tensorflow as tf

# %%
# Split image, return left and right images
def cut_image(img_fname: str):
    """
        Given a '.jpg' image filename, return the left and right parts
        of the image
    """
    # Read and decode
    img = tf.io.decode_jpeg(tf.io.read_file(img_fname))
    # Split image (half width cut)
    w = tf.shape(img)[1]//2 # Cut point
    left_img = img[:, :w]   # Prediction (first part)
    right_img = img[:, w:]     # Input image (second part)
    # Covert to float32
    left_img = tf.cast(left_img, tf.float32)
    right_img = tf.cast(right_img, tf.float32)
    return left_img, right_img

# %%
# Resize image
@tf.function()
def resize_imgs(left_img, right_img, to_height, to_width):
    # Input to generator
    left_img = tf.image.resize(left_img, (to_height, to_width), 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Prediction (for generator)
    right_img = tf.image.resize(right_img, (to_height, to_width),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Return both
    return left_img, right_img

# Normalize images
@tf.function()
def norm_imgs(left_img, right_img):
    # Input image (to the generator)
    left_img = (left_img/127.5) - 1
    # Output/prediction image (for the generator)
    right_img = (right_img/127.5) - 1
    return left_img, right_img

# Apply random crops to the images
@tf.function()
def random_crop_imgs(left_img, right_img, to_height, to_width):
    both_imgs = tf.stack([left_img, right_img], axis=0)
    cropped_imgs = tf.image.random_crop(both_imgs, 
        size=[2, to_height, to_width, 3])
    return cropped_imgs[0], cropped_imgs[1]

# Apply random jitter to an image (resize, crop & mirror)
@tf.function()
def apply_jitter(left_img, right_img, res_height, res_width):
    # Obtain final sizes (for output)
    to_height = int(tf.shape(left_img)[0])
    to_width = int(tf.shape(left_img)[1])
    # Resize image to larger size
    limg, rimg = resize_imgs(left_img, right_img, res_height, 
        res_width)
    # Apply a random crop
    l_cimg, r_cimg = random_crop_imgs(limg, rimg, to_height, to_width)
    # Apply random mirroring
    both_imgs = tf.stack([l_cimg, r_cimg])  # [2, H, W, 3]
    both_fimgs = tf.image.random_flip_left_right(both_imgs)
    l_fimg, r_fimg = both_fimgs[0], both_fimgs[1]   # Decouple
    # Return both images
    return l_fimg, r_fimg

# %%
# Training pipeline
def create_img_pipeline(res_h, res_w, training=True, lin_rout=True,
        resize_256 = True):
    # Main pipeline to be returned
    def preprocess_data(img_fname):
        # Load images from file name
        left_img, right_img = cut_image(img_fname)
        # Training or testing pipeline
        if training:
            # Paper recommends resizing images when training to 256
            if resize_256:
                left_img, right_img = resize_imgs(left_img, 
                    right_img, 256, 256)
            # Apply random jitter (for training data augmentation)
            left_img, right_img = apply_jitter(left_img, right_img, 
                res_h, res_w)
        # Normalize the image
        left_img, right_img = norm_imgs(left_img, right_img)
        # Convert image to float32
        left_img = tf.cast(left_img, dtype=tf.float32)
        right_img = tf.cast(right_img, dtype=tf.float32)
        # Input and output
        if lin_rout:
            in_img, pred_img = left_img, right_img
        else:
            in_img, pred_img = right_img, left_img
        # Return the images (input and prediction - for generator)
        return in_img, pred_img
    # Return function that does all preprocessing to data
    return preprocess_data
