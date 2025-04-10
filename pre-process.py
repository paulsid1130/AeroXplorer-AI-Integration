import numpy as np
import cv2
import math
import PIL
import tensorflow as tf
import os
import random
from PIL import Image
import io
import shutil

def main(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(100, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

"""
def expose_image(image_path):
  img_array = tf.keras.utils.img_to_array(image_path)
  img_object = []
  #create rejectable data set from accepted images?
 for x in range(25, 50, 5):
     betVal = x
     adjusted_img = cv2.convertScaleAbs(img_array, alpha=1.0, beta=betVal)
     adjusted_img2 = cv2.convertScaleAbs(img_array, alpha=1.0, beta=-betVal)
     img_object.append(adjusted_img)
     img_object.append(adjusted_img2)
  #num = random.choice([-1, 1])
  #if num==1:
    #betVal = random.randrange(25, 45, 2)
    #adjusted_img = cv2.convertScaleAbs(img_array, alpha=1.0, beta=betVal)
  #else:
    #betVal = random.randrange(25,45, 2)
    #adjusted_img = cv2.convertScaleAbs(img_array, alpha=1.0, beta=-betVal)
  #return adjusted_img
"""

def rotate_image(image, angle):

    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    res = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return res


def largest_rotated_rect(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr


def crop_around_center(image, width, height):

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def tilt_images(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_height, image_width = img.shape[0:2]
    img_object = []
    for i in range(0.6, 3):
       image_rotated = rotate_image(img, i)
       image_rotated = rotate_image(img, -i)
       image_rotated_cropped2 = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(i)
            )
        )
       image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(i)
            )
        )
       adjusted_img2 = cv2.cvtColor(image_rotated_cropped2, cv2.COLOR_RGB2BGR)
       img_object.append(adjusted_img2)
       adjusted_img = cv2.cvtColor(image_rotated_cropped, cv2.COLOR_RGB2BGR)
       img_object.append(adjusted_img)
    #i = random.uniform(0.6, 3)
    #i = random.choice([-1, 1]) * i
    #image_rotated = rotate_image(img, i)
    #image_rotated_cropped = crop_around_center(
        #image_rotated,
        #*largest_rotated_rect(
            #image_width,
            #image_height,
            #math.radians(i)
            #)
        #)
    #adjusted_img = cv2.cvtColor(image_rotated_cropped, cv2.COLOR_RGB2BGR)

"""
def crop_imageinner(img_np):
    img = Image.fromarray(img_np)
    width, height = img.size
    new_width = int(width * 0.66)

    # Calculate new height to maintain 3:2 aspect ratio
    new_height = int(new_width * 2 / 3)

    # Ensure the new height does not exceed the original height
    if new_height > height:
        new_height = height
        new_width = int(new_height * 3 / 2)

    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)

    right = left + new_width
    bottom = top + new_height

    cropped_img = img.crop((left, top, right, bottom))
    cropped_img_np = np.array(cropped_img)

    return cropped_img_np

def crop_image(img_path):
   img = cv2.imread(img_path)
   img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   cropped_img = crop_image(img)
   cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
"""