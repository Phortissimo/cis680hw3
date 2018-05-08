import os
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from PIL import Image

trainim = np.zeros((10000, 48, 48, 3))
trainl = np.zeros((10000, 1))
trainb = np.zeros((10000, 4))
trainmask = np.zeros((10000, 6, 6, 1))
labnames = 0
testim = np.zeros((10000, 48, 48, 3))
testl = np.zeros((10000, 1))
testb = np.zeros((10000, 4))
testmask = np.zeros((10000, 6, 6, 1))

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# def load():
#     global trainim, testim, trainl, testl, labnames
#     d = unpickle('data_batch_1')
#     #print(d[b'data'].shape)
#     trainim = np.moveaxis(np.reshape(d[b'data'][:10000], (10000, 3, 32, 32)), 1, -1)
#     trainl = d[b'labels'][:10000]
#     labnames = unpickle('batches.meta')[b'label_names']
#     d = unpickle('test_batch')
#     testim = np.moveaxis(np.reshape(d[b'data'][:10000], (10000, 3, 32, 32)), 1, -1)
#     testl = d[b'labels'][:10000]
#     # print(testl[1])
#     # plt.imshow(testim[1])
#     # plt.show()

def load():
    global trainim, testim, trainl, trainb, testl, testb, testmask, trainmask

def read_labeled_image_list(img_list_path, img_dir, mask_dir):
  """Reads a .txt file containing pathes and labeles
  Args:
    img_list_path: a .txt file with one /path/to/image with one label per line
    img_dir: path of directory that contains images
  Returns:
    List with all filenames
  """
  f = open(img_list_path, 'r')
  img_paths = []
  mask_paths = []
  labs = []
  loc = []
  for line in f:
    img_name, lab, x, y, w = line[:-1].split(' ')
    img_paths.append(img_dir + img_name)
    mask_paths.append(mask_dir + img_name)
    labs.append(int(lab))
    loc.append([int(x), int(y), int(w)])
  f.close()
  return img_paths, labs, loc, mask_paths

def read_images_from_disk(input_queue):
  """Consumes a single filename and label as a ' '-delimited string
  Args:
    filename_and_label_tensor: A scalar string tensor
  Returns:
    Two tensors: the decoded image, and the string label
  """
  lab = input_queue[1]
  loc = input_queue[2]
  img_path = tf.read_file(input_queue[0])
  mask_path = tf.read_file(input_queue[3])
  img = tf.image.decode_png(img_path, channels=3)
  mask = tf.image.decode_png(mask_path, channels=1)

  return img, lab, loc, mask

def get_loader(root, batch_size, batchnum, split=None, shuffle=True):
  """ Get a data loader for tensorflow computation graph
  Args:
    root: Path/to/dataset/root/, a string
    batch_size: Batch size, a integer
    split: Data for train/val/test, a string
    shuffle: If the data should be shuffled every epoch, a boolean
  Returns:
    img_batch: A (float) tensor containing a batch of images.
    lab_batch: A (int) tensor containing a batch of labels.
   """

  global trainim, testim, trainl, trainb, testl, testb, testmask, trainmask
  with tf.device('/cpu:0'):
      if batchnum == 5:
          img_paths_np, labs_np, locs_np, mask_paths_np = read_labeled_image_list('./data/labels/' + 'test' + '.txt', './data/imgs/', './data/masks/')
          img_paths = tf.convert_to_tensor(img_paths_np, dtype=tf.string)
          mask_paths = tf.convert_to_tensor(mask_paths_np, dtype=tf.string)
          labs = tf.convert_to_tensor(labs_np, dtype=tf.int64)
          locs = tf.convert_to_tensor(locs_np, dtype=tf.int64)

          input_queue = tf.train.slice_input_producer([img_paths, labs, locs, mask_paths], shuffle=shuffle, capacity=100 * batch_size)
          img, lab, loc, mask = read_images_from_disk(input_queue)
          img.set_shape([48, 48, 3])
          img = tf.cast(img, tf.float32)
          mask.set_shape([6, 6, 1])
          mask = tf.cast(mask, tf.float32)
          #input_queue[0] = tf.image.per_image_standardization(input_queue[0])
          img_batch, lab_batch, loc_batch, mask_batch = tf.train.batch([img, lab, loc, mask], num_threads=1, batch_size=batch_size, capacity=100*batch_size)
      else:
          img_paths_np, labs_np, locs_np, mask_paths_np = read_labeled_image_list('./data/labels/' + 'train' + '.txt', './data/imgs/', './data/masks/')
          img_paths = tf.convert_to_tensor(img_paths_np, dtype=tf.string)
          mask_paths = tf.convert_to_tensor(mask_paths_np, dtype=tf.string)
          labs = tf.convert_to_tensor(labs_np, dtype=tf.int64)
          locs = tf.convert_to_tensor(locs_np, dtype=tf.int64)

          input_queue = tf.train.slice_input_producer([img_paths, labs, locs, mask_paths], shuffle=shuffle, capacity=100 * batch_size)
          img, lab, loc, mask = read_images_from_disk(input_queue)
          img.set_shape([48, 48, 3])
          img = tf.cast(img, tf.float32)
          mask.set_shape([6, 6, 1])
          mask = tf.cast(mask, tf.float32)

          #input_queue[0] = tf.image.random_flip_left_right(input_queue[0])
          #input_queue[0] = tf.image.per_image_standardization(input_queue[0])
          # paddings = tf.constant([[4, 4], [4, 4], [0, 0]])
          # input_queue[0] = tf.pad(input_queue[0], paddings, "CONSTANT")
          # input_queue[0] = tf.random_crop(input_queue[0], [32, 32, 3])

          img_batch, lab_batch, loc_batch, mask_batch = tf.train.batch([img, lab, loc, mask], num_threads=1, batch_size=batch_size, capacity=100 * batch_size)

      return img_batch, lab_batch, loc_batch, mask_batch



