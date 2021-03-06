import tensorflow as tf
slim = tf.contrib.slim
import numpy as np

def conv_factory(x, hidden_num, kernel_size, stride, is_train, reuse):
  vs = tf.get_variable_scope()
  in_channels = x.get_shape()[3]
  W = tf.get_variable('weights', [kernel_size,kernel_size,in_channels,hidden_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
  b = tf.get_variable('biases', [1, 1, 1, hidden_num],
        initializer = tf.constant_initializer(0.0))

  x = tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')
  print("conv")
  #x = slim.batch_norm(x, is_training=is_train, reuse=reuse, scale=True, fused=True, scope=vs, updates_collections=None)
  x = batch_norm(x, is_train=is_train)
  x = tf.nn.relu(x)
#  x = tf.nn.sigmoid(x)
  return x

def conv_factorynorect(x, hidden_num, kernel_size, stride, is_train, reuse):
  vs = tf.get_variable_scope()
  in_channels = x.get_shape()[3]
  W = tf.get_variable('weights', [kernel_size,kernel_size,in_channels,hidden_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
  b = tf.get_variable('biases', [1, 1, 1, hidden_num],
        initializer = tf.constant_initializer(0.0))

  x = tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')
  print("conv")
  #x = slim.batch_norm(x, is_training=is_train, reuse=reuse, scale=True, fused=True, scope=vs, updates_collections=None)
  #x = batch_norm(x, is_train=is_train)
#  x = tf.nn.sigmoid(x)
  return x

#mobilenet
def conv_factory2(x, hidden_num, kernel_size, stride, is_train, reuse):
  vs = tf.get_variable_scope()
  in_channels = x.get_shape()[3]
  W = tf.get_variable('weights', [kernel_size,kernel_size,in_channels,1],
        initializer = tf.contrib.layers.variance_scaling_initializer())
  b = tf.get_variable('biases', [1, 1, 1, in_channels],
        initializer = tf.constant_initializer(0.0))

  x = tf.nn.depthwise_conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')
  print("conv")
  #x = slim.batch_norm(x, is_training=is_train, reuse=reuse, scale=True, fused=True, scope=vs, updates_collections=None)
  x = batch_norm(x, is_train=is_train)
  x = tf.nn.relu(x)

  W2 = tf.get_variable('weights2', [1, 1, in_channels, hidden_num], initializer=tf.contrib.layers.variance_scaling_initializer())
  b2 = tf.get_variable('biases2', [1, 1, 1, hidden_num], initializer=tf.constant_initializer(0.0))
  x = tf.nn.conv2d(x, W2, strides=[1,stride,stride,1], padding='SAME')
  #x = slim.batch_norm(x, is_training=is_train, reuse=False, scale=True, fused=False, updates_collections=None)
  with tf.variable_scope('batch2',reuse=reuse):
    x = batch_norm(x, is_train=is_train)
    x = tf.nn.relu(x)

  return x

def conv_factory3(x, hidden_num, kernel_size, stride, is_train, reuse):
  vs = tf.get_variable_scope()
  x2 = x
  in_channels = x.get_shape()[3]
  W = tf.get_variable('weights', [kernel_size, kernel_size, in_channels, hidden_num],
                      initializer=tf.contrib.layers.variance_scaling_initializer())
  b = tf.get_variable('biases', [1, 1, 1, hidden_num],
                      initializer=tf.constant_initializer(0.0))
  W2 = tf.get_variable('weights2', [kernel_size, kernel_size, in_channels, hidden_num],
                      initializer=tf.contrib.layers.variance_scaling_initializer())
  b2 = tf.get_variable('biases2', [1, 1, 1, hidden_num],
                      initializer=tf.constant_initializer(0.0))
  x = tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')
  print("conv")
  #x = slim.batch_norm(x, is_training=is_train, reuse=reuse, scale=True, fused=True, scope=vs, updates_collections=None)
  x = batch_norm(x, is_train=is_train)
  x = tf.nn.relu(x)
  x = tf.nn.conv2d(x, W2, strides=[1,stride,stride,1], padding='SAME')
  x = batch_norm(x, is_train=is_train)

  W3 = tf.get_variable('weights', [1, 1, in_channels, hidden_num],
                      initializer=tf.contrib.layers.variance_scaling_initializer())
  x2 = tf.nn.conv2d(x2, W3, strides=[1,stride,stride,1], padding='SAME')
  x2 = batch_norm(x2, is_train=is_train)

  x = tf.add(x, x2)
  x = tf.nn.relu(x)

  return x


def fc_factory(x, hidden_num, is_train, reuse):

  vs = tf.get_variable_scope()
  in_channels = x.get_shape()[1]
  W = tf.get_variable('weights', [in_channels,hidden_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
  b = tf.get_variable('biases', [1, hidden_num],
        initializer = tf.constant_initializer(0.0))
  print("fc")
  x = tf.matmul(x, W)
  #x = slim.batch_norm(x, is_training=is_train, reuse=reuse, scale=True, fused=False, scope=vs, updates_collections=None)
  x = batch_norm(x, is_train=is_train)
  x = tf.nn.relu(x)
#  x = tf.nn.sigmoid(x)
  return x

def leaky_relu(x):
  alpha = 0.05
  pos = tf.nn.relu(x)
  neg = alpha * (x - abs(x)) * 0.5
  return pos + neg

def batch_norm(x, is_train=True, decay=0.99, epsilon=0.001):
  shape_x = x.get_shape().as_list()
  beta = tf.get_variable('beta', shape_x[-1], initializer=tf.constant_initializer(0.0))
  gamma = tf.get_variable('gamma', shape_x[-1], initializer=tf.constant_initializer(1.0))
  moving_mean = tf.get_variable('moving_mean', shape_x[-1],
                initializer=tf.constant_initializer(0.0), trainable=False)
  moving_var = tf.get_variable('moving_var', shape_x[-1],
               initializer=tf.constant_initializer(1.0), trainable=False)

  if is_train:
    mean, var = tf.nn.moments(x, np.arange(len(shape_x)-1), keep_dims=True)
    mean = tf.reshape(mean, [mean.shape.as_list()[-1]])
    var = tf.reshape(var, [var.shape.as_list()[-1]])

    update_moving_mean = tf.assign(moving_mean, moving_mean*decay + mean*(1-decay))
    update_moving_var = tf.assign(moving_var,
                        moving_var*decay + shape_x[0]/(shape_x[0]-1)*var*(1-decay))
    update_ops = [update_moving_mean, update_moving_var]

    with tf.control_dependencies(update_ops):
      return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)

  else:
    mean = moving_mean
    var = moving_var
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
