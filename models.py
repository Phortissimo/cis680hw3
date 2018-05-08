import tensorflow as tf
from layers import *

def quick_cnn(x, labels, locs, mask, c_num, batch_size, is_train, reuse):
  with tf.variable_scope('C', reuse=reuse) as vs:

    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 32 
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    print("finished layer 1")

    # Uncomment to see vinishing gradients
    # for l in range(20):
    #   with tf.variable_scope('rd_conv_'+str(l), reuse=reuse):
    #     x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      print(x.shape)
      x = conv_factory(x, 64, 5, 1, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    print("finished layer 2")

    # conv3
    with tf.variable_scope('conv3', reuse=reuse):
      hidden_num = 128
      print(x.shape)
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    print("finished layer 3")

    # conv4
    with tf.variable_scope('conv4', reuse=reuse):
      hidden_num = 256
      print(x.shape)
      x = conv_factory2(x, hidden_num, 3, 1, is_train, reuse)
      #x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print("finished layer 4")

    # conv5
    with tf.variable_scope('conv5', reuse=reuse):
      hidden_num = 256
      print(x.shape)
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse)
    print("finished layer 5")

    # conv6
    with tf.variable_scope('conv6', reuse=reuse):
      hidden_num = 1
      x = conv_factorynorect(x, hidden_num, 1, 1, is_train, reuse)
      print(x.shape)
    print("finished layer 6")
    feat = x
   #  # fc4
   #  with tf.variable_scope('fc4', reuse=reuse):
   #    x = tf.reshape(x, [batch_size, -1])
   #    x = fc_factory(x, hidden_num, is_train, reuse)
   #    print(x.shape)
   #  feat = x
   #  print("finished layer 4")
   #  x = tf.reshape(x, [batch_size, -1])
   # # dropout
   #  if is_train:
   #    x = tf.nn.dropout(x, keep_prob=0.5)
   #
   #  # local5
   #  with tf.variable_scope('fc5', reuse=reuse):
   #
   #    W = tf.get_variable('weights', [512, c_num], initializer = tf.contrib.layers.variance_scaling_initializer())
   #    x = tf.matmul(x, W)
   #    feat = x

    # # Softmax
    # with tf.variable_scope('sm', reuse=reuse):
    #   loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
    #   accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))
    # print("finished softmax")
    with tf.variable_scope('sig', reuse=reuse):
      x = tf.reshape(x, [batch_size, -1])
      mask = tf.reshape(mask, [batch_size, -1])
      loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=mask)
      acc = tf.to_float(tf.equal(tf.round(x), mask))
      c_loss = tf.constant(0, dtype=tf.float32)
      accuracy = tf.constant(0, dtype=tf.float32)
      two = tf.constant(2, dtype=tf.float32)

      valid_regions = tf.where(tf.not_equal(mask, two))
      c_loss = tf.reduce_mean(tf.gather_nd(loss, valid_regions))
      accuracy = tf.reduce_mean(tf.gather_nd(acc, valid_regions))
      # for i in range(0, batch_size):
      #   valid_regions = tf.where(tf.not_equal(tf.gather(mask, i), two))
      #   #b = tf.gather(valid_regions, 1)
      #   losst = tf.gather(tf.gather(loss, i), valid_regions)
      #   #print(losst.shape)
      #   losst = tf.reduce_mean(losst)
      #
      #   accuracyt = tf.gather(tf.gather(acc, i), valid_regions)
      #   accuracyt = tf.reduce_mean(accuracyt)
      #
      #   c_loss = tf.add(losst, c_loss)
      #   accuracy = tf.add(accuracyt, accuracy)
      #
      # c_loss = tf.divide(c_loss, batch_size)
      # accuracy = tf.divide(accuracy, batch_size)

  variables = tf.contrib.framework.get_variables(vs)
  return c_loss, feat, accuracy, variables
