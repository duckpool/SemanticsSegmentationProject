import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

LEARN_RATE = 9e-5


def load_vgg(sess, vgg_path):
    #to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, layer3, layer4, layer7

print("\nPassed load_vgg function")


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    # KK Hyperparameters setting
    l2_value = 1e-3
    kernel_reg = tf.contrib.layers.l2_regularizer(l2_value)
    stddev = 1e-3
    kernel_init = tf.random_normal_initializer(stddev=stddev)

    # KK 1x1 convolution to preserve spatial information
    conv_1x1_7 = tf.layers.conv2d(vgg_layer7_out, num_classes,
                                  kernel_size=1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_regularizer=kernel_reg,
                                  kernel_initializer=kernel_init)

    # KK Print the shape of the 1x1
    tf.Print(conv_1x1_7, [tf.shape(conv_1x1_7)[1:3]])

    # KK Upsample by 2x so we can add it with layer4 in the skip layer to follow
    conv7_2x = tf.layers.conv2d_transpose(conv_1x1_7, num_classes,
                                          kernel_size=4,
                                          strides=(2, 2),
                                          padding='same',
                                          kernel_regularizer=kernel_reg,
                                          kernel_initializer=kernel_init)

    # KK Print the shape of the upsample
    print( '\nUpsampled layer 7 = ', tf.Print(conv7_2x, [tf.shape(conv7_2x)[1:3]]) )

    # KK 1x1 convolution to preserve spatial information
    conv_1x1_4 = tf.layers.conv2d(vgg_layer4_out, num_classes,
                                  kernel_size=1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_regularizer=kernel_reg,
                                  kernel_initializer=kernel_init)

    # KK Add the layer4 with the upsampled 1x1 convolution as a skip layer
    skip_4_to_7 = tf.add(conv7_2x, conv_1x1_4)

    # KK Upsample the combined layer4 and 1x1 by 2x
    upsample2x_skip_4_to_7 = tf.layers.conv2d_transpose(skip_4_to_7, num_classes,
                                                        kernel_size=4,
                                                        strides=(2, 2),
                                                        padding='same',
                                                        kernel_regularizer=kernel_reg,
                                                        kernel_initializer=kernel_init)

    # KK Print the resulting shape
    print( '\nUpsampled 4 and 7 = ', tf.Print(upsample2x_skip_4_to_7, [tf.shape(upsample2x_skip_4_to_7)[1:3]]))

    # KK 1x1 convolution to preserve spatial information
    conv_1x1_3 = tf.layers.conv2d(vgg_layer3_out, num_classes,
                                  kernel_size=1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_regularizer=kernel_reg,
                                  kernel_initializer=kernel_init)

    # KK Add layer 3 with the upsampled skip1 layer
    skip_3 = tf.add(upsample2x_skip_4_to_7, conv_1x1_3)

    # KK Upsample by 8x to get to original image size
    output = tf.layers.conv2d_transpose(skip_3, num_classes,
                                        kernel_size=16,
                                        strides=(8, 8),
                                        padding='same',
                                        kernel_regularizer=kernel_reg,
                                        kernel_initializer=kernel_init)

    # KK Print the resulting shape which should be the original image size
    print('\nShape of output image = ', tf.Print(output, [tf.shape(output)[1:3]]))

    return output

print("\nPassed layers function")


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    # KK Get the logits of the network
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # KK Get the loss of the network
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    #KK Regularization loss collector....Don't really understand this
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.01  # Choose an appropriate one.
    loss = cross_entropy_loss + reg_constant * sum(reg_losses)

    # KK Minimize the loss using Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, loss


print("\nPassed Optimizer")



def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    # KK loop through epochs
    for epoch in range(epochs):
        print('    Training Epoch # {}    '.format(epoch))
        # KK loop through images and labels
        for image, label in get_batches_fn(batch_size):
            #DEBUG
            print("\nTraining image shape = {}".format(tf.shape(image)))
            print("\nTraining label shape = {}".format(tf.shape(label)))
            # Training
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: LEARN_RATE})
            print('\nTraining Loss = {:.3f}'.format(loss))
    pass


print("\nPassed training function")

def graph_visualize():
    # Path to vgg model
    data_dir = './data'
    vgg_path = os.path.join(data_dir, 'vgg')

    with tf.Session() as sess:
        model_filename = os.path.join(vgg_path, 'saved_model.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
    LOGDIR = '.'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = 'data'
    runs_dir = 'runs'

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        #Tensorflow placeholders
        print("Setting up placeholders")
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        #Train NN using the train_nn function KK-DONE
        epochs = 1
        batch_size = 4

        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        #Save inference data using helper.save_inference_samples KK-DONE
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

if __name__ == '__main__':
    run()
