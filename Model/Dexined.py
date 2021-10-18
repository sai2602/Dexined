import keras.optimizers
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose
from keras.layers import BatchNormalization, add
from keras.activations import relu
import tensorflow as tf
import numpy as np


class dexined_model():

    def __init__(self, input_shape=(320, 320, 3)):
        self.input_shape = input_shape

    def create_model(self):
        input_image = keras.Input(shape=self.input_shape)

        conv_1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='conv_1_1')(input_image)
        conv_1_1 = BatchNormalization()(conv_1_1)
        conv_1_1 = relu(conv_1_1)

        conv_1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name="conv_1_2")(conv_1_1)
        conv_1_2 = BatchNormalization()(conv_1_2)
        conv_1_2 = relu(conv_1_2)

        # ######################################### create output1 data with side layer function
        output1 = self.branch_layers(conv_1_2, name='output1', filters=1, upscale=int(2**1), strides=(1, 1),
                                     kernel_size=(1, 1), sub_pixel=True)

        rconv1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='SAME', name="rconv1")(conv_1_2)

        block2_xcp = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', strides=(1, 1), name="conv_block2_1")(conv_1_2)
        block2_xcp = BatchNormalization()(block2_xcp)
        block2_xcp = relu(block2_xcp)

        block2_xcp = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', strides=(1, 1), name="conv2_block2_1")(block2_xcp)
        block2_xcp = BatchNormalization()(block2_xcp)

        maxpool2_1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME')(block2_xcp)
        add2_1 = add([maxpool2_1, rconv1])

        # ######################################### create output2 data with side layer function
        output2 = self.branch_layers(block2_xcp, name='output2', filters=1, upscale=int(2 ** 1), strides=(1, 1),
                                     kernel_size=(1, 1), sub_pixel=True)

        rconv2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='SAME', name="rconv2")(add2_1)
        rconv2 = BatchNormalization()(rconv2)

        addb2_4b3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='SAME', name="addb2_4b3")(add2_1)
        addb2_4b3 = BatchNormalization()(addb2_4b3)
        block3_xcp = add2_1
        for k in range(2):
            block3_xcp = relu(block3_xcp)
            block3_xcp = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                                padding='SAME', name="conv1_block3_{}".format(k+1))(block3_xcp)
            block3_xcp = BatchNormalization()(block3_xcp)
            block3_xcp = relu(block3_xcp)

            block3_xcp = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                                padding='SAME', name="conv2_block3_{}".format(k + 1))(block3_xcp)
            block3_xcp = BatchNormalization()(block3_xcp)
            block3_xcp = add([block3_xcp, addb2_4b3])/2

        maxpool3_1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME')(block3_xcp)
        add3_1 = add([maxpool3_1, rconv2])
        rconv3 = Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2), padding='SAME', name="rconv3")(add3_1)
        rconv3 = BatchNormalization()(rconv3)
        # ######################################### create output3 data with side layer function
        output3 = self.branch_layers(block3_xcp, name='output3', filters=1, upscale=int(2 ** 2), strides=(1, 1),
                                     kernel_size=(1, 1), sub_pixel=True)

        conv_b2b4 = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='SAME', name="conv_b2b4")(maxpool2_1)
        block4_xcp = add3_1
        add_b2b3 = add([conv_b2b4, maxpool3_1])
        addb3_4b4 = Conv2D(filters=512, kernel_size=(1, 1), padding='SAME', strides=(1, 1), name="addb3_4b4")(add_b2b3)
        addb3_4b4 = BatchNormalization()(addb3_4b4)

        for k in range(3):
            block4_xcp = relu(block4_xcp)
            block4_xcp = Conv2D(filters=512, kernel_size=(3, 3),
                                strides=(1, 1), padding='SAME', name="conv1_block4_{}".format(k+1))(block4_xcp)
            block4_xcp = BatchNormalization()(block4_xcp)
            block4_xcp = relu(block4_xcp)

            block4_xcp = Conv2D(filters=512, kernel_size=(3, 3),
                                strides=(1, 1), padding='SAME', name="conv2_block4_{}".format(k + 1))(block4_xcp)
            block4_xcp = BatchNormalization()(block4_xcp)
            block4_xcp = add([block4_xcp, addb3_4b4])

        maxpool4_1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME')(block4_xcp)
        add4_1 = add([maxpool4_1, rconv3])

        rconv4 = Conv2D(filters=512, kernel_size=(1, 1), padding='SAME', strides=(1, 1), name="rconv4")(add4_1)
        rconv4 = BatchNormalization()(rconv4)
        # ######################################### create output4 data with side layer function
        output4 = self.branch_layers(block4_xcp, name='output4', filters=1, upscale=int(2 ** 3), strides=(1, 1),
                                     kernel_size=(1, 1), sub_pixel=True)

        convb3_2ab4 = Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2), padding='SAME', name="conv_b2b5")(conv_b2b4)
        block5_xcp = add4_1
        add_b2b5 = add([convb3_2ab4, maxpool4_1])
        add_b2b5 = Conv2D(filters=512, kernel_size=(1, 1), padding='SAME', strides=(1, 1), name="add_b2b5")(add_b2b5)
        add_b2b5 = BatchNormalization()(add_b2b5)

        for k in range(3):
            block5_xcp = relu(block5_xcp)
            block5_xcp = Conv2D(filters=512, kernel_size=(3, 3),
                                strides=(1, 1), padding='SAME', name="conv1_block5_{}".format(k+1))(block5_xcp)
            block5_xcp = BatchNormalization()(block5_xcp)
            block5_xcp = relu(block5_xcp)

            block5_xcp = Conv2D(filters=512, kernel_size=(3, 3),
                                strides=(1, 1), padding='SAME', name="conv2_block5_{}".format(k + 1))(block5_xcp)
            block5_xcp = BatchNormalization()(block5_xcp)
            block5_xcp = add([block5_xcp, add_b2b5])/2

        add5_1 = add([block5_xcp, rconv4])
        # ######################################### create output5 data with side layer function
        output5 = self.branch_layers(block5_xcp, name='output5', filters=1, upscale=int(2 ** 4), strides=(1, 1),
                                     kernel_size=(1, 1), sub_pixel=True)

        block6_xcp = Conv2D(filters=256, kernel_size=(1, 1), padding='SAME', strides=(1, 1), name="conv0_b6")(add5_1)
        block6_xcp = BatchNormalization()(block6_xcp)

        addb25_2b6 = Conv2D(filters=256, kernel_size=(1, 1), padding='SAME', strides=(1, 1), name="add2b6")(block6_xcp)
        addb25_2b6 = BatchNormalization()(addb25_2b6)

        for k in range(3):
            block6_xcp = relu(block6_xcp)
            block6_xcp = Conv2D(filters=256, kernel_size=(3, 3),
                                strides=(1, 1), padding='SAME', name="conv1_block6_{}".format(k+1))(block6_xcp)
            block6_xcp = BatchNormalization()(block6_xcp)
            block6_xcp = relu(block6_xcp)

            block6_xcp = Conv2D(filters=256, kernel_size=(3, 3),
                                strides=(1, 1), padding='SAME', name="conv2_block6_{}".format(k + 1))(block5_xcp)
            block6_xcp = BatchNormalization()(block6_xcp)
            block6_xcp = add([block6_xcp, addb25_2b6])/2
        # ######################################### create output6 data with side layer function
        output6 = self.branch_layers(block6_xcp, name='output6', filters=1, upscale=int(2 ** 4), strides=(1, 1),
                                     kernel_size=(1, 1), sub_pixel=True)

        branched_outputs = [output1, output2, output3, output4, output5, output6]
        fuse = Conv2D(filters=1, kernel_size=(1, 1), name='fuse_1', strides=(1, 1),
                      padding='SAME')(tf.concat(branched_outputs, axis=3))

        final_output = branched_outputs + [fuse]
        averaged_output = (output1 + output2 + output3 + output4 + output5 + output6 + fuse)/7.0

        model = keras.Model(inputs=input_image, outputs=averaged_output)

        return model

    def branch_layers(self, inputs, filters=None, kernel_size=None, strides=(1, 1), name=None,
                      upscale=None, sub_pixel=False):

        classifier = self.upsample_block(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                         name=name, upscale=upscale, sub_pixel=sub_pixel)

        return classifier

    def upsample_block(self, inputs, filters=None, kernel_size=None, strides=(1, 1),
                       name=None, upscale=None, sub_pixel=False):

        i = 1
        scale = 2
        sub_net = inputs
        output_filters = 16

        if sub_pixel:
            while scale <= upscale:
                if scale == upscale:

                    sub_net = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                     name=name + "_conv_{}".format(i))(sub_net)
                    sub_net = relu(sub_net)

                    sub_net = Conv2DTranspose(filters=filters, kernel_size=(upscale, upscale),
                                              padding='SAME', strides=(2, 2),
                                              name=name + "_deconv_{}_{}".format(upscale, i))(sub_net)

                else:
                    sub_net = Conv2D(filters=output_filters, kernel_size=kernel_size, strides=strides,
                                     name=name + "_conv_{}".format(i))(sub_net)
                    sub_net = relu(sub_net)

                    sub_net = Conv2DTranspose(filters=output_filters, kernel_size=(upscale, upscale),
                                              strides=(2, 2), padding='SAME',
                                              name=name + "_deconv_{}_{}".format(upscale, i))(sub_net)

                i += 1
                scale = 2**i

        else:
            while scale <= upscale:
                if scale == upscale:
                    current_shape = sub_net.get_shape().as_list()
                    sub_net = Conv2D(filters=filters, kernel_size=3, strides=strides,
                                     name=name + "_conv_{}".format(i))(sub_net)
                    sub_net = relu(sub_net)
                    if not(current_shape[0] == self.input_shape[0] and current_shape[1] == self.input_shape[1]):
                        sub_net = self.upscore_layer(input=sub_net, n_outputs=output_filters, stride=upscale,
                                                     k_size=upscale, name=name + '_bdconv_{}'.format(i))
                i += 1
                scale = 2**i

        return sub_net

    def upscore_layer(self, input, n_outputs, stride=2, k_size=4, name=None, shape=None):
        in_features = input.get_shape().as_list()[3]
        in_shape = tf.shape(input)
        ot_shape = input.get_shape().as_list()

        h = ((ot_shape[1] - 1) * stride) + 1
        w = ((ot_shape[2] - 1) * stride) + 1
        new_shape = [in_shape[0], self.input_shape[0], self.input_shape[1], n_outputs]
        output_shape = tf.stack(new_shape)
        f_shape = [k_size, k_size, n_outputs, in_features]

        num_of_features = (k_size*k_size*in_features)/stride
        std_dev = (2/num_of_features) ** 0.5

        weights = self.get_deconv_filter(f_shape, name=name + '_Wb')
        deconv = tf.nn.conv2d_transpose(input=input, filters=weights, output_shape=output_shape, strides=stride,
                                        padding='SAME', name=name)


        return deconv

    def get_deconv_filter(self, f_shape, name=''):
        width = f_shape[0]
        height = f_shape[1]

        f = np.ceil(width/2.0)
        c = (2 * f - 1 - f % 2)/(2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])

        for x in range(width):
            for y in range(height):
                value = (1 - abs(x/f - c)) * (1 - abs(y/f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights)

        return tf.compat.v1.get_variable(name=name, initializer=init, shape=weights.shape, dtype=tf.float32)

