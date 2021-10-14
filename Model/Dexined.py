import keras.optimizers
from keras.layers import Conv2D, MaxPool2D
from keras.layers import BatchNormalization, add
from keras.activations import relu


class dexined_model():

    def __init__(self, input_shape):
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

        rconv1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='SAME', name="rconv1")(conv_1_2)

        block2_xcp = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', strides=(1, 1), name="conv_block2_1")(conv_1_2)
        block2_xcp = BatchNormalization()(block2_xcp)
        block2_xcp = relu(block2_xcp)

        block2_xcp = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', strides=(1, 1), name="conv2_block2_1")(block2_xcp)
        block2_xcp = BatchNormalization()(block2_xcp)

        maxpool2_1 = MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME')(block2_xcp)
        add2_1 = add([maxpool2_1, rconv1])

        # ######################################### create output2 data with side layer function

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

        model = keras.Model(inputs=input_image, outputs=block6_xcp)

        return model
