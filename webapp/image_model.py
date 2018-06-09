import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, load_model, save_model, Sequential
from keras.datasets import cifar10
import numpy as np
import os

from PIL import Image
from keras.utils import to_categorical


# Training parameters
# batch_size = 32  # orig paper trained all networks with batch_size=128
# epochs = 200
data_augmentation = True


# num_classes = 4
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
# n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
# version = 1


class ImageModel:

    def __init__(self, batch_size=12, epochs=32,
                 image_size=48, n=4, version=1):

        # hyper parameter
        self.batch_size = batch_size
        self.epochs = epochs
        self.image_size = image_size
        self.n = n
        self.version = version
        # Computed depth from supplied model parameter n
        if version == 1:
            self.depth = n * 6 + 2
        elif version == 2:
            self.depth = n * 9 + 2

        # Model name, depth and version
        self.model_type = 'ResNet%dv%d_b%d_e%d_size_%d' % (
            self.depth, self.version, self.batch_size,
            self.epochs, self.image_size)

        # todo delete
        self.kernel_size = 3

    def load_image_to_array(self, input_dir):

        x = []
        y = []
        categories = []

        # ./data/train または ./data/test 以下のカテゴリの取得
        for dir_name in os.listdir(input_dir):
            if dir_name == ".DS_Store":
                continue
            categories.append(dir_name)

        for idx, category in enumerate(categories):
            category_dir = input_dir + "/" + category
            print("---category dir:", category_dir)

            for file in os.listdir(category_dir):
                if file != ".DS_Store" and file != ".keep":
                    filepath = category_dir + "/" + file
                    image = self.preprocess_image(filepath)
                    # 出来上がった配列をimage_listに追加。
                    x.append(image)
                    # 配列label_listに正解ラベルを追加(0,1,2...)
                    y.append(idx)

        # kerasに渡すためにnumpy配列に変換。
        x = np.array(x)

        # ラベルの配列をone hotラベル配列に変更
        # 0 -> [1,0,0,0], 1 -> [0,1,0,0] という感じ。
        y = to_categorical(y)

        return (x, y, categories)

    def prepare_train_data(self):

        # 学習用のデータを作る.
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.categories = []

        (self.x_train, self.y_train,
         self.categories) = self.load_image_to_array("data/train")
        (self.x_test, self.y_test,
         self.categories) = self.load_image_to_array("data/test")

        self.num_classes = self.y_train.shape[1]

        # If subtract pixel mean is enabled
        if subtract_pixel_mean:
            self.x_train_mean = np.mean(self.x_train, axis=0)
            self.x_train -= self.x_train_mean
            self.x_test -= self.x_train_mean

        print('x_train shape:', self.x_train.shape)
        print('y_train shape:', self.y_train.shape)

    def preprocess_image(self, filepath):
        # 画像を48 x 48(pixel設定可) x 1(grey)のnp_arrayに変換
        # そして /255.で 正規化する

        # print('preprocess_image', filepath)
        image = Image.open(filepath).convert("L")
        image = np.array(image.resize((self.image_size, self.image_size)))
        image = image.reshape(self.image_size, self.image_size, 1)
        image = image / 255.
        # print('after preprocess image shape:',image.shape)

        return image

    def train(self):

        # init params
        depth = self.depth
        batch_size = self.batch_size
        epochs = self.epochs
        version = self.version
        model_type = self.model_type
        num_classes = self.num_classes

        # train & test data
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test

        input_shape = x_train.shape[1:]

        # begin build the ResNet model

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)

        if version == 2:
            model = resnet_v2(input_shape=input_shape,
                              depth=depth, num_classes=num_classes)
        else:
            model = resnet_v1(input_shape=input_shape,
                              depth=depth, num_classes=num_classes)

        print('compile')
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.lr_schedule(0)),
                      metrics=['accuracy'])
        model.summary()
        print(model_type)

        # Prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'keras_flower_%s_model.{epoch:03d}.h5' % model_type
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        print('fit')
        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True)

        lr_scheduler = LearningRateScheduler(self.lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)

        callbacks = [checkpoint, lr_reducer, lr_scheduler]

        # Run training, with or without data augmentation.
        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      shuffle=True,
                      callbacks=callbacks)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False)

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                validation_data=(x_test, y_test),
                epochs=epochs, verbose=1, workers=4,
                callbacks=callbacks)
            model.save("image_model.h5")
            self.model = model

    def predict(self, filepath):
        result = self.evaluate_single(filepath)
        print("predict result:", result)
        idx = np.argmax(result[0])
        return self.categories[idx]

    def evaluate_single(self, filepath):
        image = self.preprocess_image(filepath)

        image_list = []
        image_list.append(image)
        image = np.array(image_list)

        if subtract_pixel_mean:
            image -= self.x_train_mean
        result = self.model.predict(image)

        return result

    def evaluate(self):
        # modelのテスト

        x_test = self.x_test
        y_test = self.y_test

        score = self.model.evaluate(x_test, y_test)
        print('loss=', score[0])
        print('accuracy=', score[1])

    def load(self, filepath):
        self.model = load_model(filepath)

    def lr_schedule(self, epoch):
        """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs
        Called automatically every epoch as part of callbacks during training.
        # Arguments
            epoch (int): The number of epochs
        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > self.epochs * 0.95:
            lr *= 0.5e-3
        elif epoch > self.epochs * 0.8:
            lr *= 1e-3
        elif epoch > self.epochs * 0.6:
            lr *= 1e-2
        elif epoch > self.epochs * 0.4:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr


# static methods

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved(downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved(downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
