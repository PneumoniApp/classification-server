import os

from keras.applications.mobilenet_v2 import preprocess_input
from keras_preprocessing.image import ImageDataGenerator


class MainConfig:
    # TODO: Change data dir to the real dataset path
    DATA_DIR = 'Path to Pneumonia dataset from Kaggle'
    TRAIN_DIR = 'train'
    TEST_DIR = 'test'
    VAL_DIR = 'val'

    SEED = 6

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    COLOR_MODE = 'rgb'
    TRAIN_CLASS_MODE = 'categorical'

    # Transforms params
    ZOOM_RANGE = 0.2
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2


def get_train_generator():
    train_datagen = ImageDataGenerator(
        zoom_range=MainConfig.ZOOM_RANGE,
        horizontal_flip=True,
        width_shift_range=MainConfig.WIDTH_SHIFT_RANGE,
        height_shift_range=MainConfig.HEIGHT_SHIFT_RANGE,
        preprocessing_function=preprocess_input
    )
    return train_datagen.flow_from_directory(
        os.path.join(MainConfig.DATA_DIR, MainConfig.TRAIN_DIR),
        target_size=MainConfig.IMG_SIZE,
        color_mode=MainConfig.COLOR_MODE,
        batch_size=MainConfig.BATCH_SIZE,
        class_mode=MainConfig.TRAIN_CLASS_MODE,
        shuffle=True
    )


def get_validation_generator():
    validation_datagen = ImageDataGenerator(
        zoom_range=MainConfig.ZOOM_RANGE,
        horizontal_flip=True,
        width_shift_range=MainConfig.WIDTH_SHIFT_RANGE,
        height_shift_range=MainConfig.HEIGHT_SHIFT_RANGE,
        preprocessing_function=preprocess_input
    )
    return validation_datagen.flow_from_directory(
        os.path.join(MainConfig.DATA_DIR, MainConfig.VAL_DIR),
        target_size=MainConfig.IMG_SIZE,
        color_mode=MainConfig.COLOR_MODE,
        batch_size=16,
        class_mode=MainConfig.TRAIN_CLASS_MODE,
        shuffle=True
    )


def get_test_generator():
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return test_datagen.flow_from_directory(
        os.path.join(MainConfig.DATA_DIR, MainConfig.TEST_DIR),
        target_size=MainConfig.IMG_SIZE,
        color_mode=MainConfig.COLOR_MODE,
        batch_size=1,
        class_mode=MainConfig.TRAIN_CLASS_MODE,
        shuffle=False
    )
