from mobilenetv2tuner import MobileNetV2Tuner
from utils import get_train_generator, get_validation_generator


def main():
    mobilenetv2_tuner = MobileNetV2Tuner(50)
    tuner = mobilenetv2_tuner.tuner

    train_generator = get_train_generator()
    validation_generator = get_validation_generator()

    x_val, y_val = validation_generator.next()

    tuner.search(x=train_generator, epochs=50, validation_data=(x_val, y_val))
    print(" --- Hyperparams search completed!")


if __name__ == '__main__':
    main()
