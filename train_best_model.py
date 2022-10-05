from keras.callbacks import ModelCheckpoint

from mobilenetv2tuner import MobileNetV2Tuner
from utils import get_train_generator, get_validation_generator


def main():
    # Debemos crear el tuner exactamente igual al que usamos para el script de búsqueda de hiperparámetros
    mobilenetv2_tuner = MobileNetV2Tuner(50)
    tuner = mobilenetv2_tuner.tuner
    # tuner.results_summary()  # Descomentar para verificar que sí cargó los resultados obtenidos previamente

    checkpoint_filepath = 'tmp/checkpoint'  # Puede ser una ruta diferente

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    # model.summary()

    train_generator = get_train_generator()
    validation_generator = get_validation_generator()

    x_val, y_val = validation_generator.next()

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    model.fit(
        x=train_generator,
        validation_data=(x_val, y_val),
        epochs=500,
        callbacks=[model_checkpoint_callback]
    )


if __name__ == '__main__':
    main()
