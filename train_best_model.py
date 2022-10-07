import pandas as pd
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

from mobilenetv2tuner import MobileNetV2Tuner
from utils import get_train_generator, get_validation_generator


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.title('Loss vs Iterations')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')
    plt.legend()
    plt.savefig('Results/best_model/plot.svg')
    plt.show()


def main():
    # Debemos crear el tuner exactamente igual al que usamos para el script de búsqueda de hiperparámetros
    mobilenetv2_tuner = MobileNetV2Tuner(50)
    tuner = mobilenetv2_tuner.tuner
    # tuner.results_summary()  # Descomentar para verificar que sí cargó los resultados obtenidos previamente

    checkpoint_filepath = 'Results/best_model/checkpoint_{epoch:02d}'  # Puede ser una ruta diferente

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

    history = model.fit(
        x=train_generator,
        validation_data=(x_val, y_val),
        epochs=500,
        callbacks=[model_checkpoint_callback]
    )

    plot_history(history)
    df = pd.DataFrame(history.history)
    df.to_csv('Results/best_model/metrics.csv')


if __name__ == '__main__':
    main()
