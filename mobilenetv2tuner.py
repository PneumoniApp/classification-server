from keras import Model
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import Adam

import keras_tuner as kt


class MobileNetV2Tuner:
    def __init__(
            self,
            max_trials,
            executions_per_trial=1,
            overwrite_results=False,
            tuner='bayesian',
            img_size=224,
            activation=None
    ):
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.overwrite_results = overwrite_results
        self._tuner = tuner.lower()
        self.img_size = img_size
        if activation:
            self.activation = [activation] if type(activation) == str else activation
        else:
            self.activation = ['relu', 'tanh', 'sigmoid']

    def __model_builder(self, hp):
        base_model = MobileNetV2(input_shape=(self.img_size, self.img_size, 3), weights='imagenet', include_top=False)

        hp_activation = hp.Choice('dense_activation', values=self.activation)
        hp_first_layer_units = hp.Int('first_layer_units', min_value=256, max_value=1024, step=32)
        hp_second_layer_units = hp.Int('second_layer_units', min_value=128, max_value=512, step=32)
        hp_third_layer_units = hp.Int('third_layer_units', min_value=32, max_value=256, step=32)
        hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-3)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=hp_first_layer_units, activation=hp_activation)(x)
        x = Dense(units=hp_second_layer_units, activation=hp_activation)(x)
        x = Dense(units=hp_third_layer_units, activation=hp_activation)(x)
        preds = Dense(2, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=preds)

        for layer in model.layers[:10]:
            layer.trainable = False
        for layer in model.layers[10:]:
            layer.trainable = True

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=hp_learning_rate),
            metrics=['accuracy']
        )

        return model

    @property
    def tuner(self):
        if self._tuner == 'bayesian':
            tuner = kt.BayesianOptimization(
                self.__model_builder,
                objective='val_accuracy',
                max_trials=self.max_trials,
                executions_per_trial=self.executions_per_trial,
                overwrite=self.overwrite_results,
                directory='Results',
                project_name=self._tuner.capitalize()
            )
        elif self._tuner == 'hyperband':
            tuner = kt.Hyperband(
                self.__model_builder,
                objective='val_accuracy',
                max_epochs=self.max_trials,
                executions_per_trial=self.executions_per_trial,
                overwrite=self.overwrite_results,
                directory='Results',
                project_name=self._tuner.capitalize()
            )
        else:
            tuner = kt.RandomSearch(
                self.__model_builder,
                objective='val_accuracy',
                max_trials=self.max_trials,
                executions_per_trial=self.executions_per_trial,
                overwrite=self.overwrite_results,
                directory='Results',
                project_name=self._tuner.capitalize()
            )
        return tuner
