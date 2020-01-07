import calculators.neural_weight_calculator
import tensorflow as tf

class DeepAveragingNeuralWeightCalculator(calculators.neural_weight_calculator.NeuralWeightCalculator):
    def __init__(self, data_loader, debug_top_items, workdir, good_bad_ratio):
        calculators.neural_weight_calculator.NeuralWeightCalculator.__init__(self, data_loader, debug_top_items, workdir, good_bad_ratio)

    def _words_layers(self, filters, input):
        return tf.math.reduce_mean(input, 1)

    def _weight_layers(self, x):
        x = tf.keras.layers.Dense(self._word2vec_size, activation='sigmoid')(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='weight')(x)
        return x

    def model_name(self):
        return 'dan_model'
