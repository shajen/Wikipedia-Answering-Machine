import calculators.neural_weight_calculator
import tensorflow as tf

class DeepAveragingNeuralWeightCalculator(calculators.neural_weight_calculator.NeuralWeightCalculator):
    def __init__(self, debug_top_items, model_file, workdir, questions_words, articles_title_words, articles_content_words, good_bad_ratio, train_data_percentage):
        calculators.neural_weight_calculator.NeuralWeightCalculator.__init__(self, debug_top_items, model_file, workdir, questions_words, articles_title_words, articles_content_words, good_bad_ratio, train_data_percentage)

    def _words_layers(self, filters, input):
        return tf.math.reduce_mean(input, 1)

    def _weight_layers(self, x):
        x = tf.keras.layers.Dense(calculators.neural_weight_calculator.NeuralWeightCalculator._W2V_SIZE, activation='sigmoid')(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='weight')(x)
        return x

    def _model_name(self):
        return 'dan_model'
