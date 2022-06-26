import tensorflow as tf
import numpy as np


def prepareModel(modelFilename=None, shape=(160, 160)):

    if modelFilename is not None:
        return tf.keras.models.load_model(modelFilename)

    '''
        для демонстрации достаточно будет безголовой претрейненой модели
        вектор конечно же великоват
    '''
    base = tf.keras.applications.MobileNetV3Large(include_top=True, minimalistic=False,
                                                  weights='imagenet', alpha=1.0, input_shape=(shape[0], shape[1], 3))

    out = base.get_layer('multiply_20').output
    out = tf.keras.layers.Flatten()(out)

    model = tf.keras.Model(inputs=base.input, outputs=out)

    return model


def calcVectors(model: tf.keras.Model, generator) -> np.array:

    vectors = []
    batch = []
    batchSize = 32
    for image in generator:
        batch.append(image)
        if batchSize == len(batch):
            result = model.predict_on_batch(np.array(batch))
            vectors.extend(result)
            batch = []

    if len(batch) > 0:
        result = model.predict_on_batch(np.array(batch))
        vectors.extend(result)

    return vectors


def normalizeVectors(vectors: np.array) -> np.array:

    vectors = np.array(vectors)
    # vectors /= np.max(np.abs(vectors), axis=0)
    vectors /= np.linalg.norm(vectors, axis=0)

    return vectors
