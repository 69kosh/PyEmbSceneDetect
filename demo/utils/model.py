import tensorflow as tf
import numpy as np


def prepareModel(modelFilename=None, shape=(160, 160)):

    if modelFilename is not None:
        return tf.keras.models.load_model(modelFilename)

    '''
        для демонстрации достаточно будет безголовой претрейненой модели
        вектор конечно же великоват
    '''
    base = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet', 
                                    alpha=1.0, input_shape=(shape[0], shape[1], 3))

    # base.summary()

    out = base.get_layer('global_average_pooling2d').output

    model = tf.keras.Model(inputs=base.input, outputs=out)

    model.summary()

    return model


def calcVectors(model: tf.keras.Model, generator) -> np.array:

    vectors = []
    batch = []
    batchSize = 64
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
    