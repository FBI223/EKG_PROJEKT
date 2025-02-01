import tensorflow as tf
from tensorflow.keras import layers

def build_cnn():
    """Tworzy model CNN do analizy EKG"""
    model = tf.keras.Sequential([
        layers.Conv1D(64, 5, activation='relu', input_shape=(256, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, 5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 klasy (normalne, PVC, PAC)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
