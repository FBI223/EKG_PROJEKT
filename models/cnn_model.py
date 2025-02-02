from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_cnn(input_shape, num_classes):
    """
    Tworzy model CNN do klasyfikacji EKG.

    :param input_shape: Kształt wejściowy (num_samples, 1)
    :param num_classes: Liczba klas (liczba oznaczeń w EKG)
    :return: Model CNN
    """
    model = Sequential([
        Conv1D(filters=32, kernel_size=5, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),  # Regularization
        Dense(num_classes, activation='softmax')  # Multi-class classification
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

