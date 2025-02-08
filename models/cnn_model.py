import tensorflow.keras.backend as K
from keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

# ✅ F1-score jako funkcja kompatybilna z Keras
def f1_score(y_true, y_pred):
    """Oblicza F1-score jako średnią harmoniczną precision i recall"""
    precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())
    recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def build_cnn_multilabel(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=32, kernel_size=5, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        MaxPooling1D(pool_size=2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='sigmoid')  # 🔥 Sigmoid dla multilabel classification
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            f1_score  # ⬅️ Używamy teraz funkcji zamiast obiektu tf.keras.metrics
        ]
    )
    return model




def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=32, kernel_size=5, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),  # Nowy Dropout
        MaxPooling1D(pool_size=2),

        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.2),  # Nowy Dropout
        MaxPooling1D(pool_size=2),

        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),  # Nowy Dropout
        MaxPooling1D(pool_size=2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),  # Większy dropout dla Dense
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

