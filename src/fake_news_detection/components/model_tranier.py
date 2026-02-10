from dataclasses import dataclass
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,Bidirectional,Embedding
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts","model.h5")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.model_file_path),exist_ok=True)

    def build_model(self):
        model = Sequential()
        model.add(Embedding(
            input_dim = 20000,
            output_dim = 128,
            input_length = 300
        ))
        model.add(Bidirectional(
            LSTM(
                64,return_sequences = False
            )
        ))
        model.add(Dropout(0.5))
        model.add(Dense(64,activation="relu"))
        model.add(Dense(1,activation="sigmoid"))

        model.compile(
            optimizer="adam",
            loss = "binary_crossentropy",
            metrics = ['accuracy']
        )
        return model
    def train(self):
        print("🔹 Loading transformed data")
        train_data = np.load("artifacts/transformed_train.npz")
        test_data = np.load("artifacts/transformed_test.npz")

        X_train,y_train = train_data["X"],train_data['y']
        X_test,y_test = test_data["X"],test_data['y']


        print(f"Train data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        model = self.build_model()
        model.summary()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
            ModelCheckpoint(
                self.config.model_file_path,
                monitor="val_loss",
                save_best_only=True
            )
        ]
        print("🚀 Training model...")
        model.fit(
            X_train,
            y_train,
            validation_data = (X_test,y_test),
            epochs = 10,
            batch_size=64,
            callbacks = callbacks

        )
        print("✅ Model training completed")