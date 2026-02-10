from dataclasses import dataclass
import re
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


@dataclass
class DataTransformationConfig:
    tokenizer_path = os.path.join('artifacts','tokenizer.npy')
    transformed_train_path = os.path.join("artifacts","transformed_train.npz")
    transformed_test_path = os.path.join("artifacts","transformed_test.npz")
    max_words :int = 20000
    max_len :int = 300

class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()
        os.makedirs(os.path.dirname(self.config.tokenizer_path),exist_ok=True)

    def clean_text(self,text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    def initiate_data_transformation(self):
        print("Data Transformation Started")

        train_df = pd.read_csv("artifacts/train.csv")
        test_df = pd.read_csv("artifacts/test.csv")

        train_df.drop(columns=["Unnamed: 0"],inplace=True)
        test_df.drop(columns=["Unnamed: 0"],inplace=True)

        train_df = train_df.dropna()
        test_df = test_df.dropna()

        train_df["content"] = train_df['title'] + " " + train_df["text"]
        test_df["content"] = test_df["title"] + " " + test_df["text"]

        train_df['content'] = train_df['content'].apply(self.clean_text)
        test_df['content'] = test_df['content'].apply(self.clean_text)

        X_train = train_df['content'].values
        y_train = train_df['label'].values

        X_test = test_df['content'].values
        y_test = test_df['label'].values

        # Tokenization

        tokenizer = Tokenizer(num_words=self.config.max_words,oov_token = "<OOV>")
        tokenizer.fit_on_texts(X_train)

        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        X_train_pad = pad_sequences(
            X_train_seq,maxlen = self.config.max_len,padding="post"
        )
        X_test_pad = pad_sequences(
            X_test_seq,maxlen = self.config.max_len,padding="post"
        )

        np.save(self.config.tokenizer_path,tokenizer)

        np.savez(
            self.config.transformed_train_path,
            X = X_train_pad,
            y = y_train
        )
        np.savez(
            self.config.transformed_test_path,
            X = X_test_pad,
            y = y_test
        )
        print("Data Transformation Completed")


        