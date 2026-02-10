from dataclasses import dataclass
import os
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_path = os.path.join("artifacts",'train.csv')
    test_path = os.path.join("artifacts","test.csv")
    raw_path = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        os.makedirs(os.path.dirname(self.config.raw_path),exist_ok=True)
    def inititate_data_ingestion(self):

        df = pd.read_csv("/home/ankit-gaur/ml_project/news_fake_detection/WELFake_Dataset.csv")

        df.to_csv(self.config.raw_path,header=True,index=False)
        train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)
        train_data.to_csv(self.config.train_path,header=True,index=False)
        test_data.to_csv(self.config.test_path,header=True,index=False)
        print("Data ingestion Completed")

        return (
            train_data,
            test_data
        )
