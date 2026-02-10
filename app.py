from src.fake_news_detection.components.data_ingestion import DataIngestion
from src.fake_news_detection.components.data_transformation import DataTransformation
from src.fake_news_detection.components.model_tranier import ModelTrainer

if __name__ == "__main__":

    # data_ingestion = DataIngestion()
    # train_data,test_data = data_ingestion.inititate_data_ingestion()
    # print("Sucessfully completed")
    # data_transformation = DataTransformation()
    # data_transformation.initiate_data_transformation()
    model_trainer = ModelTrainer()
    model_trainer.train()


    print("Data Trainer Successfully")