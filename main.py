from pipelines.model_training_pipeline import ModelTrainingPipeline
from pipelines.data_preparation_pipeline import DataPreparationPipeline
from pipelines.evaluation_pipeline import EvaluationPipeline
from pipelines.explainability_pipeline import ExplainabilityPipeline
from fake_news_classifier import FakeNewsClassifier

if __name__ == "__main__":
    data_preparation_pipeline = DataPreparationPipeline(
        "configs/pipelines_config/data_preparation_config.json"
    )
    train_data, test_data, val_data = data_preparation_pipeline.run()

    training_pipeline = ModelTrainingPipeline(
        "configs/pipelines_config/model_training_config.json", train_data, val_data
    )
    model = training_pipeline.run()
    # fake_news_classifier = FakeNewsClassifier("configs/classifier_config.json", 7)
    # model = fake_news_classifier.load_pretrained("models/best_model.pth")

    evaluation_pipeline = EvaluationPipeline(
        "configs/pipelines_config/evaluation_config.json", test_data, model
    )
    evaluation_pipeline.run()

    # explainability_pipeline = ExplainabilityPipeline(
    #     "configs/pipelines_config/explainability_config.json",
    #     model.classifier,
    #     val_data,
    # )
    # explainability_pipeline.run_shap()
