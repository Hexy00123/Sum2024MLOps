import hydra
from omegaconf import DictConfig
from model import retrieve_model_with_alias
from data import read_features
from sklearn.metrics import mean_absolute_percentage_error
# import mlflow


@hydra.main(config_path='../configs', config_name='evaluate_model', version_base=None)
def evaluate(cfg: DictConfig):
    X_test, y_test = read_features('features_target', str(cfg.data_version))

    model = retrieve_model_with_alias(
        model_name='XGBoost', model_alias='champion')
    y_pred = model.predict(X_test)

    metric = mean_absolute_percentage_error(y_test, y_pred)
    print(f'mean_absolute_percentage_error: {metric}')


if __name__ == '__main__':
    import mlflow
    mlflow.artifacts.download_artifacts(run_id='b55f7fd4abc64aa59b05171fdd349127', 
                                        
                                        dst_path='./hello')
    # evaluate()
