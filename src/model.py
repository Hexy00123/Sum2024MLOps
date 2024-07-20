import importlib
import mlflow.sklearn
import mlflow
import pandas as pd
from zenml.client import Client
from sklearn.model_selection import GridSearchCV
import warnings
import giskard

warnings.filterwarnings('ignore')


def train(X_train, y_train, cfg):
    import importlib
    from sklearn.model_selection import KFold

    # Define the model hyperparameters
    model_params = cfg.model.params

    # Load "module_name.class_name"
    module_name = cfg.model.module_name
    class_name = cfg.model.class_name
    class_instance = getattr(importlib.import_module(module_name), class_name)

    estimator = class_instance(**model_params)

    # Grid search with cross validation
    cv = KFold(n_splits=cfg.model.folds,
               random_state=cfg.random_state,
               shuffle=True)

    param_grid = dict(model_params)

    # ['r2', 'mae', 'rmse']
    scoring = list(cfg.model.metrics.values())

    evaluation_metric = cfg.model.evaluation_metric

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=cfg.cv_n_jobs,
        refit=evaluation_metric,
        cv=cv,
        verbose=1,
        return_train_score=True
    )

    gs.fit(X_train, y_train)

    return gs


def retrieve_model_with_alias(model_name, model_alias="champion") -> mlflow.pyfunc.PyFuncModel:

    best_model: mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@{model_alias}")

    return best_model


def retrieve_model_with_version(model_name, model_version="v1") -> mlflow.pyfunc.PyFuncModel:

    best_model: mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}")

    return best_model


def log_metadata(cfg, gs, X_train, y_train, X_test, y_test):
    mlflow.set_tracking_uri(uri="http://localhost:5000")

    cv_results = pd.DataFrame(gs.cv_results_).filter(
        regex=r'std_|mean_|param_').sort_index(axis=1)
    best_metrics_values = [result[1][gs.best_index_]
                           for result in gs.cv_results_.items()]
    best_metrics_keys = [metric for metric in gs.cv_results_]
    best_metrics_dict = {k: v for k, v in zip(
        best_metrics_keys, best_metrics_values) if 'mean' in k or 'std' in k}

    params = best_metrics_dict

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    experiment_name = cfg.model.model_name + "_" + cfg.experiment_name

    # Retrieve an MLflow Experiment
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(
            name=experiment_name).experiment_id

    print("experiment-id : ", experiment_id)

    cv_evaluation_metric = cfg.model.cv_evaluation_metric
    run_name = "_".join([cfg.run_name, cfg.model.model_name, cfg.model.evaluation_metric, str(
        params[cv_evaluation_metric]).replace(".", "_")])  # type: ignore
    print("run name: ", run_name)

    if (mlflow.active_run()):
        mlflow.end_run()

    # Fake run
    with mlflow.start_run():
        pass

    # Parent run
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        df_train_dataset = mlflow.data.pandas_dataset.from_pandas(df=df_train,
                                                                  targets=cfg.data.target_cols[0])

        df_test_dataset = mlflow.data.pandas_dataset.from_pandas(df=df_test,
                                                                 targets=cfg.data.target_cols[0])
        mlflow.log_input(df_train_dataset, "training")
        mlflow.log_input(df_test_dataset, "testing")
        print("LOG: logged input")

        # Log the hyperparameters
        mlflow.log_params(gs.best_params_)
        print("LOG: logged model params")

        # Log the performance metrics
        mlflow.log_metrics(best_metrics_dict)
        print("LOG: logged model metrics")

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag(cfg.model.tag_key, cfg.model.tag_value)
        print("LOG: tagged model")

        # Infer the model signature
        signature = mlflow.models.infer_signature(X_train, gs.predict(X_train))
        print("LOG: set signature")

        model_info = mlflow.sklearn.log_model(
            sk_model=gs.best_estimator_,
            artifact_path=cfg.model.artifact_path,
            signature=signature,
            input_example=X_train.iloc[0].to_numpy(),
            registered_model_name=cfg.model.model_name,
            pyfunc_predict_fn=cfg.model.pyfunc_predict_fn
        )
        print("LOG: logged model")

        client = mlflow.client.MlflowClient()
        client.set_model_version_tag(
            name=cfg.model.model_name, version=model_info.registered_model_version, key="source", value="best_Grid_search_model")

        cv_results_list = list(cv_results.iterrows())
        print(f'LOG: CV results size: {len(cv_results_list)}')
        for index, result in cv_results_list:
            child_run_name = "_".join(['child', run_name, str(index)])
            print(f"LOG: performing child run: {child_run_name}")

            with mlflow.start_run(run_name=child_run_name, experiment_id=experiment_id, nested=True):
                ps = result.filter(regex='param_').to_dict()
                ms = result.filter(regex='mean_').to_dict()
                stds = result.filter(regex='std_').to_dict()

                # Remove param_ from the beginning of the keys
                ps = {k.replace("param_", ""): v for (k, v) in ps.items()}
                mlflow.log_params(ps)
                mlflow.log_metrics(ms)
                mlflow.log_metrics(stds)
                print('LOG: logged child model params & metrics')

                # We will create the estimator at runtime
                module_name = cfg.model.module_name  # e.g. "sklearn.linear_model"
                class_name = cfg.model.class_name  # e.g. "LogisticRegression"

                # Load "module.submodule.MyClass"
                class_instance = getattr(
                    importlib.import_module(module_name), class_name)

                print('LOG: created instance')
                print('LOG: fitting...')
                estimator = class_instance(**ps)
                estimator.fit(X_train, y_train)
                print('LOG: fitting done')

                signature = mlflow.models.infer_signature(
                    X_train, estimator.predict(X_train))
                model_info = mlflow.sklearn.log_model(
                    sk_model=estimator,
                    artifact_path=cfg.model.artifact_path,
                    signature=signature,
                    input_example=X_train.iloc[0].to_numpy(),
                    registered_model_name=cfg.model.model_name,
                )
                print('LOG: logged child model')

                model_uri = model_info.model_uri
                loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)
                print('LOG: loading child model')

                predictions = loaded_model.predict(X_test)  # type: ignore
                print('LOG: re-evaluating child model')

                eval_data = pd.DataFrame({
                    'label': y_test,
                    'predictions': predictions,
                })
                print('LOG: metrics preprocessing done')

                results = mlflow.evaluate(
                    model_info.model_uri,
                    data=X_test,
                    targets=y_test.values,
                    model_type='regressor',
                    evaluators=['default']
                )
                print('LOG: evaluating')

                # print(f"LOG: metrics:\n{results.metrics}")
