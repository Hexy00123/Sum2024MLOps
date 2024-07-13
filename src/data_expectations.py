# python src/data_expectations.py

# TODO: validate features

import great_expectations as gx
import pandas as pd


def validate_initial_data() -> None:
    context = gx.get_context(project_root_dir="services")

    retrieved_checkpoint = context.get_checkpoint(
        name="first_phase_checkpoint")

    results = retrieved_checkpoint.run()

    assert results.success

    print('All data satisfies the validity conditions!', results.success)


def validate_features(X: pd.DataFrame, y: pd.Series) -> None:
    context = gx.get_context(project_root_dir="services")

    y.rename('price', inplace=True)

    df = pd.concat([X, y], axis=1)

    ds = context.sources.add_or_update_pandas(name="ds")
    da = ds.add_dataframe_asset(name="df", dataframe=df)
    batch_request = da.build_batch_request()
    checkpoint = context.add_or_update_checkpoint(
        name="features_checkpoint",
        validations=[{
            "batch_request": batch_request,
            "expectation_suite_name": "features_expectations_suite"
        }]
    )

    results = checkpoint.run()

    assert results.success

    print('All features satisfy the validity conditions!', results.success)


def docs() -> None:
    """
    This function is used to generate the documentation site for the data validation.
    """
    context = gx.get_context(project_root_dir="services")

    context.build_data_docs()

    # Open the data docs in a browser
    context.open_data_docs()


if __name__ == "__main__":
    # validate_initial_data()
    docs()
