from great_expectations.data_context import FileDataContext


def validate_initial_data():
    context = FileDataContext(context_root_dir="../services/gx")

    data_source = context.sources.add_or_update_pandas(name="first_ds")
    data_asset = data_source.add_csv_asset(
        name="asset01",
        filepath_or_buffer="../data/samples/sample.csv",
    )

    batch_request = data_asset.build_batch_request()

    context.add_or_update_expectation_suite("first_expectation_suite")

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="first_expectation_suite",
    )

    # Expectations
    validator.expect_table_row_count_to_be_between(min_value=33689, max_value=168446)

    validator.expect_column_values_to_not_be_null(
        column="property_id"
    )

    validator.expect_column_values_to_not_be_null(
        column="location_id"
    )

    validator.save_expectation_suite(
        discard_failed_expectations=False
    )

    checkpoint = context.add_or_update_checkpoint(
        name="first_checkpoint",
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "first_expectation_suite",
            },
        ],
    )

    results = checkpoint.run()

    if not results.success:
        raise Exception("Data validation failed. Please check the logs for more details.")
    print("Data validation successful.")


if __name__ == "__main__":
    validate_initial_data()
