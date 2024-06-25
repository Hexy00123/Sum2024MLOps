# python tests/data_expectations.py

import great_expectations as gx


def validate_initial_data():
    """
    This function is used to validate the initial data using the data expectations.
    """
    context = gx.get_context(project_root_dir="services")

    retrieved_checkpoint = context.get_checkpoint(name="first_phase_checkpoint")

    results = retrieved_checkpoint.run()

    assert results.success


def docs():
    """
    This function is used to generate the documentation site for the data validation.
    """
    context = gx.get_context(project_root_dir="services")

    context.build_data_docs()

    # Open the data docs in a browser
    context.open_data_docs()


if __name__ == "__main__":
    validate_initial_data()
    # docs()
