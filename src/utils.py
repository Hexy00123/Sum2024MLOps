import yaml
import argparse


def update_yaml_index(file_path, new_index):
    """
    Update the 'index' parameter in the specified YAML configuration file.

    Parameters:
    - file_path: str - The path to the YAML file.
    - new_index: int - The new value for the 'index' parameter.
    """
    with open(file_path, 'r') as file:
        # Load the YAML content
        config = yaml.safe_load(file)

    # Update the index value
    if 'index' in config:
        print(f"Current index value: {config['index']}")
        config['index'] = new_index
        print(f"Updated index value to: {config['index']}")
    else:
        raise KeyError("'index' parameter not found in the YAML file.")

    # Write the updated config back to the file
    with open(file_path, 'w') as file:
        yaml.dump(config, file)

if __name__ == "__main__":
    # Path to the main.yaml configuration file
    yaml_file_path = "configs/main.yaml"

    # New index value
    # new_index_value = 1

    # parse an index from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int, help="The new index value")
    args = parser.parse_args()
    new_index_value = args.index

    # example usage: python src/utils.py 1

    # Update the index parameter in the YAML file
    update_yaml_index(yaml_file_path, new_index_value)
    print("YAML file updated successfully.")
