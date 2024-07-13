import os
import sys

print(os.getcwd())
os.chdir("../../../")
print(os.getcwd())

# sys.path.insert(0, os.getcwd())

print(sys.path)


from src.data_expectations import validate_initial_data


validate_initial_data()