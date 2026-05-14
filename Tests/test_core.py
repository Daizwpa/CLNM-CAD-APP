import pytest
import pandas as pd
import joblib
import os
from Utils.Core import load_sav_to_dataframe 



def test_load_sav_to_dataframe_invalid_extension(tmp_path):
    # Create a file with an invalid extension
    invalid_file = tmp_path / "invalid_data.txt"
    invalid_file.write_text("Invalid content")
    
    # Test loading the file with an invalid extension
    with pytest.raises(ValueError, match="The file must have a .sav extension."):
        load_sav_to_dataframe(str(invalid_file))

def test_load_sav_to_dataframe_file_not_found():
    # Test loading a non-existent file
    with pytest.raises(FileNotFoundError, match="The file at .* does not exist."):
        load_sav_to_dataframe("non_existent_file.sav")

def test_load_sav_to_dataframe_incompatible_content(tmp_path):
    # Create a .sav file with incompatible content
    incompatible_data = "This is not a list or dict"
    file_path = tmp_path / "incompatible_data.sav"
    joblib.dump(incompatible_data, file_path)
    
    # Test loading the incompatible .sav file
    with pytest.raises(ValueError, match="The content of the .sav file is not compatible with a pandas DataFrame."):
        load_sav_to_dataframe(str(file_path))
