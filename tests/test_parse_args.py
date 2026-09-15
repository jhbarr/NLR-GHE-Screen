import pytest
from pyogrio.errors import DataSourceError
from scripts.parse_args import parse_arguments
from pathlib import Path

class TestBBOXCommandLineArgs:
    """
    Test cases for the handling of command line arguments in the program
    and user inputted bounding boxes
    """

    def test_valid_arguments(self):
        test_args = ['--bbox', '47.4810', '-122.4597', '47.7341', '-122.2244']

        bbox = parse_arguments(test_args)

        assert bbox[2] == 47.7341

    def test_invalid_arguments(self):
        test_args = ['--bbox', '47.4810', 'e', '47.7341', '-122.2244']

        with pytest.raises(SystemExit):
            parse_arguments(test_args)

    def test_missing_required_arguments(self):
        test_args = ['--bbox', '47.4810', '-122.4597', '47.7341']

        with pytest.raises(SystemExit) as exc_info:
            parse_arguments(test_args)

        assert exc_info.value.code == 2

    def test_missing_required_arguments(self):
        test_args = []

        with pytest.raises(SystemExit) as exc_info:
            parse_arguments(test_args)

        assert exc_info.value.code == 2

    @pytest.mark.parametrize(
        "test_args",
        [
            # Latitude coordinates exceed possible range
            ['--bbox', '47.4810', '-122.4597', '92.7341', '-122.2244'],

            # Longitude coordinates exceed possible range
            ['--bbox', '47.4810', '-122.4597', '47.7341', '-182.2244'],

            # Latitude coordinates are swapped
            ['--bbox', '47.7341', '-122.4597', '47.4810', '-122.2244'],

            # Longitude coordinates are swapped
            ['--bbox', '47.4810', '-122.2244', '47.7341', '-122.4597'],
        ],
    )
    def test_invalid_bounding_box_crs(self, test_args):
        with pytest.raises(ValueError):
            parse_arguments(test_args)

class TestFileCommandLineArgs:
    """
    Tests cases for handling URBANopt GeoJSON input to the program 
    """

    def test_valid_input_file(self):
        """
        Verify that the program parses the file and retrieves the bounding box correctly
        """
        file_path = Path(__file__).parent / "Data" / "valid_geojson.json"
        test_args = ['--file', str(file_path)]

        bbox = parse_arguments(test_args)

        assert bbox != None

    def test_invalid_input_file(self):
        """
        Verify the program handles the case where an invalid GeoJSON is given
        """
        file_path = Path(__file__).parent / "Data" / "invalid_geojson.json"
        test_args = ['--file', str(file_path)]
 
        with pytest.raises(DataSourceError):
            bbox = parse_arguments(test_args)

    def test_invalid_input_file_path(self):
        """
        Verify that the program handles the case in which an invalid file path is given
        """
        file_path = Path(__file__).parent / "valid_geojson.json"
        test_args = ['--file', str(file_path)]
        
        with pytest.raises(FileNotFoundError):
            bbox = parse_arguments(test_args)