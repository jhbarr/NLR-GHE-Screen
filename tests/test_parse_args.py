import pytest

from scripts.parse_args import parse_arguments


class TestCommandLineArguments:
    """
    Test cases for the handling of command line arguments in the program
    and user inputted bounding boxes.
    """

    def test_valid_arguments(self):
        test_args = ['47.4810', '-122.4597', '47.7341', '-122.2244']

        bbox = parse_arguments(test_args)

        assert bbox[2] == 47.7341

    def test_invalid_arguments(self):
        test_args = ['47.4810', 'e', '47.7341', '-122.2244']

        with pytest.raises(SystemExit):
            parse_arguments(test_args)

    def test_missing_required_arguments(self):
        test_args = ['47.4810', '-122.4597', '47.7341']

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
            ['47.4810', '-122.4597', '92.7341', '-122.2244'],

            # Longitude coordinates exceed possible range
            ['47.4810', '-122.4597', '47.7341', '-182.2244'],

            # Latitude coordinates are swapped
            ['47.7341', '-122.4597', '47.4810', '-122.2244'],

            # Longitude coordinates are swapped
            ['47.4810', '-122.2244', '47.7341', '-122.4597'],
        ],
    )
    def test_invalid_bounding_box_crs(self, test_args):
        with pytest.raises(ValueError):
            parse_arguments(test_args)
