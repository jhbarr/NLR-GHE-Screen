Program Execution
=================
The program is executed through the command line interface. There are two main input methods for the user to describe their desired
bounding box region where they would like to extract information regarding possible GHE installation sites. 

1. Coordinate Inputs
--------------------
The first method to describe the desired bounding box is through direct coordinate inputs. To do so, the user needs to utilize the
--bbox flag and then input four coordinates describing the south, west, north, east limits of the bounding box (in that order).
Ex - A command to query the city of Chicago, IL
.. code-block:: bash
    uv run -m nlr_ghe_screen.cli.main --bbox 41.644 -87.940 42.023 -87.524

2. URBANopt GeoJSON Output File
-------------------------------
The second method to describe the desired bounding box is by providing a relative path to an URBANopt GeoJSON file. 
The file must contain a feature of type 'bounding box' that corresponds to a complete polygon geometry outlining the 
desired area. The user must include the --file flag along with the file path.
Ex - A command to query using a file
.. code-block:: bash
    uv run nlr_ghe_screen.cli.main --file path/to/file/example_file.json
*Note* - As of writing this documentation, the ability to draw this bounding box has not yet been implemented in URBANopt
