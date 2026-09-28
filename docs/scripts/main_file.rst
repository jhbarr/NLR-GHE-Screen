Main Execution (:py:mod:`scripts.main`)
=======================================

This page details the main execution file of the GHE Screen program

Functions
---------

.. automodule:: scripts.main
   :members:
   :undoc-members:
   :show-inheritance:


Dependencies
------------
**Python**: 

- json
- sys
- pathlib.Path
- jsonschema.ValidationError
- pyorgio.errors.DataSourceError

**scripts**

- parse_args.parse_arguments
- osm_api.get_overpass
- parse_geojson.parse_api_response
- validate_geojson.validate_geojson
- from geometry_manipulation: combine_geometries, classify_geometry
- ms_cross_validation.cross_validate_osm_spaces

