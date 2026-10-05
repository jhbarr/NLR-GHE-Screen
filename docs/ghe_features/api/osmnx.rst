OSMNX API (:py:mod:`nlr_ghe_screen.api.osmnx_api`)
==============================================================

This page details the various functions involved with querying the OpenStreetMap databased via the
OSMNX Python library. OSMNX calls the Overpass API in the background. These functions handle calling the API, recognizing and handling errors, and passing the API
data through to other scripts. 

Return Data
-----------
OSMNX returns a Geopandas GeoDataframe with all of the tag attributes of each geometry and the coordinates that make up the shape of that geometry.

Functions
---------

.. automodule:: nlr_ghe_screen.api.osmnx_api
   :members:
   :undoc-members: