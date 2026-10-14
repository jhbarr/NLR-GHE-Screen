Building Footprint Aggregation (:py:mod:`nlr_ghe_screen.api.building_footprint_api`)
==============================================================

This page details the various functions involved with querying both the OvertureMaps and ORNL USA Structures databases via 
their respective APIs. These functions handle calling the API, recognizing and handling errors, and passing the API
data through to other scripts. 

Return Data
-----------
*OvertureMaps* - Returns a fully parsed geopandas.GeoDataframe object with all of the building information
*ORNL* - Returns raw feature data from each of the API queries that can be easily parsed into a geopandas.GeoDataframe object

Query Limits
------------
*OvertureMaps* - The data is stored on the cloud in an Amazon S3 bucket, so there is no real limit with respect to how many times it can be queried. 
*ORNL* - There is no single day query limit like other APIs. However, a single API call can only respond with a maximum of 2,000 elements
    Therefore, to gather all of the building footprints in a large user defined area, the program must query the API many times. Which 
    Risks a rate limiting error. 

Functions
---------

.. automodule:: nlr_ghe_screen.api.building_footprint_api
   :members:
   :undoc-members:
