Overpass API (:py:mod:`scripts.osm_api`)
==============================================================

This page details the various functions involved with querying the OpenStreetMap databased via the
Overpass API. These functions handle calling the API, recognizing and handling errors, and passing the API
data through to other scripts. 

Overpass QL Queries
-------------------
API requests to the Overpass API have to include an actual query for specific geographic attributes. The query is written in 
Overpass QL. Additionally, it must include the geographic area that you want to run the query for. For this program, the query is pre-written 
and constant through all API requests, since the geographic attributes that the program is looking for does not change based on location. 
The only thing that does change is the desired query location, which is provided by the user.

Return Data
-----------
The API returns a JSON response body that includes the tag attributes of each geometry and the coordinates that make up the shape of that geometry.

Query Limits
------------
There are no express limnits to the size of the areas that you can query using the Overpass API. Instead the query is subject
to dynamic memory limits and a timeout depending on the server load. 
   - **Memory limit** - The default is 512 MB of RAM. Therefore, depending on the size of the bounding box search area or how dense the 
   elements are within the bounding box, the query may exceed the server limit and abort
   - **Query Limit** - Users are restricted to under 10,000 queries per day to the Overpass API
   - If all available slots to the server are full, or if the server is struggling under high load, then the API will reject with a 429 error (too many requests). 
   If this is the case, then you will need to have a cool-down period before you can re-run the query

Functions
---------

.. automodule:: scripts.osm_api
   :members:
   :undoc-members:


Dependencies
------------
**Python**:

- time
- requests 
