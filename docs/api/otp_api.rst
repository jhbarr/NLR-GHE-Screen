OpenTopography API (:py:mod:`scripts.otp_api`)
==============================================================

This page details the various functions involved with querying the OpenTopography database via their API. 
These functions handle calling the API, recognizing and handling errors, and passing the API
data through to other scripts. 

Queries
-------
The API requires four coordinates that describe a bounding box area where you want the elevation data for. Additionally,
you must provide which database you want your data to come from. For the purposes of this program, we are using the NASA
SRTMGL1 database

**API Key** - The API requires a key to work properly. Therefore, each user of the program will need to create an 
OpenTopography account and create their own personal key to user this program. The accounts and keys are both free. 

Return Data
-----------
The API returns bytes that need to be downloaded into .tif file format for usage.

Query Limits
------------
There is different rate limits based on the type of user account that you have
   - Academics / users with .edu email passwords are limited to 250 API calls in a 24-hour period
   - Non-academics are strictly limited to only 50 API calls in 24-hours
   
Additionally, the maximum query size differs based on which database you are querying. For our purposes, we will be using the SRTMGL1 database/
   - This database has a maximum query area of 450,000 km^2 per single API call



Functions
---------

.. automodule:: scripts.otp_api
   :members:
   :undoc-members:


Dependencies
------------
**Python**:

- pathlib.Path
- requests 
