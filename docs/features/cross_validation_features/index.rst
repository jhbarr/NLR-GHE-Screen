Cross Validation Features
=========================
This page describes how the user can utilize the ESA WorldCover database and cross validation scripts and python notebooks
to interactively cross validate the different greenspaces, water bodies, and other areas gathered by the main program from
OpenStreetMap. 

Additionally, the user can enrich the information pertaining to the water bodies gathered by the program using the HydroLakes
dataset and the functions to gather and spatially join that information to aggregated water data from OSM. 

Space Cross Validation Procedures
---------------------------
To ensure the accuracy of the data provided by OpenStreetMap relating to the size and location of various kinds of greenspaces and water bodies, 
the program has functionality to cross validate those spaces with data from the ESA WorldCover database. This database provides raster data that 
categorizes areas on the earth’s surface based on what kind of land use predominates that area. 

These categories include tree cover, shrubland, grassland, cropland, permanent water bodies, herbaceous wetland, mangroves, moss and lichen, as well as built-up. 
Any area categorized as built-up indicates that human settlement and construction predominates that area of land. For each area categorized as greenspace or water 
by the OpenStreetMap database, the program can take those geometries and check what percentage of them are covered in natural areas. If over, according to the ESA data, 
over 80% of the area is not built up by humans and is consistent of either greenery or water, the program considers it a cross-valid space. 

A case study was  performed on two different cities to investigate theefficacy of the cross validation procedures.
For informtion about the case studies, see :ref:`case-study`

Cross Validation Visuals
------------------------
cross_validation_visuals.ipynb

This notebook allows users to perform cross-validation on areas designated as greenspace and water using the ESA WorldCover database. 
It leverages functionality from the processing/land_area_validation.py script to determine what percentage of the designated greenspace and water 
areas can be verified as those land-use types using ESA data.

Users can also view areas that could not be cross-validated, along with information such as their location, size, and other attributes. 
This allows for visual inspection and identification of potential patterns or inconsistencies in the results.

Additionally, users can inspect individual shape vectors alongside their corresponding ESA raster data to examine the composition 
of land-use types within each shape.

Functions
---------

.. toctree::
    :maxdepth: 1
    :caption: Cross Validation Scripts

    space_cross_validation