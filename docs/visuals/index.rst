GHE Visualization Features
==========================
These are files located in the visuals/ folder of the software program respository. These Python 
notebooks provide visualizations for different data sources and the procedures to import and processes
that information.

*Note* - Some of these procedures are not implemented in the main program workflow. Instead, these files
should be used by the user to check the accuracy of the program and interact with its raw results in a more
approachable manner.

|

ghe_location_visuals.ipynb
--------------------------
This notebook simply launches a notebook cell that allows the user to visualize and explore the locations and shapes
of the different spaces compiled by the program. Additionally, in this exploratory mode, for each of the spaces, 
the user can view what kind of space it is, and any other compiled attribute. 

|

elevation_data_visuals.ipynb
----------------------------
This notebook allows the user to both view the elevaiton and gradient of specific vector shapes from the compiled 
program data, or view the entire inner elevation changes and gradient changes of the bounding box area described
by the user in the first place. 

|

cross_validation_visuals.ipynb
------------------------------
This notebook allows users to perform cross-validation on areas designated as greenspace and water using the ESA WorldCover database. 
It leverages functionality from the processing/land_area_validation.py script to determine what percentage of the designated greenspace and water 
areas can be verified as those land-use types using ESA data.

Users can also view areas that could not be cross-validated, along with information such as their location, size, and other attributes. 
This allows for visual inspection and identification of potential patterns or inconsistencies in the results.

Additionally, users can inspect individual shape vectors alongside their corresponding ESA raster data to examine the composition 
of land-use types within each shape.