Overview of the Features
========================
This page gives an overview of the different features in the program and their purposes / what they do.

GHE Screening
-------------
The GHE Screening features are designed to parse and filter information from various data sources that provide information 
about the location of greenspace, parking lots, and where those spaces sit topographically. Overall, these steps
create a finalized database that suggests locations for the possible installation of ground heat exchangers (GHEs).

|

**GeoJSON Validation**
GeoJSON validation is used to ensure that the data output by the program has all of the attribute fields required by the 
NLR URBANopt software, so that the GHE Screening functions and output can easily integrate with URBANopt down the line. 

|

**Geometry Manipulation**
Each of the spaces queried and parsed by the program area assigned different categories based on their attribute fields. They are either
categorized as greenspace or as parking lot areas. Some of the greenspace areas are designated as protected areas if they have attributes that 
designate them as such. 

Once each of the spaces is assigned a category, if two spaces that share the same category overlapo with each other physically, those two spaces are combined
into one larger geometry. Additionally, their attributes are combined. 

|

**MS Footprint Cross Validation**
MS footprint provides information regarding building heights. However, because this program is concerned with open spaces, MS footprint is used solely
to check whether areas labels as parking lots are in fact parking structures. Those spaces are excluded from the final results of the program because 
parking structures are not valid spaces for GHE installation

|

**Elevation Exclusion**
Areas that are located on steep inclines do not make for ideal GHE installation sites. Therefore, the program utilizes a topographical database 
to rule out any areas that rest a majority on a steep incline. 