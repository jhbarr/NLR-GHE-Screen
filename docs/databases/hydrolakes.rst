HydroLAKES
==========
HydroLAKES is an open database that has information pertaining to over 1.4 million lakes and resevoirs worldwide.
The dataset contains information pertaining to each lake's name, location, depth, total surface area, estimated average depth,
and estimated total volume. 
More information regarding the database can be accessed at the `HydroLAKES Website <https://www.hydrosheds.org/products/hydrolakes>`_

Usage
-----
This database is used to enrich the data pertaining to the water bodies gathered by the program from the OpenStreetMap database. 
Though this functionality is not currently integrated into the final program workflow, a user can view what water bodies within 
their defined area have relevant HydroLAKES information through the cross_validation_visuals.ipynb python notebook. 

There does not exist an API for this database, therefore, the critical information is contained in a geoparquet file in the src/nlr_ghe_screen/data/
folder. 

License
------
From the HydroLAKES website "The HydroLAKES database is licensed under a `Creative Commons Attribution (CC-BY) 4.0 International License <https://creativecommons.org/licenses/by/4.0/>`_.
By downloading and using the data the user agrees to the terms and conditions of this license."