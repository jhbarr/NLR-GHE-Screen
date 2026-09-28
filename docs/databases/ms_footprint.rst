Microsoft Footprint Reference
=============================
An open-access geospatial dataset containing over 1.4 billion computer generated building outlines globally.

Usage
-----
MS Footprint is used to cross validate that all parking lot areas compiled from the OpenStreetMap database are not parking garages. Each of the
parking lot spaces from OpenSteetMap are queried in the MS Footprint data and any spaces that have a height above a certain threshold are excluded from 
the final exported results, since parking garages are not viable for GHE installation. 

License
-------
MS Footprint was released under the Open Commons Database License (ODbL). Therefore, it is free to use for research purposes, provided that
proper attribution is given. 