OpenSteetMap Reference
======================
OpenSteetMap is a free, open-source geospatial database that provides crowdsourced information regarding the geographic location, 
physical shape and attributes of buildings and other spaces around the globe.

Usage
-----
This database was used in order to compile a dataset containing all possible greenspace and parking lot locations within a given user defined
boundary where ground heat exchangers (GHEs) could theoretically be installed. 

Tags
----
A tag is the way that the OpenSteetMap database can attach an attribute to a geographic location. For example, a parking lot 
can have a tag attached to it in the form {'amenity': 'parking'}

Tags of Interest
----------------
All tag information was retrieved from the OpenStreetMap wiki:
.. _Tag Information: https://wiki.openstreetmap.org/wiki/Map_features

For querying park locations and other greenspaces

- **Leisure** - Tags leisure and sport facilities but also parks and other such green spaces like dog parks and fields used for sports

    - dog_park - Park area for dogs
    -  garden - Any type of garden, public or private
    - park - Typically located in urban areas

- **Natural** - Describes physical land features like grasslands, woods and other such areas

    - grassland - Area where vegetation is dominated by grasses
    - heath - shrub habitat
    - shrub - area covered in shrubs
    - shrubbery - human maintained area of woods
    - wood - tree covered area 

- **Landuse** - This is used to describe the purpose for which an area of land is being used. This can include things like a private allotment of land or a forest.

    - forest - Managed forest or woodland area
    - logging - Area where trees have been cut down
    - meadow - A meadow or pasture
    - orchard - Intentional planted tree area
    - grass - An area of mown grass
    - recreation_ground - Open green space for general recreation
    - greenery - Any area covered with landscaping or decorative greenery 

For querying parking lot locations

- **Amenity** - Facilities used by humans

    - Parking - parking area for vehicles (often tagged highway=service and service=parking_aisle)

For filtering parks that may not be of use

- **Boundary** - This could mark a park as a national park or a protected area that would therefore be unusable in the case of a GHE

    - Aboriginal_lands - represents official reservation boundaries of recognized native peoples
    - Forest - land which is predominantly wooded. Considered a managed forest
    - Forest_compartment - a numbered subdivision within a delimited forest
    - National_park - denotes a national park
    - Protected_area - general protected areas, such as for national parks, heritage sites, wilderness, etc 


License
-------
OpenSteetMap was released under the Open Database License (ODbL). Therefore, it is free to use for research purposes, provided that
proper attribution is given. 
