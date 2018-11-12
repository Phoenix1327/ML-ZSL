# imagenet_parse
Imagenet is a large scale dataset that researchers use to build large scale visual recognition models. 
This database is organized according to a hierarchy defined by Wordnet. 
See http://image-net.org/index for more information.   

The code reads the graph structure from the XML file.

and implement tasks defined below:   

* Find all ancestors and descendants of a node specified by its WordNet ID/synset ID 
* Find the depth of a node from the root for a given node 
* Find the distance between two nodes specified by their WordNet IDs/synset ID’s   

The XML graph structure can be found here: http://www.image-net.org/api/xml/structure_released.xml”
