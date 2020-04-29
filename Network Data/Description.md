Each network is generated for a 2-hour interval.
Networks are pickled Python lists.
Each Pickle name is in the format network_<day index>_<interval start>-<interval end>.p.
Day indices 1 to 14 correspond to 1st to 14th of September 2017, networks of different intervals during each day are stored in a single ZIP file.
Interval start and end are in 'hhmm' format.

In the pickeld Python list, the set of tuples correspond to the network links and each tuple contains:
(source node, target node, link quality)

Nodes are referred to by integers each corresponding to a stop or stop-hub in the system.
The file nodeCoordinates.p is a Pickle made from a Python dictionary, where keys are node indices and values are tuples containing (Latitude, Longitude) of each associated node.