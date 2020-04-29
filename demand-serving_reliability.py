import numpy as np
import networkx as nx
from heapq import *
from itertools import *
import matplotlib as mpl
import matplotlib.pyplot as plt
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''                    This code is developed by Homayoun Hamedmoghadam based on the study:                      '''
'''            Percolation of heterogeneous flows uncovers the bottlenecks of demand-serving networks            '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


# find the limiting link between a SOURCE and all TARGETS on network G
# this is basically a modified version of DIJKSTRA shortes-path algorithm, implemented using HEAPS
def maximum_capacity_paths(G, source, weight, target=None):
    get_weight = lambda u, v, data: data.get(weight, 1)
    paths = {source: [source]}  # dictionary of paths
    G_succ = G.succ if G.is_directed() else G.adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    pred = {}  # dictionary of final limiting links
    seen = {source: 0}
    c = count()
    fringe = []
    push(fringe, (-1, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():
            if u == source:
                continue
            vu_dist = max([dist[v], -get_weight(v, u, e)])
            if u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]

                if dist[v] > -get_weight(v, u, e):
                    pred[u] = pred[v]
                else:
                    pred[u] = (v, u)

    dist = {i: -dist[i] for i in dist}
    return dist, pred


# Build an example network:
# a 25*25 2D lattice (grid) network with random link qualities q \in (0,1)
print('Generating a network with link-dynamics and a flow demand over it [...', end='')
gridSize = 25 * 25
net = nx.Graph()
for nodeNumber in range(1, gridSize+1):
    if nodeNumber % np.sqrt(gridSize) != 0:
        net.add_edge(nodeNumber, nodeNumber+1, q=np.random.rand())
    if nodeNumber % np.sqrt(gridSize) != 1:
        net.add_edge(nodeNumber, nodeNumber-1, q=np.random.rand())
    if (nodeNumber-1.0) / np.sqrt(gridSize) >= 1:
        net.add_edge(nodeNumber, nodeNumber-np.sqrt(gridSize), q=np.random.rand())
    if float(nodeNumber) / np.sqrt(gridSize) < np.sqrt(gridSize)-1:
        net.add_edge(nodeNumber, nodeNumber+np.sqrt(gridSize), q=np.random.rand())

# Generating a random flow demand between pairs of nodes over the network
total_demand = 10000 # predefined total units of flow
OD = {}
randOriginIdxs = np.random.choice(gridSize, total_demand, p=[1.0 / gridSize for _ in range(1, gridSize+1)])
for origin in randOriginIdxs:
    randDestination = np.random.choice(gridSize,
        p=[(1.0 / (gridSize-1)) if node != origin else 0
            for node in range(0, gridSize)])
    if (origin+1, randDestination+1) not in OD:
        OD[(origin+1, randDestination+1)] = 0
    OD[(origin+1, randDestination+1)] += 1
print('Done]')


print('Calculating the criticality score of all links [...', end='')
criticalityScores = {}
for Orig in net:
    # Find the limiting link between the selected origin and each destination available
    _, limitingLinksDict = maximum_capacity_paths(net, Orig, weight='q')
    for Dest in limitingLinksDict:
        if (Orig != Dest) and ((Orig, Dest) in OD):
            if limitingLinksDict[Dest] not in criticalityScores:
                criticalityScores[limitingLinksDict[Dest]] = 0
            criticalityScores[limitingLinksDict[Dest]] += OD[(Orig, Dest)]/float(total_demand)
# Make a list of all links with their QUALITY and calculated CRITICALITY SCORES
linkInfoList = [(link[0], link[1], net[link[0]][link[1]]['q'], criticalityScores[link]) for link in criticalityScores]
print('Done]')

# Here using the theory provided in 'Hamedmoghadam et al.' we reconstruct the Unaffected Demand as a function of \rho
# This can be found and illustrated through a percolation process on the network
# With the help of calculated criticality scores, we can mimic exactly what happens during a percolation process
alpha = np.sum([link[2]*link[3] for link in linkInfoList])
plt.figure(figsize=(8,4))
mpl.rcParams.update({'font.size': 18})
plt.plot([rho/1000.0 for rho in range(1001)],
         [np.sum([link[3] for link in linkInfoList if link[2]>rho/1000.0]) for rho in range(1001)],
         'k', marker='s', markersize=2, fillstyle='none', linewidth=2, alpha=1)
plt.fill_between([rho/1000.0 for rho in range(1001)],
                 0.00, [np.sum([link[3] for link in linkInfoList if link[2]>rho/1000.0]) for rho in range(1001)],
                 facecolor='none', hatch="//", alpha=1, interpolate=True, label=r'$\alpha$='+str(round(alpha,3)))
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Unaffected Demand')
plt.xlabel(r'$\rho$')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

