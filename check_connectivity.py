import pandas as pd
from collections import defaultdict

rel = pd.read_csv('data/Beijing/roadmap.rel')
geo = pd.read_csv('data/Beijing/roadmap.geo')

# Count connections per road
outgoing = defaultdict(int)
incoming = defaultdict(int)

for _, row in rel.iterrows():
    outgoing[row['origin_id']] += 1
    incoming[row['destination_id']] += 1

# Find dead ends and isolated roads
all_roads = set(geo['geo_id'])
connected_roads = set(outgoing.keys()) | set(incoming.keys())
isolated = all_roads - connected_roads
dead_ends = [r for r in connected_roads if outgoing[r] == 0]

print(f'Total roads: {len(all_roads)}')
print(f'Isolated roads (no connections): {len(isolated)}')
print(f'Dead-end roads (no outgoing): {len(dead_ends)}')
print(f'Roads with outgoing connections: {len([r for r in outgoing if outgoing[r] > 0])}')

# Show some examples
if isolated:
    print(f'\nExample isolated roads: {list(isolated)[:5]}')
if dead_ends:
    print(f'Example dead-end roads: {list(dead_ends)[:5]}')
