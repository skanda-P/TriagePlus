import json
with open('data/DDXPlus/release_conditions.json') as f:
    d = json.load(f)
print(list(d.keys())[:3])
print(d[list(d.keys())[0]])