import pandas as pd
import json

df = pd.read_csv('data/DDXPlus/test.csv').sample(50, random_state=42)
records = df.to_dict(orient='records')
with open('data/DDXPlus/eval_set.json', 'w') as f:
    json.dump(records, f, indent=2)
