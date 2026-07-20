import pickle
with open('data/ddxplus_kg.pkl', 'rb') as f:
    kg_data = pickle.load(f)
print('Graph nodes:', kg_data['graph'].number_of_nodes())
print('Graph edges:', kg_data['graph'].number_of_edges())
print('Conditions:', len(kg_data['conditions']))
print('Evidences:', len(kg_data['evidences']))
print('evidence_condition_counts keys:', len(kg_data['evidence_condition_counts']))
print('condition_evidence_counts keys:', len(kg_data['condition_evidence_counts']))
if kg_data['evidence_condition_counts']:
    sample_key = list(kg_data['evidence_condition_counts'].keys())[0]
    print(f'Sample evidence {sample_key}: {kg_data["evidence_condition_counts"][sample_key]}')