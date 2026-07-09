# Knowledge Bases & RAG Architecture

The TriagePlus system is built upon robust medical datasets that provide the underlying intelligence for the machine learning models and knowledge graphs.

## Dataset Acknowledgments

Due credit is given to the following exceptional open-source medical datasets which make this project possible:

1. **DDXPlus Dataset**: Provided the structural foundation for medical symptoms, antecedents (risk factors), and 49 different pathologies.
2. **MedQuAD (Medical Question Answering Dataset)**: Provided extensive Q&A pairs for clinical facts.
3. **Symptom2Disease Dataset**: Used to augment symptomatic text and conversational variance during the training phase.

## Knowledge Graph (NetworkX)

Rather than strictly relying on semantic vector search (which can hallucinate when traversing complex medical trees), TriagePlus utilizes a deterministically parsed **Directed Graph (NetworkX)** from the DDXPlus schema (`release_conditions.json` and `release_evidences.json`).

### Information Gain Algorithm

During the `node_next_question` step in the LangGraph pipeline, the Knowledge Graph executes an Information Gain algorithm:
1. Filters down possible pathologies based on confirmed symptoms.
2. Eliminates pathologies that require symptoms the patient explicitly denied.
3. Ranks the remaining un-asked evidence nodes by calculating which symptom effectively cuts the remaining candidate list in half (maximizing information gain).

## The Predictive Model (XGBoost)

Instead of relying on an LLM to "guess" the diagnosis from text, we use an XGBoost classifier.
- **Training**: The model is trained on a synthetic binary matrix derived from the DDXPlus synthetic patient records.
- **Features**: Patient Age, Sex, and a Multi-Label Binarized vector of extracted `E_*` evidence codes.
- **Output**: Predicts one of the 49 pathologies and provides a confidence probability matrix, which is used to enforce urgency flooring mitigations.
