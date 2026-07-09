# LangGraph Architecture Details

The brain of the TriagePlus conversational agent is governed by LangGraph. It provides a stateful loop that allows the agent to iteratively gather symptoms from the patient until it has sufficient confidence to make a classification.

## The Triage State (TypedDict)

```python
class TriageState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    patient_id: Optional[str]
    age: int
    gender: str
    present_symptoms: List[str]  # List of DDXPlus E_* codes
    absent_symptoms: List[str]   # List of DDXPlus E_* codes
    is_emergency: bool
    final_diagnosis: Optional[str]
    department: Optional[str]
    triage_summary: Optional[str]
    question_count: int
```

## Graph Nodes

- **`node_extract_symptoms`**: Analyzes the latest patient message using a structured LLM system prompt (`SLOT_EXTRACTION_PROMPT` / `SYMPTOM_MAPPING_PROMPT`) to convert natural language into DDXPlus evidence codes.
- **`node_emergency_check`**: A dual-layered safety net:
  1. Checks hardcoded emergency keywords (e.g., "chest pain", "suicide").
  2. Uses the `EMERGENCY_SCREENING_PROMPT` to analyze context and severity.
- **`node_next_question`**: Interrogates the Knowledge Graph (NetworkX) to select the next best question based on Information Gain from the remaining un-asked evidences.
- **`node_classify`**: Once enough symptoms are gathered, this node executes the XGBoost model to predict a pathology and selects a department. It incorporates T2 Mitigation by flooring the urgency score on low confidence.
- **`node_explain`**: Drafts a conversational response summarizing the diagnosis and routing, and pushes the completed session data directly into Supabase.

## Checkpointing (SqliteSaver)

Because LangGraph operates on WebSocket streaming, connections can drop. TriagePlus implements `SqliteSaver` pointing to `backend/triage_checkpoints.sqlite` to persist `TriageState` per thread (`session_id`). This completely eliminates session leakage and allows patients to resume abruptly disconnected sessions exactly where they left off.
