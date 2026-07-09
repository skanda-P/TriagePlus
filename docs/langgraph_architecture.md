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
    confidence: float
    urgency: int
    booking_intent: Optional[bool]
    available_slots: Optional[List[Dict[str, Any]]]
    selected_slot_id: Optional[str]
    payment_status: Optional[str]
    rag_chunks: Optional[List[str]]
    latencies: Optional[Dict[str, float]]
```

## Graph Nodes

- **`node_extract_symptoms`**: Uses a `transformers` Medical NER pipeline (`d4data/biomedical-ner-all`) to extract precise clinical entities from the chat and map them to DDXPlus evidence codes.
- **`node_emergency_check`**: A safety net that checks hardcoded emergency keywords and maps to severe DDXPlus conditions.
- **`node_next_question`**: Interrogates the Knowledge Graph (NetworkX) to select the next best question based on Information Gain, augmented with MedDialog FAISS RAG to formulate an empathetic question via LLM.
- **`node_classify`**: Once enough symptoms are gathered, executes the XGBoost model to predict a pathology and selects a department.
- **`node_explain`**: Drafts a conversational response summarizing the diagnosis using MedQuAD FAISS RAG.
- **`node_prompt_booking`**: Asks the user if they want to book an appointment with the recommended department.
- **`node_handle_booking`**: Parses the user's intent to book.
- **`node_fetch_slots`**: Queries Supabase for available open `clinician_slot` records and lists them.
- **`node_confirm_slot`**: Processes the patient's slot selection and inserts an `appointment` record.
- **`node_process_payment`**: Simulates payment processing and confirms the appointment booking.

## Diagnostics Event Stream

Each time the LangGraph updates, the state changes are broadcasted via WebSockets to `/ws/diagnostics` for real-time monitoring of RAG chunks, latencies, and state evolution in the frontend Developer Monitor.

## Checkpointing (SqliteSaver)

Because LangGraph operates on WebSocket streaming, connections can drop. TriagePlus implements `SqliteSaver` pointing to `backend/triage_checkpoints.sqlite` to persist `TriageState` per thread (`session_id`).
