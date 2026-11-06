# AI Evaluation Framework (LLM-as-Judge)

An automated evaluation framework built from scratch to evaluate an LLM feature using an **LLM-as-Judge** pattern powered by **Gemini 2.0 Flash**. 

## 🎯 Methodology & Architecture
Instead of relying on rigid, error-prone evaluation wrappers, this framework uses a transparent **LLM-as-Judge** pattern. An automated Python script (`evaluate_custom.py`) passes test inputs, generated outputs, reference answers, and retrieved contexts to Gemini, instructing it to score the outputs objectively.

## 📋 Evaluation Metrics
The framework computes 3 distinct metrics (scored on a scale from `0.0` to `1.0`):
1. **Faithfulness / Groundedness:** Measures whether the generated output is strictly grounded in the provided context (checking for hallucinations).
2. **Answer Relevancy:** Measures how directly and precisely the output addresses the user's question.
3. **Semantic Correctness:** Compares the output against the reference ground truth to ensure factual accuracy.

## 📊 Aggregated Results
- **Total Test Cases:** 15 varied inputs (covering factual, technical, and conceptual queries).
- **Aggregated Average Scores (Sample Execution):**
  - **Faithfulness:** ~0.95 / 1.0
  - **Answer Relevancy:** ~0.93 / 1.0
  - **Semantic Correctness:** ~0.91 / 1.0

*(Detailed per-row metrics are saved locally in `custom_evaluation_results.csv`).*

## 🔍 Failure Case Analysis
1. **Edge Case #1 (Nuanced Terminology):** In queries requiring precise technical nuance (e.g., quantum computing/superposition), outputs that simplified the explanation too much scored lower on semantic correctness because key mechanical terms were omitted.
2. **Edge Case #2 (Context Scope):** When context documents were concise, outputs occasionally brought in minor outside general knowledge, slightly reducing the strict *Faithfulness* score despite remaining factually correct.

---
## 🚀 Execution Steps
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
