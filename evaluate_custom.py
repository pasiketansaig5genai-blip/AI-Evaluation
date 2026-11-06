import json
import os
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set your GOOGLE_API_KEY in the environment or .env file.")

client = genai.Client(api_key=api_key)


def judge_with_gemini(prompt: str) -> dict:
    """Sends evaluation prompt to Gemini 2.0 Flash and parses JSON score response."""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return json.loads(response.text)
    except Exception as e:
        return {"score": 0.0, "reasoning": str(e)}


def run_custom_evaluation():
    print("Initializing test dataset (15 examples)...")

    # Requirement: Test set of at least 15-20 real input examples
    test_cases = [
        {
            "id": 1,
            "question": "What is the capital of France?",
            "context": "Paris is the capital and most populous city of France.",
            "reference": "Paris",
            "output": "The capital of France is Paris, known for its art and culture.",
        },
        {
            "id": 2,
            "question": "How does photosynthesis work?",
            "context": (
                "Plants convert light energy into chemical energy through"
                " photosynthesis, absorbing CO2 and water."
            ),
            "reference": (
                "Plants use sunlight, water, and CO2 to produce oxygen and"
                " energy."
            ),
            "output": (
                "Photosynthesis allows plants to make sugar and oxygen using"
                " sunlight, water, and carbon dioxide."
            ),
        },
        {
            "id": 3,
            "question": "What are the health benefits of green tea?",
            "context": (
                "Green tea is loaded with bioactive compounds and antioxidants"
                " that provide various health benefits."
            ),
            "reference": (
                "Green tea improves brain function and fat loss due to"
                " antioxidants."
            ),
            "output": (
                "Green tea helps lower weight and boosts brain health because"
                " of antioxidants."
            ),
        },
        {
            "id": 4,
            "question": "Explain the theory of relativity briefly.",
            "context": (
                "General and special relativity formulated by Albert Einstein"
                " revolutionized physics regarding space, time, and gravity."
            ),
            "reference": (
                "Space and time are linked; gravity warps spacetime."
            ),
            "output": (
                "Einstein showed that space and time are connected and gravity"
                " is the bending of spacetime."
            ),
        },
        {
            "id": 5,
            "question": "What causes seasons on Earth?",
            "context": (
                "The tilt of the Earth's rotational axis relative to its"
                " orbital plane causes seasonal changes."
            ),
            "reference": "The Earth's axis tilt causes seasons.",
            "output": (
                "Seasons happen because Earth is tilted on its axis as it"
                " revolves around the sun."
            ),
        },
        {
            "id": 6,
            "question": "How do black holes form?",
            "context": (
                "Stellar black holes form by the gravitational collapse of a"
                " heavy star."
            ),
            "reference": "Collapse of massive stars.",
            "output": (
                "Black holes appear when giant stars explode and collapse under"
                " gravity."
            ),
        },
        {
            "id": 7,
            "question": "What is the function of mitochondria?",
            "context": (
                "Mitochondria generate adenosine triphosphate (ATP), the"
                " cellular energy currency."
            ),
            "reference": "Generate chemical energy (ATP) for cells.",
            "output": (
                "Mitochondria are cell powerhouses that produce energy (ATP)."
            ),
        },
        {
            "id": 8,
            "question": "Explain how a web search engine works.",
            "context": (
                "Search engines use web crawlers to discover content and"
                " indexing systems to serve queries quickly."
            ),
            "reference": "Crawls, indexes, and ranks web pages.",
            "output": (
                "Search engines crawl websites, build an index, and rank pages"
                " for matching keywords."
            ),
        },
        {
            "id": 9,
            "question": "What is machine learning?",
            "context": (
                "Machine learning focuses on algorithmic models allowing"
                " computers to learn from experience."
            ),
            "reference": "AI systems learning from data.",
            "output": (
                "Machine learning is a subset of AI where systems learn from"
                " data patterns automatically."
            ),
        },
        {
            "id": 10,
            "question": (
                "How does the human immune system respond to viruses?"
            ),
            "context": (
                "The immune response involves innate and adaptive mechanisms"
                " to clear viral agents."
            ),
            "reference": (
                "Produces antibodies and white blood cells to neutralize"
                " viruses."
            ),
            "output": (
                "The immune system creates specific antibodies and activates"
                " cells to fight off viruses."
            ),
        },
        {
            "id": 11,
            "question": "What is blockchain technology?",
            "context": (
                "Blockchain ensures tamper-evident, decentralized"
                " record-keeping via cryptographic hashing."
            ),
            "reference": (
                "Decentralized distributed ledger for secure transactions."
            ),
            "output": (
                "Blockchain is a secure, decentralized digital ledger spanning"
                " multiple computers."
            ),
        },
        {
            "id": 12,
            "question": "Why is the sky blue?",
            "context": (
                "Rayleigh scattering causes blue light from the sun to scatter"
                " across the atmosphere."
            ),
            "reference": (
                "Atmospheric scattering of shorter wavelengths (blue light)."
            ),
            "output": (
                "The sky looks blue because atmospheric gases scatter blue"
                " sunlight in all directions."
            ),
        },
        {
            "id": 13,
            "question": "What is GDP in economics?",
            "context": (
                "GDP is the standard measure of the value added created through"
                " the production of goods and services."
            ),
            "reference": (
                "Total monetary value of finished goods and services."
            ),
            "output": (
                "GDP measures the total market value of all goods and services"
                " produced in a country."
            ),
        },
        {
            "id": 14,
            "question": "How do vaccines work?",
            "context": (
                "Vaccination stimulates antibody production by exposing the"
                " immune system to a safe antigen proxy."
            ),
            "reference": (
                "Trains immune system using safe pathogen components."
            ),
            "output": (
                "Vaccines train your body's immune defenses by introducing a"
                " safe version of a pathogen."
            ),
        },
        {
            "id": 15,
            "question": "What is quantum computing?",
            "context": (
                "Quantum computers leverage superposition and entanglement to"
                " perform complex computations."
            ),
            "reference": (
                "Uses qubits and superposition for advanced computing."
            ),
            "output": (
                "Quantum computing utilizes qubits that operate on"
                " superposition principles to compute fast."
            ),
        },
    ]

    results = []

    print(
        "Running automated evaluation script using Gemini LLM-as-Judge..."
    )

    for item in test_cases:
        print(f"Evaluating test case {item['id']}/15...")

        # 1. Faithfulness Metric Prompt
        faith_prompt = f"""
        You are an objective AI evaluator. Rate the faithfulness of the OUTPUT based strictly on the provided CONTEXT on a scale of 0.0 to 1.0 (1.0 = completely faithful, no outside assumptions or hallucinations).
        Context: {item['context']}
        Output: {item['output']}
        Return ONLY a JSON object with keys: "score" (float) and "reasoning" (string).
        """
        faith_res = judge_with_gemini(faith_prompt)

        # 2. Answer Relevancy Metric Prompt
        rel_prompt = f"""
        You are an objective AI evaluator. Rate how directly the OUTPUT addresses the QUESTION on a scale of 0.0 to 1.0 (1.0 = highly relevant, directly answers the prompt).
        Question: {item['question']}
        Output: {item['output']}
        Return ONLY a JSON object with keys: "score" (float) and "reasoning" (string).
        """
        rel_res = judge_with_gemini(rel_prompt)

        # 3. Semantic Correctness Metric Prompt
        corr_prompt = f"""
        You are an objective AI evaluator. Compare the OUTPUT to the REFERENCE answer and rate factual accuracy on a scale of 0.0 to 1.0 (1.0 = perfectly matches key facts).
        Reference: {item['reference']}
        Output: {item['output']}
        Return ONLY a JSON object with keys: "score" (float) and "reasoning" (string).
        """
        corr_res = judge_with_gemini(corr_prompt)

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "faithfulness": faith_res.get("score", 0.0),
                "relevancy": rel_res.get("score", 0.0),
                "correctness": corr_res.get("score", 0.0),
            }
        )

    df = pd.DataFrame(results)

    print("\n--- Evaluation Results Summary ---")
    print(df.head(10))

    print("\n--- Aggregated Average Scores ---")
    print(
        df[["faithfulness", "relevancy", "correctness"]].mean()
    )

    df.to_csv("custom_evaluation_results.csv", index=False)
    print(
        "\nSaved complete evaluation dataset to custom_evaluation_results.csv!"
    )


if __name__ == "__main__":
    run_custom_evaluation()