"""Answer evaluation: are answers correct and supported by the retrieved text?

Usage:
    python eval/eval_answers.py
    python eval/eval_answers.py --questions eval/questions_ta.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path

from check_questions import DEFAULT_QUESTIONS
from dotenv import load_dotenv
from eval_retrieval import build_store, load_questions, results_suffix
from google import genai
from google.genai import types

from chat_engine import ChatEngine
from config import EMBEDDING_MODEL, LLM_MODEL, RERANKER_MODEL, RETRIEVAL_MODE, TOP_K
from language import detect_language

load_dotenv()

PAUSE_SECONDS = 7  # stay under free-tier requests-per-minute limits


JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}
Expected answer: {expected}
System answer: {answer}
Retrieved context:
{context}

Grade two things:
- correct: does the system answer agree with the EXPECTED answer?
  - If the expected answer says the information is not in the document, the system
    is correct only if it says it could not find the answer.
  - If the expected answer gives information, but the system says it could not find
    the answer, correct is false, even if the retrieved context lacks the information.
    Judge correctness against the expected answer, not against the context.
- faithful: is every claim in the system answer supported by the retrieved context?
  An answer saying it could not find the information counts as faithful.

Write the reason in English. Reply with a single JSON object, not a list:
{{"correct": true or false, "faithful": true or false, "reason": "one short sentence"}}"""


def judge(client, question, expected, answer, sources):
    """Ask Gemini to grade one answer; returns a dict with correct, faithful, reason."""
    context = "\n\n".join(doc.page_content for doc in sources) or "(none)"
    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, answer=answer, context=context
    )
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json"
        ),
    )
    grade = json.loads(response.text)
    if isinstance(grade, list) and grade:  # the model sometimes wraps the object in a list
        grade = grade[0]
    if not isinstance(grade, dict) or not {"correct", "faithful"} <= grade.keys():
        raise ValueError(f"Unexpected judge output: {response.text!r}")
    return grade


def main():
    parser = argparse.ArgumentParser(description="Evaluate answer quality.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--tag", default="", help="extra label for the results file name")
    args = parser.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    results_file = Path(
        f"eval/results/answers_{RETRIEVAL_MODE}{results_suffix(args.questions)}{tag}.json"
    )

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    stores = {}
    rows = []

    for q in load_questions(args.questions):
        if q["doc"] not in stores:
            stores[q["doc"]] = build_store(q["doc"])
        result = ChatEngine(stores[q["doc"]]).get_answer(q["question"], k=TOP_K)
        time.sleep(PAUSE_SECONDS)

        grade = judge(client, q["question"], q["answer"], result["answer"], result["sources"])
        time.sleep(PAUSE_SECONDS)

        language_ok = detect_language(result["answer"]) == detect_language(q["question"])
        rows.append({"id": q["id"], "answer": result["answer"], "language_ok": language_ok, **grade})
        print(
            f"{q['id']}: correct={grade['correct']} faithful={grade['faithful']} "
            f"language_ok={language_ok} | {grade['reason']}"
        )

    n = len(rows)
    summary = {
        "questions_file": str(args.questions),
        "questions": n,
        "retrieval_mode": RETRIEVAL_MODE,
        "top_k": TOP_K,
        "model": LLM_MODEL,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "faithfulness": round(sum(r["faithful"] for r in rows) / n, 3),
        "language_match": round(sum(r["language_ok"] for r in rows) / n, 3),
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
    }
    print(json.dumps(summary, indent=2))

    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(
        json.dumps({"summary": summary, "per_question": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()