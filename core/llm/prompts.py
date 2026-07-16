"""Prompt templates, user prompt builders, and output parsers for LLM Call 1 and 2."""

import json


# ---------------------------------------------------------------------------
# LLM Call 1 — RAG query generation
# ---------------------------------------------------------------------------

LLM1_SYSTEM = """You are a radiology AI assistant. Given chest X-ray CNN detection results \
and GradCAM++ activation summaries, generate one focused retrieval query per detected condition.

Output ONLY valid JSON with this exact schema:
{
  "rag_queries": [
    {"condition": "<exact_condition_name>", "query": "<8-15 word clinical retrieval query>"}
  ]
}

Rules:
- One entry per condition in above_threshold, same order
- Query describes the radiological and clinical features relevant to the condition
- Use condition names exactly as given (e.g., Pleural_Thickening not Pleural Thickening)
- Do not perform clinical interpretation or diagnosis"""


def build_llm1_user_prompt(
    above_threshold: list[str],
    all_scores: dict[str, float],
    semantic_context: str,
) -> str:
    """Build the user message for LLM Call 1 from CNN and GradCAM outputs."""
    scores_str = "\n".join(
        f"  {cond}: {score:.3f}" for cond, score in all_scores.items()
        if cond in above_threshold
    )
    return (
        f"Conditions above threshold (sorted by score):\n{scores_str}\n\n"
        f"GradCAM++ activation summary:\n{semantic_context}"
    )


LLM1_SCHEMA = {
    "type": "object",
    "properties": {
        "rag_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "query":     {"type": "string"},
                },
                "required": ["condition", "query"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rag_queries"],
    "additionalProperties": False,
}


def parse_llm1_output(text: str) -> dict:
    """Parse and validate LLM Call 1 JSON output; raises ValueError on bad schema."""
    data = json.loads(text)
    if "rag_queries" not in data:
        raise ValueError("Missing 'rag_queries' key in LLM Call 1 output")
    for item in data["rag_queries"]:
        if "condition" not in item or "query" not in item:
            raise ValueError(f"Malformed rag_query entry: {item}")
    return data


# ---------------------------------------------------------------------------
# LLM Call 2 — Clinical explanation
# ---------------------------------------------------------------------------

LLM2_SYSTEM = """You are a clinical decision-support AI assisting radiologists and physicians \
who are domain experts. The user is the specialist reading this output — do not include \
generic disclaimers or instructions to consult a physician.

Based on chest X-ray detection results, GradCAM++ zone activation analysis, and retrieved \
medical knowledge, provide a structured clinical explanation.

Output ONLY valid JSON with this exact schema:
{
  "conditions": [
    {
      "name": "<condition_name>",
      "explanation": "<2-3 sentence clinical explanation referencing relevant zones>",
      "dominant_zones": ["<zone_code>"]
    }
  ],
  "clinical_summary": "<1-2 sentence overall summary of findings>",
  "cross_specialty_notes": "<note if findings warrant cross-specialty correlation, else null>"
}

Rules:
- Do NOT make definitive diagnoses; use calibrated clinical hedging ('consistent with', 'differential includes')
- Only populate cross_specialty_notes when findings genuinely suggest correlation with another \
specialty (e.g., cardiology for cardiomegaly with suspected heart failure, oncology for a mass \
with malignant features); otherwise set it to null. This is a peer referral note, not a disclaimer.
- Professional clinical tone appropriate for specialist-to-specialist communication
- Respond in the same language as the user prompt (Indonesian or English)"""


def build_llm2_user_prompt(
    above_threshold: list[str],
    all_scores: dict[str, float],
    semantic_context: str,
    rag_chunks: list[dict],
) -> str:
    """Build the user message for LLM Call 2 from GradCAM output and retrieved chunks."""
    # Format retrieved knowledge grouped by condition
    chunks_by_condition: dict[str, list[str]] = {}
    for chunk in rag_chunks:
        cond = chunk["condition"]
        entry = f"[{cond} - {chunk['section']}]: {chunk['text']}"
        chunks_by_condition.setdefault(cond, []).append(entry)

    knowledge_str = "\n\n".join(
        "\n".join(entries) for entries in chunks_by_condition.values()
    )

    scores_str = ", ".join(
        f"{c} ({all_scores[c]:.2f})" for c in above_threshold
    )

    return (
        f"Detected conditions: {scores_str}\n\n"
        f"GradCAM++ activation analysis:\n{semantic_context}\n\n"
        f"Retrieved clinical knowledge:\n{knowledge_str}"
    )


LLM2_SCHEMA = {
    "type": "object",
    "properties": {
        "conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":           {"type": "string"},
                    "explanation":    {"type": "string"},
                    "dominant_zones": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "explanation", "dominant_zones"],
                "additionalProperties": False,
            },
        },
        "clinical_summary":      {"type": "string"},
        "cross_specialty_notes": {"type": ["string", "null"]},
    },
    "required": ["conditions", "clinical_summary", "cross_specialty_notes"],
    "additionalProperties": False,
}


def parse_llm2_output(text: str) -> dict:
    """Parse and validate LLM Call 2 JSON output; raises ValueError on bad schema."""
    data = json.loads(text)
    for key in ("conditions", "clinical_summary", "cross_specialty_notes"):
        if key not in data:
            raise ValueError(f"Missing '{key}' key in LLM Call 2 output")
    for item in data["conditions"]:
        for field in ("name", "explanation", "dominant_zones"):
            if field not in item:
                raise ValueError(f"Malformed condition entry: {item}")
    return data


# ---------------------------------------------------------------------------
# Text Q&A path — general clinical question answering
# ---------------------------------------------------------------------------

TEXT_QA_SYSTEM = """You are a clinical decision-support AI assisting radiologists and physicians \
who are domain experts. The user is asking a direct clinical question — do not include generic \
disclaimers or instructions to consult a physician.

If prior image findings or conversation history are provided, treat this as a continuing \
discussion about the same case — refer to those findings naturally rather than re-explaining \
them from scratch.

Based on the retrieved clinical knowledge, answer clearly and accurately.

Output ONLY valid JSON with this exact schema:
{
  "answer": "<direct clinical answer, 2-5 sentences>",
  "cross_specialty_notes": "<note if the question touches another specialty, else null>"
}

Rules:
- Use calibrated clinical hedging where evidence is inconclusive ('typically', 'in most cases')
- If retrieved knowledge is insufficient to answer confidently, state that clearly rather than guessing
- Only populate cross_specialty_notes when genuinely relevant, else null
- Respond in the same language as the user's question (Indonesian or English)"""

TEXT_QA_SCHEMA = {
    "type": "object",
    "properties": {
        "answer":                {"type": "string"},
        "cross_specialty_notes": {"type": ["string", "null"]},
    },
    "required": ["answer", "cross_specialty_notes"],
    "additionalProperties": False,
}


def build_text_qa_user_prompt(
    query: str,
    rag_chunks: list[dict],
    prior_context: dict | None = None,
) -> str:
    """Build the user message for text Q&A from the query, retrieved chunks, and optional prior context."""
    parts = []

    if prior_context:
        above = prior_context.get("above_threshold") or []
        if above:
            parts.append(f"Prior image findings in this conversation: {', '.join(above)}")
        conversation = prior_context.get("conversation") or []
        if conversation:
            history_str = "\n".join(f"{t['role']}: {t['content']}" for t in conversation[-6:])
            parts.append(f"Recent conversation history:\n{history_str}")

    parts.append(f"Question: {query}")

    if rag_chunks:
        knowledge_str = "\n\n".join(
            f"[{c['condition']} - {c['section']}]: {c['text']}" for c in rag_chunks
        )
        parts.append(f"Retrieved clinical knowledge:\n{knowledge_str}")
    else:
        parts.append("No relevant knowledge base entries retrieved.")

    return "\n\n".join(parts)


def parse_text_qa_output(text: str) -> dict:
    """Parse and validate text Q&A JSON output; raises ValueError on bad schema."""
    data = json.loads(text)
    for key in ("answer", "cross_specialty_notes"):
        if key not in data:
            raise ValueError(f"Missing '{key}' key in text Q&A output")
    return data