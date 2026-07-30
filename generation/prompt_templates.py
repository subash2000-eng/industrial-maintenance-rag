"""
Phase 3 - Prompt Templates
Structures the information we send to the LLM.
Good prompts = precise, safe, cited answers.
Bad prompts  = hallucinations and vague answers.
"""


# ── System prompt — defines LLM's role ──────────────────
SYSTEM_PROMPT = """You are an expert industrial maintenance \
engineer with 20 years of hands-on experience.
You help field technicians diagnose machine faults and \
perform repairs using official maintenance manuals.
Always be precise, safety-conscious, and cite your sources.
Never guess or make up information not found in the manuals."""


def build_rag_prompt(
    query           : str,
    retrieved_chunks: list
) -> str:
    """
    Build the complete prompt for the LLM.

    Combines:
    - Retrieved manual sections as context
    - Technician's fault description
    - Clear instructions for the LLM
    - Required output format

    Args:
        query           : Technician's fault description
        retrieved_chunks: Top-3 chunks from retrieval pipeline

    Returns:
        Complete prompt string ready for LLM
    """

    # ── Build context from retrieved chunks ──────────────
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        block = (
            f"[SOURCE {i+1}]\n"
            f"Manual  : {chunk['manual_name']}\n"
            f"Page    : {chunk['page_number']}\n"
            f"Section : {chunk['section_title']}\n"
            f"Content :\n{chunk['text']}"
        )
        context_blocks.append(block)

    # Join all source blocks
    context = "\n\n" + ("─" * 56) + "\n\n"
    context = context.join(context_blocks)

    # ── Build full prompt ────────────────────────────────
    prompt = f"""{SYSTEM_PROMPT}

{"="*56}
RETRIEVED MANUAL SECTIONS:
{"="*56}

{context}

{"="*56}
TECHNICIAN FAULT DESCRIPTION:
{"="*56}
{query}

{"="*56}
YOUR INSTRUCTIONS:
{"="*56}
1. Diagnose the most likely fault based on the description
   and the manual content provided above.
2. Provide clear numbered step-by-step repair instructions.
3. Add safety warnings where needed — prefix with WARNING.
4. List required tools and parts if mentioned in manuals.
5. Cite sources using [SOURCE 1], [SOURCE 2] notation.
6. If the manuals do not contain enough information,
   say so clearly — do not guess or make up steps.

{"="*56}
REQUIRED OUTPUT FORMAT:
{"="*56}

## Fault Diagnosis
[Your diagnosis based on manual content]

## Step-by-Step Repair Instructions
1. [First step]
2. [Second step]
3. [Continue as needed]

## Safety Warnings
[Any safety warnings from the manuals]

## Required Tools and Parts
[Tools and parts mentioned in the manuals]

## Sources Used
[SOURCE 1] - Manual name | Page number | Section
[SOURCE 2] - Manual name | Page number | Section
"""
    return prompt


def build_no_context_response() -> str:
    """
    Response when no relevant chunks are found.
    Used instead of letting LLM guess with no context.
    """
    return """No relevant information was found in the
available maintenance manuals for this fault description.

Please try:
1. Rephrasing your query with specific fault codes
   or symptoms
2. Checking if the relevant manual has been ingested
3. Using more specific technical terms from the manual
4. Contacting your maintenance supervisor directly

The system will only provide answers grounded in the
ingested maintenance manuals."""


def preview_prompt(query: str, retrieved_chunks: list) -> None:
    """
    Print a preview of the prompt that will be sent to LLM.
    Useful for debugging prompt structure.
    """
    prompt = build_rag_prompt(query, retrieved_chunks)

    print(f"\n{'='*56}")
    print(f"  PROMPT PREVIEW")
    print(f"  Query: '{query}'")
    print(f"  Chunks: {len(retrieved_chunks)}")
    print(f"  Total characters: {len(prompt)}")
    print(f"  Approx tokens: {len(prompt)//4}")
    print(f"{'='*56}\n")
    print(prompt[:1000])
    print("\n... [prompt continues] ...\n")


# ── Run this file directly to test ──────────────────────
if __name__ == "__main__":

    # Create sample chunks to test prompt building
    sample_chunks = [
        {
            "manual_name"  : "SIEMENS Motor Manual",
            "page_number"  : 16,
            "section_title": "Motor Troubleshooting Chart",
            "text"         : (
                "Hot bearings: Cause — Excessive belt pull. "
                "Remedy — Decrease belt tension. "
                "Pulley too far away from bearing — "
                "Move pulley closer to bearing."
            ),
            "rerank_score" : 6.82,
            "confidence"   : "Very High"
        },
        {
            "manual_name"  : "Compressed Air Manual",
            "page_number"  : 97,
            "section_title": "Electrical Installation",
            "text"         : (
                "Motor protection includes thermal overload "
                "relay set to rated current. "
                "Check temperature class of motor insulation. "
                "Bearing temperature must not exceed 90 degrees C."
            ),
            "rerank_score" : 3.12,
            "confidence"   : "High"
        },
        {
            "manual_name"  : "SIEMENS Motor Manual",
            "page_number"  : 13,
            "section_title": "Safety Precautions",
            "text"         : (
                "Before starting any work on the motor, "
                "disconnect all power sources. "
                "Allow motor to cool completely before touching. "
                "Use insulated tools only."
            ),
            "rerank_score" : 1.45,
            "confidence"   : "Medium"
        }
    ]

    sample_query = "motor bearing overheating after 2 hours of operation"

    # Preview the prompt
    preview_prompt(sample_query, sample_chunks)

    # Show no context response
    print("\n" + "─"*56)
    print("NO CONTEXT RESPONSE:")
    print("─"*56)
    print(build_no_context_response())