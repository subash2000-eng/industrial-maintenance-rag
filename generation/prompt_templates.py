SYSTEM_PROMPT = (
    "You are an expert industrial maintenance engineer "
    "with 20 years of hands-on experience. "
    "You help field technicians diagnose machine faults "
    "and perform repairs using official maintenance manuals. "
    "Always be precise, safety-conscious, and cite your sources. "
    "Never guess or make up information not found in the manuals."
)


def build_rag_prompt(query: str, retrieved_chunks: list) -> str:
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

    context = ("\n\n" + "-" * 56 + "\n\n").join(context_blocks)

    return f"""{SYSTEM_PROMPT}

{"=" * 56}
RETRIEVED MANUAL SECTIONS:
{"=" * 56}

{context}

{"=" * 56}
TECHNICIAN FAULT DESCRIPTION:
{"=" * 56}
{query}

{"=" * 56}
INSTRUCTIONS:
{"=" * 56}
1. Diagnose the most likely fault based on the manual content.
2. Provide numbered step-by-step repair instructions.
3. Add safety warnings where needed — prefix with WARNING.
4. List required tools and parts if mentioned in manuals.
5. Cite sources using [SOURCE 1], [SOURCE 2] notation.
6. If manuals do not contain enough information, say so clearly.

{"=" * 56}
REQUIRED OUTPUT FORMAT:
{"=" * 56}

## Fault Diagnosis
[Your diagnosis]

## Step-by-Step Repair Instructions
1. [Step]
2. [Step]

## Safety Warnings
[Warnings from manuals]

## Required Tools and Parts
[Tools and parts mentioned]

## Sources Used
[SOURCE 1] - Manual name | Page | Section
"""


def build_no_context_response() -> str:
    return (
        "No relevant information was found in the available "
        "maintenance manuals for this fault description.\n\n"
        "Please try:\n"
        "1. Rephrasing with specific fault codes or symptoms\n"
        "2. Checking if the relevant manual has been uploaded\n"
        "3. Using more specific technical terms"
    )