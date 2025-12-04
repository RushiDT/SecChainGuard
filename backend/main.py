from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from .llm import LLMWrapper
from .rag import SimpleRAG

app = FastAPI(
    title="SecChainGuard API",
    description="IoT Security + Blockchain Access Control + ML Anomaly Detection Assistant",
)

# Initialize global objects
llm = LLMWrapper()
rag = SimpleRAG()


class AnalyzeRequest(BaseModel):
    query: str
    system_description: str


class AnalyzeResponse(BaseModel):
    answer: str
    retrieved_contexts: List[str]


@app.get("/")
def root():
    return {"message": "SecChainGuard backend is running."}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    # 1. Retrieve background context
    contexts = rag.get_relevant(request.query)
    context_text = "\n\n---\n\n".join(contexts)

    # 2. Build prompt
    prompt = f"""
You are an IoT Security, Blockchain Access Control, and ML Anomaly Detection expert.

Use:
- The system description
- The user question
- The retrieved context (as background knowledge only; do NOT copy sentences)

to write a detailed, scenario-specific analysis.

System Description:
{request.system_description}

User Question:
{request.query}

Retrieved Context (background knowledge only):
{context_text}

You MUST answer in EXACTLY these 4 numbered sections with these headings:

1. Threat Analysis (STRIDE)
2. Blockchain Access Control Design
3. ML Anomaly Detection Role
4. Recommended Mitigations

Formatting rules:
- Each section should have 3–5 bullet points.
- Use concise sentences (1–2 lines per bullet).
- Refer explicitly to the devices, actors, and network from the system description.
- Keep the overall answer around 250–350 words.

Answer:
"""

    answer = llm.generate(prompt, max_new_tokens=600)

    return AnalyzeResponse(answer=answer, retrieved_contexts=contexts)
