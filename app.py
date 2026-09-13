import os
import time
import gradio as gr
from ddgs import DDGS
from groq import Groq


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL = "openai/gpt-oss-120b"

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Add it as an Environment Variable in Render."
    )

client = Groq(api_key=api_key)


# ---------------------------------------------------------
# Web Search
# ---------------------------------------------------------

def web_search(query, max_results=3):
    """Search the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"[search warning] '{query}' failed: {e}")
        return []


# ---------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------

def groq_chat(prompt, retries=2, temperature=0.3):
    """Send a prompt to Groq with retry handling."""
    last_err = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_completion_tokens=1200
            )

            return response.choices[0].message.content

        except Exception as e:
            last_err = e

            # A 413 means the request is too large.
            # Retrying will not fix it.
            if "413" in str(e) or "Request too large" in str(e):
                raise RuntimeError(
                    "The request is too large for the Groq token limit. "
                    "Try a shorter question or reduce the research depth."
                )

            wait = 2 ** attempt
            print(
                f"[retry {attempt + 1}/{retries}] "
                f"error: {e} -> waiting {wait}s"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"groq_chat failed after {retries} attempts: {last_err}"
    )


# ---------------------------------------------------------
# Step 1 - Planning
# ---------------------------------------------------------

def plan_research(topic, n_subquestions=4):

    prompt = f"""
You are a research planner.

Break this research topic into {n_subquestions}
focused, non-overlapping sub-questions.

Topic:
"{topic}"

Return ONLY a numbered list of questions.
"""

    text = groq_chat(prompt, temperature=0.3)

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    questions = []

    for line in lines:
        cleaned = line.lstrip("0123456789.-) ").strip()

        if cleaned:
            questions.append(cleaned)

    return questions[:n_subquestions] if questions else [topic]


# ---------------------------------------------------------
# Step 2 - Research
# ---------------------------------------------------------

def research_subquestions(subquestions, results_per_query=3):

    results = []

    for i, q in enumerate(subquestions, 1):

        print(
            f"[{i}/{len(subquestions)}] Researching: {q}"
        )

        hits = web_search(
            q,
            max_results=results_per_query
        )

        if hits:

            context = "\n\n".join(
                f"Source {j + 1}: {h.get('title', '')}\n"
                f"URL: {h.get('href', '')}\n"
                f"Snippet: {h.get('body', '')[:700]}"
                for j, h in enumerate(hits)
            )

            prompt = f"""
Answer the question using ONLY the search results.

Question:
{q}

Search results:
{context}

Give a short factual answer with citations
like [1], [2].
"""

        else:

            prompt = f"""
Answer this question briefly from general knowledge.

Clearly say that the information is unverified.

Question:
{q}
"""

        answer = groq_chat(
            prompt,
            temperature=0.3
        )

        sources = [
            {
                "title": h.get("title", ""),
                "uri": h.get("href", "")
            }
            for h in hits
        ]

        results.append(
            {
                "question": q,
                "answer": answer,
                "sources": sources
            }
        )

    return results


# ---------------------------------------------------------
# Step 3 - Final Report
# ---------------------------------------------------------

def synthesize_report(topic, results):

    findings_block = "\n\n".join(
        f"Question: {r['question']}\n"
        f"Answer: {r['answer']}"
        for r in results
    )

    prompt = f"""
Create a clear research report about:

{topic}

Use ONLY the findings below.

{findings_block}

Include:
- Short summary
- Main findings
- Key takeaways

Keep the report concise and use Markdown.
"""

    return groq_chat(
        prompt,
        temperature=0.4
    )


# ---------------------------------------------------------
# Full Research Pipeline
# ---------------------------------------------------------

def deep_research(topic):

    print(
        f"Deep Research Agent starting on: {topic}"
    )

    print("Step 1/3 - Planning...")
    subquestions = plan_research(
        topic,
        n_subquestions=4
    )

    print("Step 2/3 - Researching...")
    results = research_subquestions(
        subquestions,
        results_per_query=3
    )

    print("Step 3/3 - Synthesizing...")
    report = synthesize_report(
        topic,
        results
    )

    return report, results


# ---------------------------------------------------------
# Gradio Web Interface
# ---------------------------------------------------------

def answer_question(question, history):

    if not question or not question.strip():
        return "Please enter a research question."

    try:

        report, results = deep_research(
            question
        )

        seen = set()
        source_lines = []

        for result in results:

            for source in result["sources"]:

                uri = source.get("uri", "")
                title = source.get(
                    "title",
                    "Source"
                )

                if uri and uri not in seen:

                    seen.add(uri)

                    source_lines.append(
                        f"- [{title}]({uri})"
                    )

        if source_lines:

            report += (
                "\n\n---\n"
                "## 🔗 Sources\n"
                + "\n".join(source_lines)
            )

        return report

    except Exception as e:

        return (
            "### ❌ Error\n\n"
            f"{e}"
        )


demo = gr.ChatInterface(
    fn=answer_question,
    title="🔎 Deep Research AI Agent",
    description=(
        "Ask a question and the agent will search "
        "the web and generate a cited answer."
    ),
    examples=[
        "What are the environmental impacts of AI?",
        "How will AI affect software engineering jobs?",
        "What are the latest AI trends?"
    ]
)


# ---------------------------------------------------------
# Start Server
# ---------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.getenv("PORT", "7860")
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
