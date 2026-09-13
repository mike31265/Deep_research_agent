import os
import re
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


def _keywords(text, min_len=4):
    """Extract lowercase significant words from text for a crude
    relevance check."""
    return {
        w for w in re.findall(r"[a-zA-Z]+", text.lower())
        if len(w) >= min_len
    }


def filter_relevant_hits(query, hits, min_overlap=1):
    """
    Drop hits that share no meaningful keyword overlap with the query.

    This exists because ddgs (DuckDuckGo search) can get rate-limited
    or soft-blocked on hosted IPs (Render, Railway, etc.) and, instead
    of failing loudly, sometimes returns unrelated cached/fallback
    results. Those results look "successful" (no exception) but are
    completely off-topic, and would otherwise get attached to the
    final report as bogus sources.
    """

    query_words = _keywords(query)

    if not query_words:
        return hits

    relevant = []

    for h in hits:
        combined = f"{h.get('title', '')} {h.get('body', '')}"
        hit_words = _keywords(combined)

        if len(query_words & hit_words) >= min_overlap:
            relevant.append(h)
        else:
            print(
                f"    [filtered irrelevant hit] "
                f"{h.get('title', '')} | {h.get('href', '')}"
            )

    return relevant


# ---------------------------------------------------------
# Web Search
# ---------------------------------------------------------

def web_search(query, max_results=3):
    """Search the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
            return filter_relevant_hits(query, hits)
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
    """
    Researches each sub-question and keeps a single, shared source
    registry across ALL sub-questions. This means citation numbers
    like [1], [2] stay consistent everywhere -- in each sub-answer
    AND in the final synthesized report -- instead of each
    sub-question restarting its own [1], [2], ...
    """

    results = []

    # Shared across every sub-question:
    # url -> global source index (1-based)
    url_to_index = {}
    # ordered list of {"index": n, "title": ..., "uri": ...}
    source_registry = []

    def register_sources(hits):
        """Assign/reuse a global index for each hit, return them
        annotated with their global index."""
        annotated = []

        for h in hits:
            uri = h.get("href", "")
            title = h.get("title", "") or uri or "Source"

            if not uri:
                continue

            if uri not in url_to_index:
                idx = len(source_registry) + 1
                url_to_index[uri] = idx
                source_registry.append(
                    {
                        "index": idx,
                        "title": title,
                        "uri": uri
                    }
                )
            else:
                idx = url_to_index[uri]

            annotated.append(
                {
                    "index": idx,
                    "title": title,
                    "uri": uri,
                    "body": h.get("body", "")
                }
            )

        return annotated

    for i, q in enumerate(subquestions, 1):

        print(
            f"[{i}/{len(subquestions)}] Researching: {q}"
        )

        hits = web_search(
            q,
            max_results=results_per_query
        )

        # DEBUG: log raw hits so you can see if ddgs is returning
        # results unrelated to the sub-question itself.
        for h in hits:
            print(
                f"    hit -> {h.get('title', '')} | {h.get('href', '')}"
            )

        annotated_hits = register_sources(hits)

        if annotated_hits:

            context = "\n\n".join(
                f"Source [{h['index']}]: {h['title']}\n"
                f"URL: {h['uri']}\n"
                f"Snippet: {h['body'][:700]}"
                for h in annotated_hits
            )

            prompt = f"""
Answer the question using ONLY the search results.

Question:
{q}

Search results:
{context}

Give a short factual answer. Cite sources using their
exact bracket number from the search results above,
e.g. [{annotated_hits[0]['index']}].
Do not renumber the sources.
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
                "title": h["title"],
                "uri": h["uri"],
                "index": h["index"]
            }
            for h in annotated_hits
        ]

        results.append(
            {
                "question": q,
                "answer": answer,
                "sources": sources
            }
        )

    return results, source_registry


# ---------------------------------------------------------
# Step 3 - Final Report
# ---------------------------------------------------------

def synthesize_report(topic, results, source_registry):

    findings_block = "\n\n".join(
        f"Question: {r['question']}\n"
        f"Answer: {r['answer']}"
        for r in results
    )

    sources_block = "\n".join(
        f"[{s['index']}] {s['title']} - {s['uri']}"
        for s in source_registry
    )

    prompt = f"""
Create a clear research report about:

{topic}

Use ONLY the findings below.

{findings_block}

Available sources (for reference only - use their
existing bracket numbers, do not renumber them):
{sources_block}

Include:
- Short summary
- Main findings
- Key takeaways

Keep any existing [n] citations from the findings intact
so they still point to the same source. Do not invent new
citations. Keep the report concise and use Markdown.
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
    results, source_registry = research_subquestions(
        subquestions,
        results_per_query=3
    )

    print("Step 3/3 - Synthesizing...")
    report = synthesize_report(
        topic,
        results,
        source_registry
    )

    return report, results, source_registry


# ---------------------------------------------------------
# Gradio Web Interface
# ---------------------------------------------------------

def answer_question(question, history):

    if not question or not question.strip():
        return "Please enter a research question."

    try:

        report, results, source_registry = deep_research(
            question
        )

        if source_registry:

            source_lines = [
                f"{s['index']}. [{s['title']}]({s['uri']})"
                for s in source_registry
            ]

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
