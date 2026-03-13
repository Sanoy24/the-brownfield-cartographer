"""
Semanticist Agent — LLM-Powered Purpose Analyst.

Uses LLMs to generate semantic understanding of code that static analysis
cannot provide. Supports multiple LLM providers (OpenAI, Gemini, Ollama,
OpenRouter) configured via environment variables.

Core capabilities:
  - Purpose statement generation grounded in actual code (not docstrings)
  - Documentation drift detection (docstring vs. implementation mismatch)
  - Domain clustering via TF-IDF + k-means
  - Five FDE Day-One Question answering with evidence citations
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.graph.knowledge_graph import KnowledgeGraph
from src.models.schemas import ModuleNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Provider Abstraction
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """
    Dynamic LLM configuration resolved from environment variables.

    Supports the following providers (checked in order):
      - OPENAI_API_KEY     → OpenAI (gpt-4o-mini default)
      - GEMINI_API_KEY     → Google Gemini (gemini-2.0-flash default)
      - OPENROUTER_API_KEY → OpenRouter (any model via openrouter.ai)
      - OLLAMA_BASE_URL    → Ollama local (llama3 default)

    Override the model name with LLM_MODEL env var.
    """

    provider: str = ""
    api_key: str = ""
    model: str = ""
    base_url: str = ""

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Resolve the best available LLM provider from environment."""
        model_override = os.getenv("LLM_MODEL", "")

        if api_key := os.getenv("OPENAI_API_KEY", ""):
            return cls(
                provider="openai",
                api_key=api_key,
                model=model_override or "gpt-4o-mini",
            )
        if api_key := os.getenv("GEMINI_API_KEY", ""):
            return cls(
                provider="gemini",
                api_key=api_key,
                model=model_override or "gemini-2.0-flash",
            )
        if api_key := os.getenv("OPENROUTER_API_KEY", ""):
            return cls(
                provider="openrouter",
                api_key=api_key,
                model=model_override or "google/gemini-2.0-flash-exp:free",
                base_url="https://openrouter.ai/api/v1",
            )
        if base_url := os.getenv("OLLAMA_BASE_URL", ""):
            return cls(
                provider="ollama",
                base_url=base_url,
                model=model_override or "llama3",
            )

        return cls()  # No provider available


def _create_chat_model(config: LLMConfig) -> Any:
    """
    Create a LangChain chat model from the resolved LLM config.

    Returns None if no provider is configured or dependencies are missing.
    """
    if not config.provider:
        logger.warning(
            "No LLM provider configured. Set one of: OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, OLLAMA_BASE_URL"
        )
        return None

    try:
        if config.provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model, api_key=config.api_key, temperature=0.2
            )

        if config.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=config.model,
                google_api_key=config.api_key,
                temperature=0.2,
            )

        if config.provider == "openrouter":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model,
                api_key=config.api_key,
                base_url=config.base_url,
                temperature=0.2,
            )

        if config.provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=config.model,
                base_url=config.base_url,
                temperature=0.2,
            )

    except ImportError as exc:
        logger.warning("Missing LLM dependency for %s: %s", config.provider, exc)
    except Exception as exc:
        logger.warning("Failed to initialize %s LLM: %s", config.provider, exc)

    return None


# ---------------------------------------------------------------------------
# Context Window Budget
# ---------------------------------------------------------------------------


@dataclass
class ContextWindowBudget:
    """
    Track cumulative token usage across LLM calls.

    Uses a simple heuristic (chars / 4) to estimate tokens without
    requiring a tokenizer dependency. Logs warnings at 80% usage.
    """

    max_tokens: int = 500_000
    used_tokens: int = 0
    call_count: int = 0
    _warned: bool = field(default=False, repr=False)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length (≈4 chars per token)."""
        return max(1, len(text) // 4)

    def consume(self, prompt: str, response: str) -> None:
        """Record token usage for a prompt-response pair."""
        self.used_tokens += self.estimate_tokens(prompt) + self.estimate_tokens(
            response
        )
        self.call_count += 1

        if not self._warned and self.used_tokens > self.max_tokens * 0.8:
            logger.warning(
                "Token budget at %.0f%% (%d/%d). Consider reducing scope.",
                (self.used_tokens / self.max_tokens) * 100,
                self.used_tokens,
                self.max_tokens,
            )
            self._warned = True

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def exhausted(self) -> bool:
        return self.used_tokens >= self.max_tokens


# ---------------------------------------------------------------------------
# Core Analysis Functions
# ---------------------------------------------------------------------------


def _invoke_llm(llm: Any, prompt: str, budget: ContextWindowBudget) -> str:
    """Invoke the LLM with budget tracking. Returns empty string on failure."""
    if budget.exhausted:
        logger.debug("Token budget exhausted — skipping LLM call")
        return ""
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        budget.consume(prompt, text)
        return text.strip()
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return ""


def generate_purpose_statement(
    llm: Any,
    module: ModuleNode,
    source_code: str,
    budget: ContextWindowBudget,
) -> str:
    """
    Generate a Purpose Statement for a module using LLM analysis.

    Prompts with the actual code (not docstring) and asks for a 2–3
    sentence explanation of WHAT the module does (business function),
    not HOW it does it (implementation detail).
    """
    # Truncate large files to fit context window
    code_snippet = source_code[:8000] if len(source_code) > 8000 else source_code

    prompt = f"""Analyze this source file and write a 2-3 sentence Purpose Statement.
Explain WHAT this module does (its business function), NOT HOW it works.
Do not repeat the filename. Be specific about the domain and role.

File: {module.path}
Language: {module.language.value}
Public functions: {', '.join(module.public_functions) or 'None'}
Classes: {', '.join(module.classes) or 'None'}

Source code:
```
{code_snippet}
```

Purpose Statement:"""

    return _invoke_llm(llm, prompt, budget)


def detect_doc_drift(
    llm: Any,
    module: ModuleNode,
    source_code: str,
    purpose_statement: str,
    budget: ContextWindowBudget,
) -> list[str]:
    """
    Compare existing docstrings against the LLM-generated purpose.

    Returns a list of drift descriptions if the docstring contradicts
    or significantly diverges from the actual implementation.
    """
    # Extract the module-level docstring
    docstring_match = re.match(
        r'^(?:\s*#[^\n]*\n)*\s*(?:\'\'\'|""")(.+?)(?:\'\'\'|""")',
        source_code,
        re.DOTALL,
    )
    if not docstring_match:
        return []

    existing_docstring = docstring_match.group(1).strip()
    if len(existing_docstring) < 20:
        return []

    prompt = f"""Compare this module's existing docstring against its actual purpose (determined by code analysis).
If the docstring contradicts, misrepresents, or significantly diverges from what the code actually does, describe each discrepancy in one sentence.
If they are consistent, respond with exactly: "CONSISTENT"

Module: {module.path}

Existing docstring:
\"{existing_docstring}\"

Actual purpose (from code analysis):
\"{purpose_statement}\"

Discrepancies (one per line, or "CONSISTENT"):"""

    response = _invoke_llm(llm, prompt, budget)
    if not response or "CONSISTENT" in response.upper():
        return []

    return [
        line.strip()
        for line in response.split("\n")
        if line.strip() and not line.strip().startswith("-")
    ]


def cluster_into_domains(
    purpose_statements: dict[str, str],
    n_clusters: int = 6,
) -> dict[str, str]:
    """
    Cluster modules into inferred business domains using TF-IDF + k-means.

    Groups modules by semantic similarity of their purpose statements,
    then labels each cluster by its most representative terms.

    Args:
        purpose_statements: Mapping of module path → purpose statement text.
        n_clusters: Number of clusters (default: 6).

    Returns:
        Mapping of module path → inferred domain label.
    """
    if len(purpose_statements) < 3:
        return {path: "core" for path in purpose_statements}

    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        logger.warning("scikit-learn not installed — skipping domain clustering")
        return {path: "unclustered" for path in purpose_statements}

    paths = list(purpose_statements.keys())
    texts = [purpose_statements[p] for p in paths]

    # Adjust cluster count to data size
    k = min(n_clusters, len(texts))

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)

    # Label each cluster by its top terms
    cluster_labels: dict[int, str] = {}
    for i in range(k):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-3:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices]
        cluster_labels[i] = "_".join(top_terms[:2])

    return {paths[idx]: cluster_labels[label] for idx, label in enumerate(labels)}


def answer_day_one_questions(
    llm: Any,
    surveyor_results: dict[str, Any],
    hydrologist_results: dict[str, Any],
    purpose_statements: dict[str, str],
    budget: ContextWindowBudget,
) -> dict[str, str]:
    """
    Synthesize Surveyor + Hydrologist output to answer the Five FDE Day-One Questions.

    Returns a dict with keys q1..q5 mapping to evidence-backed answers.
    """
    context = json.dumps(
        {
            "surveyor": surveyor_results,
            "hydrologist": hydrologist_results,
            "module_purposes": dict(list(purpose_statements.items())[:30]),  # Top 30
        },
        indent=2,
        default=str,
    )

    # Truncate context if too large
    if len(context) > 12000:
        context = context[:12000] + "\n... (truncated)"

    prompt = f"""You are an expert Forward-Deployed Engineer analyzing a codebase.
Based on the analysis results below, answer these 5 Day-One Questions.
For each answer, cite specific file paths and line numbers as evidence.

Analysis Results:
{context}

Answer each question in 2-4 sentences with file:line evidence:

Q1: What is the primary data ingestion path?
Q2: What are the 3-5 most critical output datasets/endpoints?
Q3: What is the blast radius if the most critical module fails?
Q4: Where is the business logic concentrated vs. distributed?
Q5: What has changed most frequently in the last 90 days (git velocity)?

Format as JSON with keys "q1" through "q5":"""

    response = _invoke_llm(llm, prompt, budget)
    if not response:
        return {
            f"q{i}": "LLM unavailable — run with an API key configured"
            for i in range(1, 6)
        }

    try:
        # Try to parse JSON from the response
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: return raw text split by Q markers
    return {"q1": response, "q2": "", "q3": "", "q4": "", "q5": ""}


# ---------------------------------------------------------------------------
# Semanticist Agent — Public API
# ---------------------------------------------------------------------------


class Semanticist:
    """
    The Semanticist agent adds LLM-powered semantic understanding.

    Generates purpose statements, detects documentation drift,
    clusters modules into business domains, and answers the Five
    FDE Day-One Questions with evidence citations.

    Requires an LLM API key set via environment variables.
    Gracefully degrades if no LLM is available.
    """

    def __init__(self) -> None:
        self.llm_config = LLMConfig.from_env()
        self.llm = _create_chat_model(self.llm_config)
        self.budget = ContextWindowBudget()

    @property
    def available(self) -> bool:
        """Whether an LLM provider is configured and ready."""
        return self.llm is not None

    def run(
        self,
        repo_root: Path,
        knowledge_graph: KnowledgeGraph,
        surveyor_results: dict[str, Any] | None = None,
        hydrologist_results: dict[str, Any] | None = None,
        changed_modules: set[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """
        Execute semantic analysis on the codebase.

        Steps:
          1. Generate purpose statements for modules (optionally only changed ones)
          2. Detect documentation drift
          3. Cluster modules into business domains
          4. Answer the Five FDE Day-One Questions

        On large repos, only a prioritized subset of modules is analyzed
        to keep runtime bounded.
        """
        logger.info(
            "Semanticist: analyzing %s (LLM: %s)%s",
            repo_root,
            self.llm_config.provider or "none",
            " [incremental]" if changed_modules else "",
        )

        purpose_statements: dict[str, str] = {}
        doc_drift_flags: dict[str, list[str]] = {}
        domain_clusters: dict[str, str] = {}
        day_one_answers: dict[str, str] = {}

        if not self.available:
            logger.warning(
                "No LLM configured — Semanticist will produce minimal output"
            )
            return {
                "purpose_statements": purpose_statements,
                "doc_drift_flags": doc_drift_flags,
                "domain_clusters": domain_clusters,
                "day_one_answers": {f"q{i}": "LLM not configured" for i in range(1, 6)},
                "llm_provider": "none",
                "token_budget": {"used": 0, "max": self.budget.max_tokens},
            }

        module_nodes = knowledge_graph._module_nodes

        # Performance controls for large repos
        max_modules = int(os.getenv("SEMANTICIST_MAX_MODULES", "60"))
        max_seconds = int(os.getenv("SEMANTICIST_MAX_SECONDS", "900"))
        max_consecutive_failures = int(
            os.getenv("SEMANTICIST_MAX_CONSECUTIVE_FAILURES", "5")
        )
        start = time.monotonic()

        def _prioritized_paths() -> list[str]:
            # Incremental: only changed modules
            if changed_modules is not None:
                return sorted([p for p in changed_modules if p in module_nodes])

            paths = list(module_nodes.keys())
            if len(paths) <= max_modules:
                return sorted(paths)

            priority: list[str] = []
            seen: set[str] = set()

            # Prefer critical hubs and high-velocity files if available
            if surveyor_results:
                for hub in surveyor_results.get("top_architectural_hubs", [])[
                    :max_modules
                ]:
                    p = hub.get("path")
                    if p and p in module_nodes and p not in seen:
                        priority.append(p)
                        seen.add(p)
                for hv in surveyor_results.get("high_velocity_files", [])[
                    :max_modules
                ]:
                    p = hv.get("path")
                    if p and p in module_nodes and p not in seen:
                        priority.append(p)
                        seen.add(p)

            # Fill remaining slots deterministically
            for p in sorted(paths):
                if p not in seen:
                    priority.append(p)
                    seen.add(p)
                if len(priority) >= max_modules:
                    break

            return priority[:max_modules]

        prioritized_paths = _prioritized_paths()

        # In incremental mode, keep existing purposes for unchanged modules
        if changed_modules is not None:
            for path in module_nodes:
                if path in changed_modules:
                    continue
                if knowledge_graph.graph.has_node(path):
                    existing = knowledge_graph.graph.nodes[path].get(
                        "purpose_statement"
                    )
                    if existing:
                        purpose_statements[path] = existing

        modules_to_process = {
            p: module_nodes[p] for p in prioritized_paths if p in module_nodes
        }
        if changed_modules is None and len(module_nodes) > max_modules:
            logger.info(
                "Semanticist: limiting purpose generation to %d/%d modules "
                "(set SEMANTICIST_MAX_MODULES to override)",
                len(modules_to_process),
                len(module_nodes),
            )

        consecutive_failures = 0
        # Step 1 & 2: Purpose statements + doc drift
        for path, module in modules_to_process.items():
            if time.monotonic() - start > max_seconds:
                logger.warning(
                    "Semanticist time budget exceeded (%ds) — stopping early after %d modules",
                    max_seconds,
                    len(
                        [
                            p
                            for p in purpose_statements
                            if p in modules_to_process.keys()
                        ]
                    ),
                )
                break
            if self.budget.exhausted:
                logger.warning("Token budget exhausted at module %s", path)
                break

            source_path = repo_root / path
            if not source_path.exists():
                continue

            try:
                source_code = source_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            purpose = generate_purpose_statement(
                self.llm, module, source_code, self.budget
            )
            if purpose:
                consecutive_failures = 0
                purpose_statements[path] = purpose
                module.purpose_statement = purpose
                if knowledge_graph.graph.has_node(path):
                    knowledge_graph.graph.nodes[path]["purpose_statement"] = purpose

                drift = detect_doc_drift(
                    self.llm, module, source_code, purpose, self.budget
                )
                if drift:
                    doc_drift_flags[path] = drift
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(
                        "Semanticist encountered %d consecutive LLM failures — stopping early",
                        consecutive_failures,
                    )
                    break

        logger.info(
            "Generated %d purpose statements, %d drift flags",
            len(purpose_statements),
            len(doc_drift_flags),
        )

        # Step 3: Domain clustering
        if purpose_statements:
            domain_clusters = cluster_into_domains(purpose_statements)
            for path, domain in domain_clusters.items():
                if path in module_nodes:
                    module_nodes[path].domain_cluster = domain
                    if path in knowledge_graph.graph:
                        knowledge_graph.graph.nodes[path]["domain_cluster"] = domain

            logger.info("Clustered %d modules into domains", len(domain_clusters))

        # Step 4: Day-One Questions
        day_one_answers = answer_day_one_questions(
            self.llm,
            surveyor_results or {},
            hydrologist_results or {},
            purpose_statements,
            self.budget,
        )

        logger.info(
            "Semanticist complete: %d tokens used (%d calls)",
            self.budget.used_tokens,
            self.budget.call_count,
        )

        return {
            "purpose_statements": purpose_statements,
            "doc_drift_flags": doc_drift_flags,
            "domain_clusters": domain_clusters,
            "day_one_answers": day_one_answers,
            "llm_provider": self.llm_config.provider,
            "token_budget": {
                "used": self.budget.used_tokens,
                "max": self.budget.max_tokens,
                "calls": self.budget.call_count,
            },
        }
