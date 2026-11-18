"""Offline integration and failure-path tests; no API credits or internet needed."""

from __future__ import annotations

import json
import os
import re
import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

# Tests stay outside the exact five-file submission layout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lumina_rescue"))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from loop_logic import RoutingState, should_continue
from state_graph_config import (
    CHUNK_SIZE, DEFAULT_CORPUS, PROJECT_ROOT, RECURSION_LIMIT, STATIC_HEADERS,
    AgentBackend, AgentState, CritiqueResult, DemoBackend, DuckDuckGoSearchArgs,
    LiveBackend, SearchFunction, SearchResponse, build_agentic_rag_graph,
    chunk_medical_corpus, chunk_text, create_live_model, demo_search, enrich_metadata, initial_state,
    make_search_tool, medical_web_search, retrieve_top_k_chunks,
)


def run_graph(backend: AgentBackend, search: SearchFunction = demo_search,
              state: AgentState | None = None) -> AgentState:
    graph = build_agentic_rag_graph(backend, search=search)
    return cast(AgentState, graph.invoke(
        state or initial_state("CBT depression fMRI"), {"recursion_limit": RECURSION_LIMIT}
    ))


class VariantBackend(DemoBackend):
    """Exercise adverse model outputs against the real graph nodes."""

    def __init__(self, variant: str) -> None:
        super().__init__("exhausted")
        self.variant = variant
        self.plans = 0
        self.reviews = 0

    def plan(self, state: AgentState) -> AIMessage:
        self.plans += 1
        if self.variant == "planner_error":
            raise TimeoutError("simulated provider timeout")
        if self.variant == "no_tools":
            return AIMessage(content="Unverified draft; no tool call.")
        response = super().plan(state)
        if self.variant == "multiple_tools":
            response.tool_calls.append(ToolCall(name="medical_web_search", args={"query": "another query"},
                                               id=f"extra-{self.plans}", type="tool_call"))
        elif self.variant == "unknown_tool":
            response.tool_calls[0]["name"] = "unknown_tool"
        elif self.variant == "bad_args":
            response.tool_calls[0]["args"] = {"query": " "}
        return response

    def critique(self, state: AgentState) -> CritiqueResult:
        self.reviews += 1
        if self.variant == "critique_error":
            raise ValidationError.from_exception_data("CritiqueResult", [])
        result = super().critique(state)
        if self.variant in {"invented_citation", "unmarked_citation", "empty_citation", "success"}:
            result.sufficient = True
            result.missing_information = []
        if self.variant == "invented_citation":
            result.cited_source_ids = ["not-a-real-source"]
            result.answer = "Unverified statement [not-a-real-source]"
        elif self.variant == "unmarked_citation":
            result.answer = "No citations appear in this answer."
        elif self.variant == "empty_citation":
            result.cited_source_ids = []
        return result


class IngestionTests(unittest.TestCase):
    def test_real_corpus_is_lossless_and_all_chunks_are_500_characters(self) -> None:
        raw = DEFAULT_CORPUS.read_text(encoding="utf-8")
        documents: list[dict[str, str]] = json.loads(raw)
        chunks = enrich_metadata(chunk_medical_corpus(raw))
        self.assertEqual(len(documents), 50)
        self.assertEqual(len(chunks), 100)
        for document in documents:
            with self.subTest(source=document["id"]):
                source_chunks = [chunk for chunk in chunks if chunk["source_id"] == document["id"]]
                self.assertEqual("".join(chunk_text(chunk) for chunk in source_chunks), document["abstract_text"])
                for chunk in source_chunks:
                    self.assertEqual(len(chunk["content"]), CHUNK_SIZE)
                    self.assertIsNotNone(datetime.fromisoformat(chunk["processed_timestamp"]).tzinfo)
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))

    def test_padding_preserves_real_trailing_spaces_and_boundary_lengths(self) -> None:
        for size in [1, 499, 500, 501, 1000, 1001]:
            with self.subTest(size=size):
                original = "A" * (size - 1) + " "
                raw = json.dumps([{"id": "test", "title": "Boundary", "abstract_text": original,
                                   "authors": [], "publication_date": "2023-01-01"}])
                chunks = enrich_metadata(chunk_medical_corpus(raw))
                self.assertEqual("".join(chunk_text(chunk) for chunk in chunks), original)
                self.assertEqual(len(chunks), (size + 499) // 500)

    def test_invalid_json_records_and_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            chunk_medical_corpus("not json")
        with self.assertRaises(ValidationError):
            chunk_medical_corpus('[{"id":"missing-fields"}]')
        document = {"id": "same", "title": "x", "abstract_text": "x", "authors": [], "publication_date": "2023"}
        with self.assertRaises(ValueError):
            chunk_medical_corpus(json.dumps([document, document]))

    def test_retrieval_ranks_by_query_instead_of_returning_first_records(self) -> None:
        chunks = enrich_metadata(chunk_medical_corpus(DEFAULT_CORPUS.read_text(encoding="utf-8")))
        top = retrieve_top_k_chunks("CBT depression neuroplasticity fMRI", chunks, k=2)
        self.assertTrue(top)
        self.assertEqual(top[0]["source_id"], "LNAI-MED-002")
        self.assertLessEqual(len(top), 2)
        self.assertEqual(retrieve_top_k_chunks("xyznotinthecorpus", chunks), [])
        with self.assertRaises(ValueError):
            retrieve_top_k_chunks("CBT", chunks, 0)
        with self.assertRaises(ValueError):
            retrieve_top_k_chunks("  ", chunks)

    def test_metadata_enrichment_precedes_every_retrieval(self) -> None:
        with patch("state_graph_config.retrieve_top_k_chunks", wraps=retrieve_top_k_chunks) as retrieval:
            run_graph(DemoBackend())
        for call in retrieval.call_args_list:
            enriched = call.args[1]
            self.assertTrue(enriched)
            self.assertTrue(all(chunk["source_id"] and chunk["processed_timestamp"] for chunk in enriched))


class SearchToolTests(unittest.TestCase):
    def test_schema_rejects_blank_queries_extra_fields_and_unbounded_counts(self) -> None:
        for payload in [{"query": " "}, {"query": "CBT", "max_results": 6},
                        {"query": "CBT", "max_results": "3"}, {"query": "CBT", "extra": True}]:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                DuckDuckGoSearchArgs.model_validate(payload)
        self.assertEqual(DuckDuckGoSearchArgs(query="  CBT depression  ").query, "CBT depression")
        self.assertIs(make_search_tool().args_schema, DuckDuckGoSearchArgs)

    def test_headers_timeout_urls_and_result_cap_are_applied_to_real_adapter(self) -> None:
        with patch("state_graph_config.Duckduckgo") as constructor, patch("state_graph_config.httpx.get") as get:
            engine = constructor.return_value
            get.return_value = SimpleNamespace(status_code=200, text="<html>fixture</html>")
            engine.post_extract_results.return_value = [
                SimpleNamespace(title="Study", href=f"https://example.test/{i}", body="Snippet")
                for i in range(8)
            ]
            result = SearchResponse.model_validate_json(medical_web_search("CBT depression", 2))
            constructor.assert_called_once_with(timeout=10, verify=True)
            get.assert_called_once()
            self.assertEqual(get.call_args.kwargs["headers"], STATIC_HEADERS)
            self.assertEqual(get.call_args.kwargs["params"], {"q": "CBT depression"})
            self.assertEqual(get.call_args.kwargs["timeout"], 10)
            self.assertTrue(get.call_args.kwargs["follow_redirects"])
            engine.http_client.get.assert_not_called()
            self.assertEqual(len(result.results), 2)
            self.assertFalse(result.error)

    def test_network_failure_and_empty_search_are_observations(self) -> None:
        with patch("state_graph_config.httpx.get", side_effect=TimeoutError):
            result = SearchResponse.model_validate_json(medical_web_search("CBT depression"))
            self.assertFalse(result.results)
            self.assertIn("TimeoutError", result.error)
        with patch("state_graph_config.Duckduckgo") as constructor, patch("state_graph_config.httpx.get") as get:
            get.return_value = SimpleNamespace(status_code=200, text="<html></html>")
            constructor.return_value.post_extract_results.return_value = []
            result = SearchResponse.model_validate_json(medical_web_search("CBT depression"))
            self.assertFalse(result.results)
            self.assertIn("HTTP 200", result.error)

    def test_http_error_is_distinguished_from_empty_results(self) -> None:
        for status in [202, 403, 429, 503]:
            with self.subTest(status=status), patch("state_graph_config.Duckduckgo") as constructor, patch("state_graph_config.httpx.get") as get:
                get.return_value = SimpleNamespace(status_code=status, text="Unavailable")
                result = SearchResponse.model_validate_json(medical_web_search("CBT depression"))
                self.assertFalse(result.results)
                self.assertIn(f"HTTP {status}", result.error)
                constructor.assert_not_called()

    def test_verification_page_is_not_misreported_as_a_success_or_no_matches(self) -> None:
        with patch("state_graph_config.Duckduckgo") as constructor, patch("state_graph_config.httpx.get") as get:
            get.return_value = SimpleNamespace(
                status_code=202, text='<form id="challenge-form">Unfortunately, bots use DuckDuckGo too.</form>',
            )
            result = SearchResponse.model_validate_json(medical_web_search("CBT depression"))
            self.assertFalse(result.results)
            self.assertIn("bot-verification", result.error)
            constructor.assert_not_called()

    def test_real_ddgs_parser_preserves_a_source_url_from_a_results_page(self) -> None:
        html = ('<html><div class="result__body"><h2>Study title</h2>'
                '<a class="result__snippet" href="https://pubmed.ncbi.nlm.nih.gov/123/">'
                'Research snippet</a></div></html>')
        with patch("state_graph_config.httpx.get", return_value=SimpleNamespace(status_code=200, text=html)):
            result = SearchResponse.model_validate_json(medical_web_search("CBT depression"))
        self.assertFalse(result.error)
        self.assertEqual(result.results[0]["url"], "https://pubmed.ncbi.nlm.nih.gov/123/")
        self.assertEqual(result.results[0]["snippet"], "Research snippet")

    def test_source_redirect_links_are_resolved_and_unsafe_schemes_filtered(self) -> None:
        html = ('<html><div class="result__body"><h2>Source</h2>'
                '<a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpubmed.ncbi.nlm.nih.gov%2F123%2F">'
                'Research snippet</a></div><div class="result__body"><h2>Invalid</h2>'
                '<a href="javascript:void(0)">Invalid snippet</a></div></html>')
        with patch("state_graph_config.httpx.get", return_value=SimpleNamespace(status_code=200, text=html)):
            result = SearchResponse.model_validate_json(medical_web_search("CBT depression"))
        self.assertFalse(result.error)
        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0]["url"], "https://pubmed.ncbi.nlm.nih.gov/123/")

    def test_live_backend_uses_bound_tool_calls_and_validated_review(self) -> None:
        model = MagicMock(spec=BaseChatModel)
        planner_response = AIMessage(content="", tool_calls=[{
            "name": "medical_web_search", "args": {"query": "CBT depression"},
            "id": "real-protocol-test", "type": "tool_call",
        }])
        review = CritiqueResult(sufficient=False, missing_information=["Follow-up"],
                               refined_query="CBT follow-up", answer="Insufficient.", cited_source_ids=[])
        model.bind_tools.return_value.invoke.return_value = planner_response
        model.with_structured_output.return_value.invoke.return_value = review
        backend = LiveBackend(cast(BaseChatModel, model), make_search_tool())
        self.assertEqual(backend.plan(initial_state("CBT depression")), planner_response)
        self.assertEqual(backend.critique(initial_state("CBT depression")), review)
        model.bind_tools.assert_called_once()
        self.assertFalse(model.bind_tools.call_args.kwargs["parallel_tool_calls"])


class ProviderConfigurationTests(unittest.TestCase):
    def test_auto_uses_module_two_gemini_configuration_with_bounded_requests(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "dummy-test-key", "GEMINI_MODEL": "test-model"}, clear=True):
            with patch("state_graph_config.ChatGoogleGenerativeAI") as constructor:
                create_live_model()
                constructor.assert_called_once()
                self.assertEqual(constructor.call_args.kwargs["model"], "test-model")
                self.assertFalse(constructor.call_args.kwargs["vertexai"])
                self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
                self.assertEqual(constructor.call_args.kwargs["timeout"], 30)

    def test_gemini_key_alias_is_supported(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-test-key", "GEMINI_MODEL": "test-model"}, clear=True):
            with patch("state_graph_config.ChatGoogleGenerativeAI") as constructor:
                create_live_model("gemini")
                constructor.assert_called_once()

    def test_explicit_provider_does_not_silently_switch(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "dummy-test-key", "GEMINI_MODEL": "test-model"}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                create_live_model("openai")

    def test_missing_model_key_or_invalid_provider_is_a_configuration_error(self) -> None:
        for values, provider in [({}, "gemini"), ({"GOOGLE_API_KEY": "dummy-test-key"}, "gemini"),
                                 ({"LLM_PROVIDER": "unknown"}, "auto")]:
            with self.subTest(provider=provider, fields=list(values)):
                with patch.dict(os.environ, values, clear=True), self.assertRaises(ValueError):
                    create_live_model(provider)

    def test_openai_remains_an_explicit_supported_alternative(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-test-key", "OPENAI_MODEL": "test-model"}, clear=True):
            with patch("state_graph_config.ChatOpenAI") as constructor:
                create_live_model("openai")
                constructor.assert_called_once()
                self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)

    def test_gemini_binding_does_not_receive_openai_specific_parallel_flag(self) -> None:
        model = MagicMock(spec=ChatGoogleGenerativeAI)
        model.bind_tools.return_value.invoke.return_value = AIMessage(content="Draft")
        model.with_structured_output.return_value.invoke.return_value = CritiqueResult(
            sufficient=False, missing_information=["Follow-up"], refined_query="CBT follow-up",
            answer="Insufficient.", cited_source_ids=[],
        )
        backend = LiveBackend(cast(BaseChatModel, model), make_search_tool())
        model.bind_tools.assert_called_once()
        self.assertNotIn("parallel_tool_calls", model.bind_tools.call_args.kwargs)
        backend.plan(initial_state("CBT depression"))
        backend.critique(initial_state("CBT depression"))
        for runnable in [model.bind_tools.return_value, model.with_structured_output.return_value]:
            self.assertEqual(runnable.invoke.call_args.kwargs["automatic_function_calling"], {"disable": True})


class GraphSafetyTests(unittest.TestCase):
    def test_success_after_refinement_keeps_message_history_and_tool_call_ids(self) -> None:
        seen_queries: list[str] = []

        def search(query: str, limit: int) -> str:
            seen_queries.append(query)
            return demo_search(query, limit)

        result = run_graph(DemoBackend(), search)
        self.assertEqual(result["loop_count"], 2)
        self.assertEqual(result["termination_reason"], "sufficient")
        self.assertNotEqual(seen_queries[0], seen_queries[1])
        self.assertIsInstance(result["messages"][0], HumanMessage)
        self.assertIsInstance(result["messages"][-1], AIMessage)
        calls = {call["id"] for message in result["messages"] if isinstance(message, AIMessage)
                 for call in message.tool_calls}
        observations = {message.tool_call_id for message in result["messages"] if isinstance(message, ToolMessage)}
        self.assertEqual(calls, observations)
        self.assertEqual(len(calls), 2)
        self.assertTrue(result["final_answer"])

    def test_persistent_insufficiency_performs_exactly_four_searches(self) -> None:
        search = MagicMock(side_effect=demo_search)
        result = run_graph(DemoBackend("exhausted"), cast(SearchFunction, search))
        self.assertEqual(result["loop_count"], 4)
        self.assertEqual(search.call_count, 4)
        self.assertEqual(result["termination_reason"], "retry_limit")
        self.assertIn("INSUFFICIENT EVIDENCE", result["final_answer"])
        self.assertEqual(len(result["web_results"]), 1)  # URL deduplication.

    def test_sufficient_first_attempt_ends_immediately(self) -> None:
        result = run_graph(VariantBackend("success"))
        self.assertEqual(result["loop_count"], 1)
        self.assertEqual(result["termination_reason"], "sufficient")

    def test_no_tool_calls_still_consume_attempt_budget(self) -> None:
        search = MagicMock(side_effect=demo_search)
        result = run_graph(VariantBackend("no_tools"), cast(SearchFunction, search))
        self.assertEqual(result["loop_count"], 4)
        search.assert_not_called()

    def test_multiple_tool_requests_execute_only_one_but_all_get_responses(self) -> None:
        search = MagicMock(side_effect=demo_search)
        result = run_graph(VariantBackend("multiple_tools"), cast(SearchFunction, search))
        self.assertEqual(search.call_count, 4)
        observations = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(len(observations), 8)
        self.assertEqual(len(result["tool_errors"]), 4)

    def test_unknown_tools_and_invalid_arguments_do_not_execute_search(self) -> None:
        for variant in ["unknown_tool", "bad_args"]:
            with self.subTest(variant=variant):
                search = MagicMock(side_effect=demo_search)
                result = run_graph(VariantBackend(variant), cast(SearchFunction, search))
                search.assert_not_called()
                self.assertEqual(result["loop_count"], 4)
                self.assertEqual(len(result["tool_errors"]), 4)

    def test_raised_tool_error_and_malformed_output_remain_bounded(self) -> None:
        for search in [MagicMock(side_effect=TimeoutError), MagicMock(return_value="not JSON")]:
            with self.subTest(search=search):
                result = run_graph(DemoBackend("exhausted"), cast(SearchFunction, search))
                self.assertEqual(result["loop_count"], 4)
                self.assertEqual(len(result["tool_errors"]), 4)
                self.assertEqual(search.call_count, 4)

    def test_invented_empty_or_unmarked_citations_cannot_claim_success(self) -> None:
        for variant in ["invented_citation", "unmarked_citation", "empty_citation"]:
            with self.subTest(variant=variant):
                result = run_graph(VariantBackend(variant))
                self.assertEqual(result["termination_reason"], "retry_limit")
                self.assertNotIn("Unverified statement", result["final_answer"])

    def test_model_and_critique_errors_stop_without_further_retry(self) -> None:
        for variant in ["planner_error", "critique_error"]:
            with self.subTest(variant=variant):
                backend = VariantBackend(variant)
                result = run_graph(backend)
                self.assertEqual(result["loop_count"], 1)
                self.assertEqual(result["termination_reason"], "fatal_error")
                self.assertEqual(backend.plans, 1)
                self.assertLessEqual(backend.reviews, 1)
                self.assertTrue(result["final_answer"])

    def test_resumed_exhausted_state_cannot_spend_another_model_call(self) -> None:
        backend = VariantBackend("success")
        state = initial_state("CBT depression")
        state["loop_count"] = 4
        result = run_graph(backend, state=state)
        self.assertEqual(backend.plans, 0)
        self.assertEqual(backend.reviews, 0)
        self.assertEqual(result["termination_reason"], "fatal_error")

    def test_message_reducer_appends_new_ids_and_updates_existing_ids(self) -> None:
        first = HumanMessage(content="question", id="q1")
        answer = AIMessage(content="draft", id="a1")
        merged = add_messages(first, answer)
        updated = add_messages(merged, AIMessage(content="revised", id="a1"))
        assert isinstance(updated, list)
        self.assertEqual(len(updated), 2)
        last = updated[-1]
        assert isinstance(last, BaseMessage)
        self.assertEqual(last.content, "revised")

    def test_graph_topology_has_no_cycle_that_bypasses_critique(self) -> None:
        graph = build_agentic_rag_graph(DemoBackend(), search=demo_search).get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}
        self.assertEqual(edges, {("__start__", "agent"), ("agent", "tools"),
                                 ("tools", "critique"), ("critique", "agent"),
                                 ("critique", "__end__")})

    def test_documented_mermaid_states_and_edges_match_the_compiled_graph(self) -> None:
        document = (PROJECT_ROOT / "architecture_proposal.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```mermaid\s*\n(.*?)```", document, re.DOTALL)
        self.assertEqual(len(blocks), 1)
        diagram = blocks[0]
        self.assertTrue(diagram.lstrip().startswith("stateDiagram-v2"))
        transitions = re.findall(r"^\s*(\[\*\]|\w+)\s*-->\s*(\[\*\]|\w+)", diagram, re.MULTILINE)
        documented = {(source if source != "[*]" else "__start__",
                       target if target != "[*]" else "__end__")
                      for source, target in transitions}
        graph = build_agentic_rag_graph(DemoBackend(), search=demo_search).get_graph()
        self.assertEqual(documented, {(edge.source, edge.target) for edge in graph.edges})
        self.assertIn("loop_count <= 3", diagram)
        self.assertIn("loop_count > 3", diagram)

    def test_routing_boundaries_and_success_override(self) -> None:
        for count, expected in [(1, "agent"), (3, "agent"), (4, "END")]:
            state: RoutingState = {"loop_count": count, "needs_more_info": True}
            self.assertEqual(should_continue(state), expected)
        for count in [1, 3, 4]:
            self.assertEqual(should_continue({"loop_count": count, "needs_more_info": False}), "END")


if __name__ == "__main__":
    unittest.main(verbosity=2)
