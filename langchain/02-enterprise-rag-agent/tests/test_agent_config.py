"""Offline regressions for SQL authorization boundaries and mixed-source answers.

Run from the project folder: python -m unittest discover -s tests -v
No Gemini, SERP, or production PostgreSQL connection is made.
"""
import asyncio
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import agent_config_stub as config
from langchain_core.language_models.fake import FakeListLLM
from sqlalchemy.exc import DBAPIError


class SQLValidationTests(unittest.TestCase):
    def test_readonly_analytics_are_accepted(self):
        queries = [
            "SELECT SUM(q2_revenue) FROM regional_sales;",
            "SELECT s.region FROM regional_sales AS s ORDER BY s.q2_revenue DESC LIMIT 3",
            "WITH totals AS (SELECT SUM(q2_revenue) AS total FROM regional_sales) SELECT total FROM totals",
            "SELECT region FROM regional_sales UNION SELECT region FROM regional_sales",
            "SELECT s.region FROM regional_sales s JOIN regional_sales t ON s.id = t.id",
            "SELECT 'SQL mutation words DELETE; UPDATE are harmless literal text'",
            "SELECT COUNT(*), ROUND(AVG(q2_revenue), 2) FROM public.regional_sales",
        ]
        for dialect in ("sqlite", "postgres"):
            for query in queries:
                with self.subTest(dialect=dialect, query=query):
                    self.assertTrue(config.SQLQueryOutputParser(dialect=dialect).parse(query))

    def test_mutations_and_unsafe_selects_are_rejected(self):
        queries = [
            "DELETE FROM regional_sales",
            "UPDATE regional_sales SET q2_revenue = 0",
            "DROP TABLE regional_sales",
            "WITH changed AS (DELETE FROM regional_sales RETURNING *) SELECT * FROM changed",
            "WITH changed AS (UPDATE regional_sales SET q2_revenue = 0 RETURNING *) SELECT * FROM changed",
            "SELECT 1; DELETE FROM regional_sales;",
            "SELECT * INTO copied_sales FROM regional_sales",
            "SELECT * FROM regional_sales FOR UPDATE",
            "SELECT set_config('transaction_read_only', 'off', false)",
            "SELECT pg_sleep(10)",
            "SELECT load_extension('untrusted')",
            "SELECT * FROM pg_catalog.pg_roles",
            "SELECT * FROM unknown_table",
            "SELECT * FROM foreign_schema.regional_sales",
            "SELECT public.side_effect()",
        ]
        for dialect in ("sqlite", "postgres"):
            for query in queries:
                with self.subTest(dialect=dialect, query=query):
                    with self.assertRaises(ValueError):
                        config.SQLQueryOutputParser(dialect=dialect).parse(query)

    def test_sql_labels_and_markdown_are_supported(self):
        self.assertIn("SUM", config.SQLQueryOutputParser().parse(
            "SQLQuery: ```sql\nSELECT SUM(q2_revenue) FROM regional_sales;\n```\nSQLResult:"
        ))

    def test_sqlite_itself_blocks_writes_even_without_the_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "metrics.db"
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.executescript((config.MODULE_DIR / "data" / "enterprise_schema.sql").read_text())
            with patch.object(config, "initialize_local_database", return_value=database_path):
                database = config.connect_enterprise_database(demo=True)
                try:
                    with self.assertRaises(DBAPIError):
                        database.run("DELETE FROM regional_sales")
                    self.assertIn("5240000", database.run("SELECT SUM(q2_revenue) FROM regional_sales"))
                finally:
                    database._engine.dispose()


class AgentEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "metrics.db"
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executescript((config.MODULE_DIR / "data" / "enterprise_schema.sql").read_text())
        self.database_patch = patch.object(config, "initialize_local_database", return_value=self.path)
        self.database_patch.start()
        self.addCleanup(self.database_patch.stop)

    def build_agent(self, responses):
        with patch.object(config, "demo_model", return_value=FakeListLLM(responses=responses)):
            agent = config.build_nexus_executive_agent(demo=True)
        # Close the database captured by the SQL tool closure before deleting the fixture.
        sql_tool = next(tool for tool in agent.tools if tool.name == "EnterpriseSQLDB")
        sql_chain = sql_tool.func.__closure__[0].cell_contents
        self.addCleanup(sql_chain.database._engine.dispose)
        return agent

    def test_multi_tool_answer_keeps_all_sql_wiki_and_news_evidence(self):
        agent = self.build_agent([
            "Thought: Do I need to use a tool? Yes\nAction: EnterpriseSQLDB\nAction Input: What is total Q2 revenue?",
            "SELECT SUM(q2_revenue) FROM regional_sales",
            "Thought: Do I need to use a tool? Yes\nAction: EnterpriseSQLDB\nAction Input: What is total Q1 revenue?",
            "SELECT SUM(q1_revenue) FROM regional_sales",
            "Thought: Do I need to use a tool? Yes\nAction: HR_Ops_Wiki\nAction Input: How many remote-work days are allowed?",
            "Engineers may work remotely up to 3 days per week (HR Guidelines).",
            "Thought: Do I need to use a tool? Yes\nAction: LiveMarketNews\nAction Input: latest global market news",
            "Thought: Do I need to use a tool? No\nAI: Unsupported total $999 and invented news.",
        ])
        answer = agent.invoke({"input": "Compare Q1 and Q2 revenue, remote-work policy, and global market news."})["output"]
        for evidence in ("5240000", "4710000", "3 days", "OFFLINE DEMO", "HR and operations guidelines", "Market news"):
            self.assertIn(evidence, answer)
        self.assertNotIn("$999", answer)
        self.assertIn(answer, agent.memory.load_memory_variables({"input": "earlier revenue"})["chat_history"])

    def test_skipped_sql_is_verified_in_sync_and_async_paths(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                agent = self.build_agent([
                    "Thought: Do I need to use a tool? No\nAI: Revenue is $999.",
                    "SELECT SUM(q2_revenue) FROM regional_sales",
                ])
                if asynchronous:
                    result = asyncio.run(agent.ainvoke({"input": "What is total Q2 revenue?"}))
                else:
                    result = agent.invoke({"input": "What is total Q2 revenue?"})
                self.assertIn("5240000", result["output"])
                self.assertNotIn("$999", result["output"])
                self.assertEqual(result["intermediate_steps"][0][0].tool, "EnterpriseSQLDB")


class MemoryAndRoleBoundaryTests(unittest.TestCase):
    def test_old_exchange_can_be_retrieved_after_leaving_recent_window(self):
        semantic = config.initialize_semantic_memory(config.OfflineEmbeddings())
        memory = config.CombinedMemory(memories=[semantic, config.initialize_conversational_buffer()])
        memory.save_context(
            {"input": "cobaltbudget cobaltbudget cobaltbudget", "chat_history": "INJECTED_HISTORY", "semantic_history": "INJECTED_HISTORY"},
            {"output": "Approved cobaltbudget at 700", "intermediate_steps": "PRIVATE_TOOL_TRACE"},
        )
        for index in range(8):
            memory.save_context({"input": f"unrelatedtopic{index}"}, {"output": f"unrelatedanswer{index}"})
        loaded = memory.load_memory_variables({"input": "cobaltbudget cobaltbudget cobaltbudget"})
        self.assertNotIn("cobaltbudget", loaded["chat_history"])
        self.assertIn("Approved cobaltbudget at 700", loaded["semantic_history"])
        self.assertNotIn("INJECTED_HISTORY", loaded["semantic_history"])
        self.assertNotIn("PRIVATE_TOOL_TRACE", loaded["semantic_history"])

    def test_postgres_audit_rejects_unsafe_privileges(self):
        connection = MagicMock()
        connection.execute.return_value.scalar_one.return_value = True
        with self.assertRaisesRegex(ValueError, "SELECT-only"):
            config.assert_postgres_readonly_role(connection)

    def test_postgres_audit_rejects_missing_select_grant(self):
        connection = MagicMock()
        connection.execute.return_value.scalar_one.side_effect = [False, True, False]
        with self.assertRaisesRegex(ValueError, "lacks SELECT"):
            config.assert_postgres_readonly_role(connection)

    def test_postgres_audit_allows_safe_login_with_all_select_grants(self):
        connection = MagicMock()
        connection.execute.return_value.scalar_one.side_effect = [False, True, True, True]
        config.assert_postgres_readonly_role(connection)


if __name__ == "__main__":
    unittest.main()
