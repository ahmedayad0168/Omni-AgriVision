"""
FarmIntelligenceAgent — LangChain-based agent for farm analysis.

Windows Smart App Control (SAC) workaround
==========================================
Recent `langchain-core` versions import `uuid_utils`, a compiled Rust
extension (`_uuid_utils.pyd`). SAC blocks the DLL because it is unsigned,
which breaks every langchain import. We fix this in code by registering a
pure-Python `uuid_utils` stub in `sys.modules` BEFORE any langchain import.
"""

# ---------------------------------------------------------------------------
# KEEP THIS BLOCK FIRST — must run before any langchain import
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types
import os as _os
import time as _time
import uuid as _uuid

if "uuid_utils" not in _sys.modules:
    def _pure_uuid7():
        """RFC 9562 UUIDv7 implemented in pure Python (no DLL needed)."""
        ms = int(_time.time() * 1000) & 0xFFFFFFFFFFFF
        rand_a = int.from_bytes(_os.urandom(2), "big") & 0x0FFF
        rand_b = int.from_bytes(_os.urandom(8), "big") & 0x3FFFFFFFFFFFFFFF
        value = (ms << 80) | (0x7 << 76) | (rand_a << 64) | (0x2 << 62) | rand_b
        return _uuid.UUID(int=value)

    _uuid_utils_stub = _types.ModuleType("uuid_utils")
    _uuid_utils_compat_stub = _types.ModuleType("uuid_utils.compat")
    _uuid_utils_compat_stub.uuid7 = _pure_uuid7
    _uuid_utils_stub.compat = _uuid_utils_compat_stub

    _sys.modules["uuid_utils"] = _uuid_utils_stub
    _sys.modules["uuid_utils.compat"] = _uuid_utils_compat_stub
# ---------------------------------------------------------------------------
# End of SAC workaround
# ---------------------------------------------------------------------------


import logging

from langchain_ollama import ChatOllama

try:
    from langchain_classic.agents import (
        AgentExecutor,
        create_tool_calling_agent,
    )
    _AGENT_FACTORY = "tool_calling"
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent
    _AGENT_FACTORY = "react"

from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# Suppress LangChain deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from src.agent.tools.field_tool import FieldTool
from src.agent.tools.weather_tool import WeatherTool
from src.agent.tools.satellite_tool import SatelliteTool
from src.agent.tools.research_tool import ResearchTool
from src.agent.tools.report_tool import ReportTool
from src.notifications.email_sender import EmailSender
from src.notifications.telegram_sender import TelegramSender
from src.agent.prompts.system_prompts import AGENT_SYSTEM_PROMPT
from src.data.database import DatabaseManager
from configs.settings import settings

logger = logging.getLogger(__name__)


class FarmIntelligenceAgent:
    def __init__(self, field_id: int, db_manager: DatabaseManager):
        self.field_id = field_id
        self.db = db_manager

        # Ollama LLM
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        # Tool implementations (each already bound to this field_id)
        self.field_tool = FieldTool(db_manager, field_id)
        self.weather_tool = WeatherTool()
        self.satellite_tool = SatelliteTool()
        self.research_tool = ResearchTool()
        self.report_tool = ReportTool()
        self.email_sender = EmailSender()
        self.telegram_sender = TelegramSender()

        self.tools = self._build_tools()

        # Memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        self.agent = self._create_agent()

    # -----------------------------------------------------------------------
    # Tool wrappers
    #
    # StructuredTool reads the function signature and passes named args
    # properly. Using `Tool` (unstructured) breaks any function with more
    # than one argument. Each wrapper:
    #   - has a clean, well-typed signature
    #   - never raises (returns an error string instead)
    #   - exposes the arguments the LLM actually needs
    # -----------------------------------------------------------------------
    def _build_tools(self):
        # ---- get_field_data (no args) -------------------------------------
        def _get_field_data() -> str:
            try:
                return self.field_tool.get_data()
            except Exception as e:
                return f"Error getting field data: {e}"

        # ---- get_historical_trends (days:int) -----------------------------
        def _get_historical_trends(days: int = 30) -> str:
            try:
                return self.field_tool.get_historical_trends(days=int(days))
            except Exception as e:
                return f"Error getting historical trends: {e}"

        # ---- get_weather_forecast (no args) -------------------------------
        def _get_weather_forecast() -> str:
            try:
                return self.weather_tool.get_forecast()
            except Exception as e:
                return f"Error getting weather forecast: {e}"

        # ---- get_satellite_indices (no args) ------------------------------
        def _get_satellite_indices() -> str:
            try:
                return self.satellite_tool.get_ndvi()
            except Exception as e:
                return f"Error getting satellite indices: {e}"

        # ---- search_knowledge_base (query, k) -----------------------------
        def _search_knowledge_base(query: str, k: int = 5) -> str:
            try:
                return self.research_tool.search_knowledge_base(
                    query=query, k=int(k)
                )
            except Exception as e:
                return f"Error searching knowledge base: {e}"

        # ---- search_web (query, max_results) ------------------------------
        def _search_web(query: str, max_results: int = 5) -> str:
            try:
                return self.research_tool.search_web(
                    query=query, max_results=int(max_results)
                )
            except Exception as e:
                return f"Error searching web: {e}"

        # ---- generate_report (no args) ------------------------------------
        def _generate_report() -> str:
            try:
                return self.report_tool.generate_report({}, field_id=self.field_id, db_manager=self.db)
            except Exception as e:
                return f"Error generating report: {e}"

        # ---- send_email (to, subject, body) -------------------------------
        def _send_email(to: str, subject: str, body: str) -> str:
            try:
                self.email_sender.send(to, subject, body)
                return f"Email sent successfully to {to}"
            except Exception as e:
                return f"Error sending email: {e}"

        # ---- send_telegram (message) -------------------------------------
        def _send_telegram(message: str) -> str:
            try:
                self.telegram_sender.send(message)
                return "Telegram message sent successfully"
            except Exception as e:
                return f"Error sending Telegram message: {e}"

        return [
            StructuredTool.from_function(
                func=_get_field_data,
                name="get_field_data",
                description=(
                    "Get the latest scan summary for the current field "
                    "(plants, pests, weeds, diseases). Takes no arguments."
                ),
            ),
            StructuredTool.from_function(
                func=_get_historical_trends,
                name="get_historical_trends",
                description=(
                    "Get historical scan trends for the current field over the "
                    "last N days. Argument `days` is an integer (default 30)."
                ),
            ),
            StructuredTool.from_function(
                func=_get_weather_forecast,
                name="get_weather_forecast",
                description=(
                    "Get a 7-day weather forecast for the current field location. "
                    "Takes no arguments."
                ),
            ),
            StructuredTool.from_function(
                func=_get_satellite_indices,
                name="get_satellite_indices",
                description=(
                    "Get NDVI/NDRE satellite vegetation indices for the current "
                    "field. Takes no arguments."
                ),
            ),
            StructuredTool.from_function(
                func=_search_knowledge_base,
                name="search_knowledge_base",
                description=(
                    "Search the local agricultural knowledge base for crop / "
                    "disease / pest information. Argument `query` is a string."
                ),
            ),
            StructuredTool.from_function(
                func=_search_web,
                name="search_web",
                description=(
                    "Search the web (DuckDuckGo) for general agricultural "
                    "information. Argument `query` is a string."
                ),
            ),
            StructuredTool.from_function(
                func=_generate_report,
                name="generate_report",
                description=(
                    "Generate a comprehensive HTML report for the current field. "
                    "Takes no arguments."
                ),
            ),
            StructuredTool.from_function(
                func=_send_email,
                name="send_email",
                description=(
                    "Send an email with the report or analysis results. "
                    "Arguments: to (recipient email), subject (email subject), body (email content)."
                ),
            ),
            StructuredTool.from_function(
                func=_send_telegram,
                name="send_telegram",
                description=(
                    "Send a Telegram message with the report or analysis results. "
                    "Argument: message (the message content to send)."
                ),
            ),
        ]

    # -----------------------------------------------------------------------
    # Agent construction
    # -----------------------------------------------------------------------
    def _create_agent(self):
        if _AGENT_FACTORY == "tool_calling":
            prompt = ChatPromptTemplate.from_messages([
                ("system", AGENT_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            return create_tool_calling_agent(self.llm, self.tools, prompt)

        # ReAct fallback
        react_system = (
            AGENT_SYSTEM_PROMPT
            + "\n\nYou have access to the following tools:\n{tools}\n"
            + "Use the following format:\n"
            + "Question: the input question\n"
            + "Thought: what to do\n"
            + "Action: one of [{tool_names}]\n"
            + "Action Input: input to the action\n"
            + "Observation: result\n"
            + "... (repeat as needed)\n"
            + "Thought: I now know the final answer\n"
            + "Final Answer: the final answer"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", react_system),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}\n\n{agent_scratchpad}"),
        ])
        return create_react_agent(self.llm, self.tools, prompt)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def ask(self, query: str) -> str:
        agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=False,
        )
        result = agent_executor.invoke({"input": query})
        return result.get("output", "(no output)")

    def analyze(self) -> str:
        query = """Perform a complete analysis of this farm:
        1. Assess overall crop health using the latest scan.
        2. Identify disease and pest pressures, including severity and affected zones.
        3. Analyze trends from historical data.
        4. Consider weather impacts (forecast and recent).
        5. Generate a risk assessment (disease, pest, yield).
        6. Provide actionable recommendations (priority order).
        7. Include evidence and cite sources where possible.
        Format the response as a structured farm report.
        """
        return self.ask(query)