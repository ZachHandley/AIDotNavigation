"""
ObjectAgent - AI-Powered Large Data Navigation Workflow with Sorting

A LlamaIndex Workflow that demonstrates how AI agents can intelligently navigate,
sort, and extract information from complex nested data structures that are too large
for normal context windows.

Features:
- AI agent uses function calling to explore large data structures
- Intelligent path discovery and data extraction
- Advanced sorting capabilities for arrays and collections
- Preview paths with depth limiting to manage context size
- Goal-oriented data extraction and analysis

Author: Zachary Handley
License: MIT
"""

from typing import Any, Dict, List, Optional, Type, Union
import json
from pydantic import BaseModel, Field
from loguru import logger

from llama_index.core.workflow import (
    Event,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Context,
)
from llama_index.core.tools import FunctionTool, ToolSelection, ToolOutput
from llama_index.core.tools.types import BaseTool
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from app.config import get_settings
from app.path_navigator import PathNavigator, SortType


class FunctionResponse(BaseModel):
    """Standardized response format for all function calls"""

    success: bool
    message: str
    data: Any = None
    error: Optional[str] = None


class ObjectAnalysisParams(BaseModel):
    """Parameters for object analysis workflow"""

    data: Any = Field(..., description="The complex data structure to analyze")
    goal: str = Field(..., description="What you want to achieve with this data")
    max_depth: int = Field(default=4, description="Maximum depth for path traversal")
    max_paths: int = Field(
        default=100, description="Maximum number of paths to explore"
    )
    preview_depth: int = Field(default=2, description="Depth for path previews")


class AnalyzeDataEvent(Event):
    """Event to start data analysis"""

    params: ObjectAnalysisParams


class LLMAnalysisEvent(Event):
    """Event for LLM to analyze data"""

    params: ObjectAnalysisParams


class ToolCallEvent(Event):
    """Event for handling tool calls"""

    tool_calls: List[ToolSelection]
    params: ObjectAnalysisParams


class ObjectAgent(Workflow):
    """
    AI-powered workflow for navigating, sorting, and analyzing large data structures.

    The AI agent uses function calling to intelligently explore data that would
    otherwise be too large to fit in context, making targeted queries to extract
    the information needed to achieve the specified goal.
    """

    chat_history: List[ChatMessage] = []
    _current_data: Any | None = None
    tools: List[BaseTool] = []
    settings = get_settings()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = OpenAI(model="gpt-4o", temperature=0)
        assert self.llm.metadata.is_function_calling_model

        # Current data being analyzed
        self._current_data = None

        # Register analysis tools (including sorting tools)
        self.tools = self._register_analysis_tools()

    def _register_analysis_tools(self) -> List[BaseTool]:
        """Register tools for AI to navigate, sort, and analyze data"""
        return [
            # Navigation tools
            FunctionTool.from_defaults(fn=self.get_structure_overview),
            FunctionTool.from_defaults(fn=self.preview_paths),
            FunctionTool.from_defaults(fn=self.get_value_by_path),
            FunctionTool.from_defaults(
                fn=self.get_values_from_path
            ),  # New tool for targeted extraction
            FunctionTool.from_defaults(fn=self.list_available_paths),
            FunctionTool.from_defaults(fn=self.search_paths),
            FunctionTool.from_defaults(fn=self.get_path_info),
            # Sorting tools
            FunctionTool.from_defaults(fn=self.sort_by_path),
            FunctionTool.from_defaults(fn=self.get_sortable_paths),
        ]

    @step
    async def start_analysis(self, ctx: Context, ev: StartEvent) -> AnalyzeDataEvent:
        """Entry point - extract parameters and begin analysis"""

        # Handle different parameter formats
        if hasattr(ev, "params"):
            if isinstance(ev.params, dict):
                params = ObjectAnalysisParams(**ev.params)
            else:
                params = ev.params
        else:
            # Fallback - try to construct from event attributes
            params_dict = {}
            for field_name in ObjectAnalysisParams.model_fields.keys():
                if hasattr(ev, field_name):
                    params_dict[field_name] = getattr(ev, field_name)
            params = ObjectAnalysisParams(**params_dict)

        self._current_data = params.data

        await ctx.set("params", params)

        logger.info(f"Starting AI analysis of data structure with goal: {params.goal}")

        return AnalyzeDataEvent(params=params)

    @step
    async def prepare_analysis_chat(
        self, ctx: Context, ev: AnalyzeDataEvent
    ) -> LLMAnalysisEvent:
        """Prepare the initial chat for AI analysis"""

        params = ev.params

        # Clear previous chat history
        self.chat_history = []

        system_prompt = f"""You are an expert AI data analyst with access to powerful navigation and sorting tools for exploring complex data structures.

ANALYSIS GOAL: {params.goal}

AVAILABLE TOOLS:

NAVIGATION TOOLS:
1. get_structure_overview() - Get high-level structure analysis of the entire data structure
2. preview_paths(depth=N) - Preview all available paths up to specified depth with sample values
3. get_value_by_path(path="dot.notation.path") - Get specific value using dot notation (e.g., "user.profile.contacts[0].email")
4. get_values_from_path(path="dot.notation.path", keys=["key1", "key2"]) - Extract specific keys from objects at a path
5. list_available_paths(max_paths=N) - List all available navigation paths in the data structure
6. search_paths(search_term="keyword") - Find all paths containing specific terms or keywords
7. get_path_info(path="specific.path") - Get detailed information about a specific path including children

SORTING TOOLS:
8. sort_by_path(path="array.path", sort_key="property", reverse=False, sort_type="auto", limit=None) - Sort arrays by values or object properties
9. get_sortable_paths() - Find all arrays that can be sorted and their available sort options

IMPORTANT CONTEXT MANAGEMENT:
- If get_value_by_path returns too much data and causes context issues, you'll get an error message
- When this happens, use get_values_from_path() to extract only specific keys you need
- Or use get_value_by_path() with more specific paths (like array[0] instead of entire array)
- The system will guide you to use smaller chunks when needed

SORTING CAPABILITIES:
- Sort simple arrays alphabetically, numerically, or by boolean values
- Sort arrays of objects by any property (supports nested dot notation like "user.profile.name")
- Auto-detect best sort type (alphabetical, numerical, boolean) or specify manually
- Sort in ascending or descending order
- Limit results after sorting
- Get metadata about sortable arrays and their properties

SORT EXAMPLES:
- sort_by_path("users") - Sort a users array naturally (auto-detect best sort method)
- sort_by_path("products", sort_key="price", sort_type="numerical") - Sort products by price numerically
- sort_by_path("articles", sort_key="publishDate", reverse=True, limit=10) - Get top 10 most recent articles
- sort_by_path("customers", sort_key="profile.address.city") - Sort by nested property

STRATEGY:
1. Start with get_structure_overview() to understand the data layout
2. Use search_paths() to find paths relevant to your goal
3. Identify sortable arrays using get_sortable_paths() if sorting would help achieve your goal
4. Sort relevant arrays to organize data effectively using sort_by_path()
5. Extract specific values using get_value_by_path() or get_values_from_path()
6. Get detailed info about promising paths using get_path_info()
7. Synthesize findings into a comprehensive analysis

IMPORTANT: 
- The data structure is too large for normal context, so use these tools strategically
- Focus on paths that are most relevant to achieving your goal
- Use sorting to organize data when it would help with analysis (e.g., finding top/bottom items, alphabetical ordering)
- You can call multiple tools in succession to explore the data systematically
- All tools return standardized responses with success/error information
- If you get context limit errors, use more targeted extraction methods

Begin your analysis by exploring the structure and identifying the most promising paths and sortable arrays for your goal.
"""

        user_prompt = f"""Please analyze this data structure to achieve the goal: "{params.goal}"

The data is too large to view directly, so use your available tools to explore it systematically. Pay special attention to any arrays that might benefit from sorting to help achieve the goal.

Provide a thorough analysis with specific findings and actionable insights.
"""

        # Create the initial conversation
        system_message = ChatMessage(role="system", content=system_prompt)
        user_message = ChatMessage(role="user", content=user_prompt)

        self.chat_history.append(system_message)
        self.chat_history.append(user_message)

        return LLMAnalysisEvent(params=params)

    @step
    async def handle_llm_analysis(
        self, ctx: Context, ev: LLMAnalysisEvent
    ) -> ToolCallEvent | StopEvent:
        """Handle LLM analysis and tool calls"""

        try:
            response = await self.llm.achat_with_tools(
                chat_history=self.chat_history, tools=self.tools, verbose=True
            )
        except Exception as e:
            error_msg = str(e)

            # Check if it's a context length error (basic ass check)
            if "maximum context length" in error_msg.lower():
                logger.warning("Context length exceeded, condensing chat history")
                self._condense_chat_history(ev.params.goal)

                # Try again with condensed history
                try:
                    response = await self.llm.achat_with_tools(
                        chat_history=self.chat_history, tools=self.tools, verbose=True
                    )
                except Exception as retry_error:
                    logger.error(f"Error calling LLM after condensing: {retry_error}")
                    return StopEvent(
                        result={
                            "error": f"LLM call failed even after condensing: {str(retry_error)}",
                            "success": False,
                        }
                    )
            else:
                logger.error(f"Error calling LLM: {e}")
                return StopEvent(
                    result={"error": f"LLM call failed: {str(e)}", "success": False}
                )

        if not response or not response.message:
            logger.error("No response received from LLM")
            return StopEvent(result={"error": "No response from LLM", "success": False})

        # Handle response content - it's OK if content is None for tool-only responses
        response_content = ""
        if response.message.content:
            response_content = response.message.content
            logger.info(f"AI Response: {response_content[:200]}...")

        # Add response to chat history
        try:
            self.chat_history.append(response.message)
        except Exception as e:
            logger.error(f"Error adding response to chat history: {e}")
            return StopEvent(
                result={"error": f"Chat history error: {str(e)}", "success": False}
            )

        # Check for tool calls
        try:
            tool_calls = self.llm.get_tool_calls_from_response(
                response, error_on_no_tool_call=False
            )
        except Exception as e:
            logger.warning(f"Error extracting tool calls: {e}")
            tool_calls = []

        if tool_calls:
            logger.info(
                f"AI wants to use {len(tool_calls)} tools: {[tc.tool_name for tc in tool_calls]}"
            )
            return ToolCallEvent(tool_calls=tool_calls, params=ev.params)
        else:
            # Analysis complete - return final result
            logger.info("Analysis complete - no more tool calls requested")

            # If we have content, use it; otherwise indicate analysis is complete
            final_analysis = (
                response_content
                if response_content
                else "Analysis completed through tool exploration"
            )

            return StopEvent(
                result={
                    "success": True,
                    "goal": ev.params.goal,
                    "analysis": final_analysis,
                    "data_size_info": (
                        self.get_structure_overview().model_dump()
                        if hasattr(self.get_structure_overview(), "model_dump")
                        else self.get_structure_overview()
                    ),
                }
            )

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> LLMAnalysisEvent:
        """Execute tool calls and continue analysis"""

        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        for tool_call in ev.tool_calls:
            logger.info(
                f"AI executing tool: {tool_call.tool_name} with args: {tool_call.tool_kwargs}"
            )
            tool = tools_by_name.get(tool_call.tool_name)

            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }

            if not tool:
                error_content = json.dumps(
                    {
                        "success": False,
                        "error": f"Tool {tool_call.tool_name} does not exist",
                    }
                )
                tool_message = ChatMessage(
                    role="tool",
                    content=error_content,
                    additional_kwargs=additional_kwargs,
                )
                self.chat_history.append(tool_message)
                continue

            try:
                # Execute the tool
                if hasattr(tool, "acall"):
                    tool_output = await tool.acall(**tool_call.tool_kwargs)
                else:
                    tool_output = tool(**tool_call.tool_kwargs)

                # Format output for AI - ensure it's JSON serializable
                if isinstance(tool_output.raw_output, FunctionResponse):
                    content = tool_output.raw_output.model_dump_json(indent=2)
                elif hasattr(tool_output.raw_output, "model_dump"):
                    content = tool_output.raw_output.model_dump_json(indent=2)
                elif isinstance(tool_output.raw_output, dict):
                    content = json.dumps(tool_output.raw_output, indent=2, default=str)
                else:
                    content = json.dumps(
                        {"success": True, "data": tool_output.raw_output},
                        indent=2,
                        default=str,
                    )

                # Check if content is too large (rough estimate)
                if len(content) > self.settings.TOO_LARGE_THRESHOLD:  # Large response
                    if tool_call.tool_name == "get_value_by_path":
                        # For get_value_by_path, give AI guidance to use smaller chunks
                        guidance_message = f"""The path '{tool_call.tool_kwargs.get('path', '')}' returned too much data for the context window. 

Instead, try one of these approaches:
1. Use get_values_from_path(path="{tool_call.tool_kwargs.get('path', '')}", keys=["key1", "key2"]) to extract only specific keys you need
2. Use more specific paths like {tool_call.tool_kwargs.get('path', '')}[0] to get individual items
3. Use get_path_info() first to understand the structure before extracting

Please try a more targeted approach to get the information you need for your analysis goal."""

                        tool_message = ChatMessage(
                            role="tool",
                            content=json.dumps(
                                {
                                    "success": False,
                                    "error": "Response too large for context",
                                    "guidance": guidance_message,
                                },
                                indent=2,
                            ),
                            additional_kwargs=additional_kwargs,
                        )
                    else:
                        # For other tools, just note it was too large
                        tool_message = ChatMessage(
                            role="tool",
                            content=json.dumps(
                                {
                                    "success": False,
                                    "error": f"Tool response too large for context. Try using more specific parameters or breaking into smaller queries.",
                                },
                                indent=2,
                            ),
                            additional_kwargs=additional_kwargs,
                        )
                else:
                    # Normal sized response, use as-is
                    tool_message = ChatMessage(
                        role="tool",
                        content=content,
                        additional_kwargs=additional_kwargs,
                    )

                self.chat_history.append(tool_message)

            except Exception as e:
                logger.warning(f"Error executing tool {tool_call.tool_name}: {e}")
                error_response = FunctionResponse(
                    success=False, message=f"Tool execution failed", error=str(e)
                )
                error_content = error_response.model_dump_json(indent=2)

                tool_message = ChatMessage(
                    role="tool",
                    content=error_content,
                    additional_kwargs=additional_kwargs,
                )
                self.chat_history.append(tool_message)

        return LLMAnalysisEvent(params=ev.params)

    def _condense_chat_history(self, goal: str):
        """Condense chat history when context gets too large, keeping system and user messages"""
        if len(self.chat_history) <= 2:
            return

        logger.info(f"Condensing chat history from {len(self.chat_history)} messages")

        # Keep first two messages (system prompt and initial user message)
        system_message = self.chat_history[0]
        user_message = self.chat_history[1]

        # Get all the tool interactions
        tool_interactions = self.chat_history[2:]

        # Create a summary of the tool interactions focused on the goal
        summary_parts = []
        for msg in tool_interactions:
            if msg.role == "assistant" and msg.content:
                summary_parts.append(f"AI: {msg.content[:200]}...")
            elif msg.role == "tool":
                try:
                    tool_data = json.loads(msg.content)
                    if tool_data.get("success"):
                        summary_parts.append(
                            f"Tool {msg.additional_kwargs.get('name', 'unknown')}: Success - {tool_data.get('message', '')[:100]}"
                        )
                    else:
                        summary_parts.append(
                            f"Tool {msg.additional_kwargs.get('name', 'unknown')}: Failed - {tool_data.get('error', '')[:100]}"
                        )
                except:
                    summary_parts.append(f"Tool response: {msg.content[:100]}...")

        # Create condensed summary message
        condensed_content = f"""CONDENSED CHAT HISTORY - Previous tool exploration summary for goal: "{goal}"

{chr(10).join(summary_parts[:10])}  

Key findings from tool exploration have been preserved. Continue analysis from here."""

        condensed_message = ChatMessage(role="assistant", content=condensed_content)

        # Replace chat history with system, user, and condensed summary
        self.chat_history = [system_message, user_message, condensed_message]

        logger.info(f"Chat history condensed to {len(self.chat_history)} messages")

    # Tool Methods for AI Agent
    def get_structure_overview(self) -> FunctionResponse:
        """
        Get a high-level overview of the entire data structure.

        Returns:
            FunctionResponse with data containing:
            - data_type: The root type of the data structure
            - size_info: Information about the size/length of the data
            - top_level_info: Sample keys, value types, and structural information
        """
        try:
            data = self._current_data
            if data is None:
                return FunctionResponse(
                    success=False,
                    message="No data available for analysis",
                    error="Data not loaded",
                )

            overview = {
                "data_type": type(data).__name__,
                "size_info": {},
                "top_level_info": {},
            }

            if isinstance(data, dict):
                overview["size_info"]["keys_count"] = len(data)
                overview["top_level_info"]["sample_keys"] = list(data.keys())[:10]

                # Analyze top level value types
                type_counts = {}
                for key, value in list(data.items())[:20]:
                    value_type = type(value).__name__
                    type_counts[value_type] = type_counts.get(value_type, 0) + 1
                overview["top_level_info"]["value_types"] = type_counts

            elif isinstance(data, list):
                overview["size_info"]["length"] = len(data)
                if data:
                    overview["top_level_info"]["item_type"] = type(data[0]).__name__
                    if len(data) > 1:
                        overview["top_level_info"]["sample_types"] = [
                            type(item).__name__ for item in data[:5]
                        ]
            else:
                overview["size_info"]["type"] = type(data).__name__
                if hasattr(data, "__len__"):
                    try:
                        overview["size_info"]["length"] = len(data)
                    except:
                        pass

            return FunctionResponse(
                success=True,
                message=f"Successfully analyzed structure of {overview['data_type']} with {overview['size_info']}",
                data=overview,
            )

        except Exception as e:
            return FunctionResponse(
                success=False, message="Failed to get structure overview", error=str(e)
            )

    def preview_paths(self, depth: int = 2) -> FunctionResponse:
        """
        Get a preview of all available paths up to specified depth with sample values.

        Args:
            depth: Maximum depth to explore (default: 2)

        Returns:
            FunctionResponse with data containing:
            - total_paths: Total number of paths found
            - depth_limit: The depth limit used
            - path_previews: List of paths with their sample values and types
        """
        try:
            if self._current_data is None:
                return FunctionResponse(
                    success=False,
                    message="No data available for path preview",
                    error="Data not loaded",
                )

            # Get all paths up to the specified depth
            all_paths = PathNavigator.get_all_available_paths(
                self._current_data, max_depth=depth
            )

            preview = {
                "total_paths": len(all_paths),
                "depth_limit": depth,
                "path_previews": [],
            }

            # Sample paths and get their values with previews
            sample_paths = all_paths[:50]  # Reasonable limit for preview

            for path in sample_paths:
                try:
                    success, value, error = PathNavigator.get_value_by_path(
                        self._current_data, path
                    )

                    path_info = {
                        "path": path,
                        "depth": path.count(".") + path.count("["),
                        "success": success,
                    }

                    if success:
                        path_info["value_type"] = type(value).__name__
                        path_info["value_preview"] = self._get_safe_value_preview(value)
                    else:
                        path_info["error"] = error

                    preview["path_previews"].append(path_info)
                except Exception as path_error:
                    logger.warning(f"Error processing path {path}: {path_error}")
                    continue

            return FunctionResponse(
                success=True,
                message=f"Found {len(all_paths)} paths up to depth {depth}, showing preview of {len(preview['path_previews'])} paths",
                data=preview,
            )

        except Exception as e:
            return FunctionResponse(
                success=False, message="Failed to get path preview", error=str(e)
            )

    def get_value_by_path(self, path: str) -> FunctionResponse:
        """
        Get a specific value from the data structure using dot notation path.

        Args:
            path: Dot notation path (e.g., "user.profile.contacts[0].email")

        Returns:
            FunctionResponse with the requested value or error information and suggestions
        """
        try:
            if self._current_data is None:
                return FunctionResponse(
                    success=False, message="No data available", error="Data not loaded"
                )

            success, value, error = PathNavigator.get_value_by_path(
                self._current_data, path
            )

            if success:
                return FunctionResponse(
                    success=True,
                    message=f"Successfully retrieved value from path: {path}",
                    data={
                        "path": path,
                        "value": value,  # Return full value - let context management handle size
                        "value_type": type(value).__name__,
                        "size_info": self._get_safe_size_info(value),
                    },
                )
            else:
                # Provide suggestions for similar paths
                try:
                    all_paths = PathNavigator.get_all_available_paths(
                        self._current_data, max_depth=3
                    )
                    similar_paths = [
                        p
                        for p in all_paths
                        if any(part in p for part in path.split(".") if part)
                    ][:5]
                except:
                    similar_paths = []

                return FunctionResponse(
                    success=False,
                    message=f"Path not found: {path}",
                    error=error,
                    data={"suggestions": similar_paths},
                )

        except Exception as e:
            return FunctionResponse(
                success=False, message=f"Error accessing path: {path}", error=str(e)
            )

    def get_values_from_path(self, path: str, keys: List[str]) -> FunctionResponse:
        """
        Extract specific keys from objects at a given path. Useful for getting targeted data
        without overwhelming the context with large objects.

        Args:
            path: Dot notation path to the object or array of objects
            keys: List of keys to extract from each object

        Returns:
            FunctionResponse with extracted key-value pairs
        """
        try:
            if self._current_data is None:
                return FunctionResponse(
                    success=False, message="No data available", error="Data not loaded"
                )

            success, value, error = PathNavigator.get_value_by_path(
                self._current_data, path
            )

            if not success:
                return FunctionResponse(
                    success=False,
                    message=f"Path not found: {path}",
                    error=error,
                )

            extracted_data = []

            if isinstance(value, list):
                # Extract keys from each object in the array
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        extracted_item = {"_index": i}
                        for key in keys:
                            if "." in key:
                                # Handle nested keys
                                nested_success, nested_value, _ = (
                                    PathNavigator.get_value_by_path(item, key)
                                )
                                if nested_success:
                                    extracted_item[key] = nested_value
                                else:
                                    extracted_item[key] = None
                            else:
                                extracted_item[key] = item.get(key)
                        extracted_data.append(extracted_item)
                    else:
                        extracted_data.append(
                            {
                                "_index": i,
                                "_value": item,
                                "_note": f"Item is {type(item).__name__}, not dict",
                            }
                        )
            elif isinstance(value, dict):
                # Extract keys from single object
                extracted_item = {}
                for key in keys:
                    if "." in key:
                        # Handle nested keys
                        nested_success, nested_value, _ = (
                            PathNavigator.get_value_by_path(value, key)
                        )
                        if nested_success:
                            extracted_item[key] = nested_value
                        else:
                            extracted_item[key] = None
                    else:
                        extracted_item[key] = value.get(key)
                extracted_data = extracted_item
            else:
                return FunctionResponse(
                    success=False,
                    message=f"Path does not point to object or array of objects",
                    error=f"Value at path is {type(value).__name__}, expected dict or list",
                )

            return FunctionResponse(
                success=True,
                message=f"Successfully extracted keys {keys} from path: {path}",
                data={
                    "path": path,
                    "requested_keys": keys,
                    "extracted_data": extracted_data,
                    "total_items": (
                        len(extracted_data) if isinstance(extracted_data, list) else 1
                    ),
                },
            )

        except Exception as e:
            return FunctionResponse(
                success=False,
                message=f"Error extracting values from path: {path}",
                error=str(e),
            )

    def list_available_paths(self, max_paths: int = 100) -> FunctionResponse:
        """
        List all available navigation paths in the data structure.

        Args:
            max_paths: Maximum number of paths to return (default: 100)

        Returns:
            FunctionResponse with list of available paths
        """
        try:
            if self._current_data is None:
                return FunctionResponse(
                    success=False, message="No data available", error="Data not loaded"
                )

            all_paths = PathNavigator.get_all_available_paths(
                self._current_data, max_depth=5
            )

            return FunctionResponse(
                success=True,
                message=f"Found {len(all_paths)} total paths, returning first {min(len(all_paths), max_paths)}",
                data={
                    "total_available": len(all_paths),
                    "showing": min(len(all_paths), max_paths),
                    "paths": all_paths[:max_paths],
                },
            )

        except Exception as e:
            return FunctionResponse(
                success=False, message="Failed to list available paths", error=str(e)
            )

    def search_paths(self, search_term: str) -> FunctionResponse:
        """
        Search for paths containing specific terms or keywords.

        Args:
            search_term: Term to search for in path names (case-insensitive)

        Returns:
            FunctionResponse with matching paths and sample values
        """
        try:
            if self._current_data is None:
                return FunctionResponse(
                    success=False, message="No data available", error="Data not loaded"
                )

            all_paths = PathNavigator.get_all_available_paths(
                self._current_data, max_depth=4
            )

            # Case-insensitive search
            matching_paths = [
                path for path in all_paths if search_term.lower() in path.lower()
            ]

            # Get sample values for matching paths
            path_samples = []
            for path in matching_paths[:10]:
                try:
                    success, value, _ = PathNavigator.get_value_by_path(
                        self._current_data, path
                    )
                    if success:
                        path_samples.append(
                            {
                                "path": path,
                                "value_type": type(value).__name__,
                                "preview": self._get_safe_value_preview(value),
                            }
                        )
                except:
                    continue

            return FunctionResponse(
                success=True,
                message=f"Found {len(matching_paths)} paths containing '{search_term}'",
                data={
                    "search_term": search_term,
                    "total_matches": len(matching_paths),
                    "matching_paths": matching_paths[:20],
                    "path_samples": path_samples,
                },
            )

        except Exception as e:
            return FunctionResponse(
                success=False,
                message=f"Failed to search paths for term: {search_term}",
                error=str(e),
            )

    def get_path_info(self, path: str) -> FunctionResponse:
        """
        Get detailed information about a specific path including children and metadata.

        Args:
            path: The specific path to analyze

        Returns:
            FunctionResponse with detailed path information, children, and metadata
        """
        try:
            if self._current_data is None:
                return FunctionResponse(
                    success=False, message="No data available", error="Data not loaded"
                )

            success, value, error = PathNavigator.get_value_by_path(
                self._current_data, path
            )

            info = {
                "path": path,
                "exists": success,
                "depth": path.count(".") + path.count("["),
            }

            if success:
                info.update(
                    {
                        "value_type": type(value).__name__,
                        "size_info": self._get_safe_size_info(value),
                        "value_preview": self._get_safe_value_preview(value),
                    }
                )

                # Check if this path has children
                try:
                    all_paths = PathNavigator.get_all_available_paths(
                        self._current_data,
                        max_depth=path.count(".") + path.count("[") + 2,
                    )
                    child_paths = [
                        p
                        for p in all_paths
                        if p.startswith(path + ".") or p.startswith(path + "[")
                    ]
                    info["has_children"] = len(child_paths) > 0
                    info["children_count"] = len(child_paths)
                    info["sample_children"] = child_paths[:5]
                except:
                    info["has_children"] = False
                    info["children_count"] = 0
                    info["sample_children"] = []

                return FunctionResponse(
                    success=True,
                    message=f"Retrieved detailed info for path: {path}",
                    data=info,
                )
            else:
                info["error"] = error
                return FunctionResponse(
                    success=False,
                    message=f"Path does not exist: {path}",
                    error=error,
                    data=info,
                )

        except Exception as e:
            return FunctionResponse(
                success=False,
                message=f"Failed to get path info for: {path}",
                error=str(e),
            )

    # Sorting Tool Methods
    def sort_by_path(
        self,
        path: str,
        sort_key: Optional[str] = None,
        reverse: bool = False,
        sort_type: str = "auto",
        limit: Optional[int] = None,
    ) -> FunctionResponse:
        """
        Sort an array at the specified path by value or object property.

        Args:
            path: Dot notation path to the array to sort
            sort_key: For arrays of objects, the property to sort by (supports nested dot notation)
            reverse: Whether to sort in descending order (default: False for ascending)
            sort_type: Sort type - "auto", "alphabetical", "numerical", or "boolean" (default: "auto")
            limit: Optional limit on number of items to return after sorting

        Returns:
            FunctionResponse with sorted array data and sorting metadata
        """
        try:
            if self._current_data is None:
                return FunctionResponse(
                    success=False, message="No data available", error="Data not loaded"
                )

            # Convert string sort_type to enum
            sort_type_enum = SortType.AUTO
            if sort_type.lower() == "alphabetical":
                sort_type_enum = SortType.ALPHABETICAL
            elif sort_type.lower() == "numerical":
                sort_type_enum = SortType.NUMERICAL
            elif sort_type.lower() == "boolean":
                sort_type_enum = SortType.BOOLEAN

            # Perform the sort
            success, sorted_data, error = PathNavigator.sort_array_by_path(
                self._current_data,
                path,
                sort_key=sort_key,
                reverse=reverse,
                sort_type=sort_type_enum,
                limit=limit,
            )

            if success:
                return FunctionResponse(
                    success=True,
                    message=f"Successfully sorted array at path '{path}'"
                    + (f" by key '{sort_key}'" if sort_key else "")
                    + (f" (limited to {limit} items)" if limit else ""),
                    data={
                        "path": path,
                        "sort_key": sort_key,
                        "reverse": reverse,
                        "sort_type": sort_type,
                        "limit": limit,
                        "result_count": (
                            len(sorted_data) if isinstance(sorted_data, list) else 1
                        ),
                        "sorted_data": sorted_data,  # Return full data
                        "full_data_available": True,
                    },
                )
            else:
                return FunctionResponse(
                    success=False,
                    message=f"Failed to sort array at path '{path}'",
                    error=error,
                )

        except Exception as e:
            return FunctionResponse(
                success=False,
                message=f"Error sorting array at path '{path}'",
                error=str(e),
            )

    def get_sortable_paths(self) -> FunctionResponse:
        """
        Find all arrays in the data structure that can be sorted, with sorting metadata.

        Returns:
            FunctionResponse with information about sortable arrays and their properties
        """
        try:
            if self._current_data is None:
                return FunctionResponse(
                    success=False, message="No data available", error="Data not loaded"
                )

            sortable_paths = PathNavigator.get_sortable_paths(
                self._current_data, max_depth=4
            )

            # Organize results by path
            organized_results = []
            for path_info in sortable_paths:
                organized_results.append(
                    {
                        "path": path_info["path"],
                        "array_length": path_info["length"],
                        "can_sort_directly": path_info["can_sort_directly"],
                        "detected_sort_type": path_info["detected_type"],
                        "available_sort_keys": [
                            {
                                "key": key_info["key"],
                                "type": key_info["type"],
                                "coverage": f"{key_info['coverage']:.0%}",
                                "sample_values": key_info["sample_values"],
                            }
                            for key_info in path_info["sort_keys"]
                        ],
                        "sample_array_values": path_info["sample_values"],
                    }
                )

            return FunctionResponse(
                success=True,
                message=f"Found {len(organized_results)} sortable arrays in the data structure",
                data={
                    "total_sortable_arrays": len(organized_results),
                    "sortable_paths": organized_results,
                },
            )

        except Exception as e:
            return FunctionResponse(
                success=False, message="Failed to analyze sortable paths", error=str(e)
            )

    # Helper methods
    def _get_safe_value_preview(self, value: Any, max_length: int = 200) -> Any:
        """Get a safe preview of a value with size limits and type checking"""
        try:
            if value is None:
                return None
            elif isinstance(value, str):
                return value[:max_length] + "..." if len(value) > max_length else value
            elif isinstance(value, (list, tuple)):
                if len(value) <= 3:
                    return [
                        self._get_safe_value_preview(item, max_length=50)
                        for item in value
                    ]
                else:
                    preview_items = [
                        self._get_safe_value_preview(item, max_length=50)
                        for item in value[:3]
                    ]
                    return preview_items + [f"... {len(value) - 3} more items"]
            elif isinstance(value, dict):
                if len(value) <= 5:
                    return {
                        k: self._get_safe_value_preview(v, max_length=50)
                        for k, v in value.items()
                    }
                else:
                    items = list(value.items())[:5]
                    result = {
                        k: self._get_safe_value_preview(v, max_length=50)
                        for k, v in items
                    }
                    result["..."] = f"{len(value) - 5} more keys"
                    return result
            elif hasattr(value, "__str__"):
                str_val = str(value)
                return (
                    str_val[:max_length] + "..."
                    if len(str_val) > max_length
                    else str_val
                )
            else:
                return f"<{type(value).__name__} object>"
        except Exception as e:
            return f"<Error previewing {type(value).__name__}: {str(e)}>"

    def _get_safe_size_info(self, value: Any) -> Dict[str, Any]:
        """Get safe size information about a value"""
        try:
            info = {"type": type(value).__name__}

            if hasattr(value, "__len__"):
                try:
                    info["size"] = len(value)
                except:
                    pass

            if isinstance(value, dict):
                try:
                    info["keys"] = list(value.keys())
                except:
                    info["keys"] = ["<error reading keys>"]
            elif isinstance(value, list) and value:
                try:
                    info["sample_item_types"] = [
                        type(item).__name__ for item in value[:3]
                    ]
                except:
                    info["sample_item_types"] = ["<error reading types>"]

            return info
        except Exception as e:
            return {"type": "unknown", "error": str(e)}
