"""The MCP server: the tools of ``ceva.assistant.tools``, read-only, over streamable HTTP at ``/mcp``.

Any MCP client can use it to question the dataset directly; the dashboard's chat (``ceva.assistant.chat``)
calls the same tools in the same process.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from ceva import __version__
from ceva.assistant.data import SITE_URL, Dataset
from ceva.assistant.tools import Tools, tool_names

INSTRUCTIONS = """Tools over CEVA, a dataset of satellite-derived shorelines of the sand and gravel coasts of Europe
and the Mediterranean, 1984-2025: yearly positions and rates of change on transects 100 m apart, with a
coastal vulnerability index. Rates are in m/yr, positive seaward (accretion), negative landward (erosion);
a trend is significant when its 95 % range excludes zero. Places are given as longitude and latitude in
degrees. Answers carry map links and the dataset's DOI; rock, cliffs, mud and structures are not measured."""


def build(ds: Dataset) -> MCPServer:
    """The server with every tool bound to the dataset."""
    tools = Tools(ds)
    server = MCPServer(name="ceva", title="CEVA shorelines", instructions=INSTRUCTIONS, website_url=SITE_URL,
                       version=__version__)
    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                                open_world_hint=False)
    for name in tool_names():
        server.add_tool(getattr(tools, name), name=name, annotations=read_only)
    return server
