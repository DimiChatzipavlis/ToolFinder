"""Back-compat shim: the ToolFinder MCP bridge now lives in `toolfinder.mcp_server`.

Kept so existing host configs that launch this file by path keep working:

    {"command": "python", "args": ["ToolFinder_mcp_server.py"]}

Equivalent, after `pip install -e .`:  the `toolfinder-mcp` console command, or
`python -m toolfinder.mcp_server`. See docs/MCP_SERVER.md.
"""

from toolfinder.mcp_server import main, mcp  # noqa: F401  (re-export `mcp` for direct fastmcp use)

if __name__ == "__main__":
    main()
