import requests
from typing import Dict, Any, Optional


class MCPGateway:
    """Client for querying external MCP servers (College databases, Entrance APIs)."""

    def __init__(self, mcp_server_url: Optional[str] = None):
        self.server_url = mcp_server_url or "https://api.education-mcp.org/v1"

    def query_mcp_service(self, service_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Routes structured tools/resource requests through MCP endpoint."""
        try:
            endpoint = f"{self.server_url}/{service_name}"
            response = requests.post(endpoint, json=payload, timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"error": f"MCP returned status code {response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to connect to MCP Server: {str(e)}"}


mcp_gateway = MCPGateway()