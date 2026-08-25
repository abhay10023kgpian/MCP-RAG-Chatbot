from fastmcp import FastMCP
import os
import json
from dotenv import load_dotenv
import httpx
import redis
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API_VERSION = "2022-11-28"

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")  # localhost for local dev, redis:6379 inside Docker
r = redis.from_url(REDIS_URL)

mcp = FastMCP("github_issues_server")

@mcp.tool()
async def github_post_issues(owner:str, repo : str, title : str, body : str, idempotency: str) -> dict:
    """
    use this tool when user ask to raise a issue on github

    Args:
        title (str): Title of the issue
        body (str): Body of the issue


    return: 
    status (int): "status code"

    """
    cached = r.get(f"idempotency:{idempotency}")

    if cached:
        return {
            "status" : 200
        }

    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION
    }
    
    issue = {
        "title" : title,
        "body" : body,
    }

    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=issue)
        
        r.setex(f"idempotency:{idempotency}", 3600, json.dumps(response.json()))  # must serialize dict → JSON string
        return {
            "status" : response.status_code
        }


if __name__ == "__main__":
    # Start the MCP server
    # Use the -u flag to ensure proper JSON-RPC formatting over stdout
    # Note: The actual execution inside Docker is handled by docker-compose
    import sys
    sys.argv = ["", "run", "--stdio", "mcp_rag_chatbot/server/github_issues_server.py"]
    mcp.run()

    