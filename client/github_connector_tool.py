import os
import json
from dotenv import load_dotenv
import httpx
import redis
import hashlib

from langchain_core.tools import tool
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API_VERSION = "2022-11-28"

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")  # localhost for local dev, redis:6379 inside Docker
r = redis.from_url(REDIS_URL)

@tool()
async def github_post_issues(title : str, body : str) -> dict:
    """
    use this tool when user ask to raise a issue on github
    do not ask for owner and repo
    owner : abhay10023kgpian
    repo : testing_github_connector

    after issue creation show url to see the created issue

    Args:
        title (str): Title of the issue
        body (str): Body of the issue


    return: 
    status (int): "status code"
    message (str): "message"

    """

    owner = "abhay10023kgpian"
    repo = "testing_github_connector"
    content_string = f"{owner}-{repo}-{title}-{body}"
    
    idempotency = hashlib.sha256(content_string.encode()).hexdigest()

    cached = r.get(f"idempotency:{idempotency}")

    if cached:
        cached_data = json.loads(cached)
        return {
            "status" : 208,
            "message" : "issue already exists",
            "url": cached_data.get("html_url", "")
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

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=issue)
            
            if response.status_code == 201:
                data = response.json()
                r.set(f"idempotency:{idempotency}", json.dumps(data), ex=3600)
                return {
                    "status": response.status_code,
                    "message": "issue created successfully",
                    "url": data.get("html_url", "")
                }
            
            return {
                "status" : response.status_code,
                "message" : f"issue creation failed: {response.text}"
            }
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
        return {
            "status" : 500,
            "message" : "issue creation failed, check you internet connection"
        }
    except Exception as exc:
        print(f"An unexpected error occurred: {exc}")
        return {
            "status" : 500,
            "message" : "issue creation failed, internal error on from github"
        }


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(github_post_issues.ainvoke({"title": "Testing 10", "body": "This is a test issue."})))
