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

idempotency_store = {} 

async def github_post_issues(owner:str, repo : str, title : str, body : str, idempotency: str) -> dict:

    cached = r.get(f"idempotency:{idempotency}")

    if cached:
        return {
            "status" : 200,
            "data" : json.loads(cached),  # deserialize from JSON string
            "idem" : idempotency
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
            "status" : response.status_code,
            "data" : response.json(),
            "idem" : idempotency
        }

    return{
        "error" : "Something went wrong",
        "code" : 500, 
        "idem" : idempotency
    }


if __name__ == "__main__":
    import asyncio
    import uuid
    unique_key = f"idempotency_key_{uuid.uuid4()}"
    print(asyncio.run(github_post_issues("abhay10023kgpian", "testing_github_connector", "Testing 1", "This is a test issue.", "idempotency_key_4a045523-2dda-4d73-b160-8c30f10b4748")))
    