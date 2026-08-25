import os
from dotenv import load_dotenv
import httpx

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API_VERSION = "2022-11-28"

idempotency_store = {}

async def github_post_issues(owner:str, repo : str, title : str, body : str, idempotency: str) -> dict:
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

    if idempotency in idempotency_store:
        return idempotency_store[idempotency]
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=issue)
        
        idempotency_store[idempotency] = response.json()
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
    print(asyncio.run(github_post_issues("abhay10023kgpian", "testing_github_connector", "Testing 1", "This is a test issue.", unique_key)))
    