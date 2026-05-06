import json
from datetime import datetime
import requests
import os
from tools.quota import register_search


def _parse_published_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def web_search(query, date, top_k=3, **kwargs):
    """
    Fetches search results using the Serper API and extracts URLs and snippets.

    Parameters:
    query (str): The search query.
    date (datetime.date): The date to filter results up to.
    top_k (int): The number of search results to fetch.

    Returns:
    tuple: A tuple containing two lists:
           - List of URLs from the organic results.
           - List of snippets from the organic results (empty string if snippet is missing).
    """
    # Bocha Web Search API endpoint
    url = "https://api.bochaai.com/v1/web-search"

    # Prepare the payload
    payload = json.dumps(
        {
            "query": query,
            "summary": True,
            "freshness": "noLimit",
            "count": max(top_k * 3, top_k),
        }
    )

    # Headers including the API key
    bocha_api_key = os.getenv("BOCHA_API_KEY", "")

    headers = {
        "Authorization": f"Bearer {bocha_api_key}",
        "Content-Type": "application/json",
    }

    quota_state = register_search(query)
    print(
        f"[BOCHA] query_count={quota_state['count']}/{quota_state['soft_limit']} "
        f"remaining={max(quota_state['soft_limit'] - quota_state['count'], 0)} "
        f"query={query}",
        flush=True,
    )

    # Make the POST request to the Serper API
    response = requests.request("POST", url, headers=headers, data=payload, timeout=20)

    # Check if the request was successful
    if response.status_code == 200:
        try:
            results = response.json()
        except json.JSONDecodeError:
            raise Exception(f"Bocha returned non-JSON response for query: {query}. Raw body: {response.text[:500]}")

        # Extract URLs and snippets from the Bocha response
        urls = []
        snippets = []

        web_pages = results.get("data", {}).get("webPages", {}).get("value", [])
        claim_dt = datetime.strptime(date, "%d-%m-%Y")
        for item in web_pages:
            if len(urls) >= top_k:
                break
            url = item.get("link", "")
            if not url:
                url = item.get("url", "")
            if url.endswith("pdf"):
                continue
            published_dt = _parse_published_date(item.get("datePublished"))
            if published_dt and published_dt.replace(tzinfo=None) > claim_dt:
                continue
            urls.append(url)
            snippets.append(item.get("summary") or item.get("snippet", ""))
        return urls, snippets
    else:
        raise Exception(
            f"Failed to fetch search results. Status code: {response.status_code}, Response: {response.text}"
        )
