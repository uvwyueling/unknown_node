import requests

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "UnknownNodes/0.1 (educational project)"})


def fetch_neighbors(node: str) -> list[str]:
    """返回 Wikipedia 词条 node 所链接的全部文章标题（含 See also）。"""
    params = {
        "action": "query",
        "prop": "links",
        "titles": node,
        "redirects": "",        # 自动跟随重定向，拿到正文页的链接
        "pllimit": "max",
        "plnamespace": 0,       # 只要主命名空间（正文文章）
        "format": "json",
        "formatversion": "2",
    }

    neighbors: list[str] = []

    while True:
        try:
            resp = _SESSION.get(_WIKI_API, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Wikipedia API 请求失败：{exc}") from exc

        for page in data.get("query", {}).get("pages", []):
            for link in page.get("links", []):
                neighbors.append(link["title"])

        if "continue" not in data:
            break
        params["plcontinue"] = data["continue"]["plcontinue"]

    return neighbors
