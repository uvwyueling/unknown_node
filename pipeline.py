import functools
import requests

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "UnknownNodes/0.1 (educational project)"})

# 打分层权重（w1 + w2 = 1）
_W1 = 0.6   # 邻居 Jaccard 重叠（图距离）
_W2 = 0.4   # 分类 Jaccard 重叠（语义距离）


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


# ─────────────────────────────────────────────
# 打分层内部工具（不对外暴露）
# ─────────────────────────────────────────────

@functools.lru_cache(maxsize=512)
def _neighbors_set(node: str) -> frozenset[str]:
    """fetch_neighbors 的缓存版，返回 frozenset 供集合运算。"""
    return frozenset(fetch_neighbors(node))


@functools.lru_cache(maxsize=512)
def _fetch_categories(node: str) -> frozenset[str]:
    """返回词条的非隐藏 Wikipedia 分类集合（分页取完）。"""
    params = {
        "action": "query",
        "prop": "categories",
        "titles": node,
        "redirects": "",
        "cllimit": "max",
        "clshow": "!hidden",    # 排除维护性隐藏分类
        "format": "json",
        "formatversion": "2",
    }
    categories: list[str] = []
    while True:
        try:
            resp = _SESSION.get(_WIKI_API, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Wikipedia API 请求失败：{exc}") from exc

        for page in data.get("query", {}).get("pages", []):
            for cat in page.get("categories", []):
                categories.append(cat["title"])

        if "continue" not in data:
            break
        params["clcontinue"] = data["continue"]["clcontinue"]

    return frozenset(categories)


def _jaccard(a: frozenset, b: frozenset) -> float:
    """两个集合的 Jaccard 相似度，空集 ∩ 空集 → 0.0（避免除零）。"""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ─────────────────────────────────────────────
# 打分层公开契约
# ─────────────────────────────────────────────

def score_node(候选: str, 当前: str, 终点: str) -> float:
    """
    计算候选词条的路径相关性得分。

    rel = w1 × A  +  w2 × B
      A：候选 与 当前 的邻居 Jaccard 重叠   → 越大说明两词条在图里越近
      B：候选 与 终点 的分类 Jaccard 重叠   → 越大说明候选越靠近目标语义区

    返回值 ∈ [0.0, 1.0]。
    """
    A = _jaccard(_neighbors_set(候选), _neighbors_set(当前))
    B = _jaccard(_fetch_categories(候选), _fetch_categories(终点))
    return _W1 * A + _W2 * B
