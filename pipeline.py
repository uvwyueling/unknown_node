import functools
import json
import math
import pathlib
import re
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import requests

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_SESSION   = requests.Session()
_SESSION.headers.update({"User-Agent": "UnknownNodes/0.1 (educational project)"})

# 打分层权重（w1 + w2 = 1）
_W1 = 0.6   # 邻居 Jaccard 重叠（图距离）
_W2 = 0.4   # 分类 Jaccard 重叠（语义距离）

# 编排层参数
_INNER_SIZE    = 5     # 内圈最多词条数
_OUTER_SIZE    = 10    # 外圈最多词条数
_MMR_LAMBDA    = 0.7   # λ：越大越偏相关性，越小越偏多样性
_CANDIDATE_CAP = 20    # 最多对多少候选词做完整 score_node

# 全局限速：两次 HTTP 请求之间的最小间隔（秒）
_REQUEST_INTERVAL = 0.25
_last_req: list[float] = [0.0]   # 用列表包一层，方便在函数里赋值


# ─────────────────────────────────────────────
# 记忆层数据结构
# ─────────────────────────────────────────────

@dataclass
class Step:
    """
    一次用户选词步骤的完整快照。

    必填字段（调用方提供）：
      当前   — 本轮中心词条
      终点   — 用户设定的目标词条
      内圈   — 本轮展示的内圈候选列表
      外圈   — 本轮展示的外圈候选列表
      选择   — 用户点击的词条（必须在 候选 中）

    衍生属性（由实例自动计算，无需传入）：
      候选   — 内圈 + 外圈 的全集
      未选   — 候选中用户未点击的词条

    可选字段（save_step 会自动填入）：
      时间戳 — ISO 8601 UTC，如 "2026-06-04T21:30:00Z"
    """
    当前:   str
    终点:   str
    内圈:   list[str]
    外圈:   list[str]
    选择:   str
    时间戳: str = ""      # "" → save_step 自动填入 UTC 时间

    @property
    def 候选(self) -> list[str]:
        """全部展示给用户的候选词（内圈 + 外圈）。"""
        return self.内圈 + self.外圈

    @property
    def 未选(self) -> list[str]:
        """展示了但用户没有点击的词条。"""
        return [c for c in self.候选 if c != self.选择]


# ─────────────────────────────────────────────
# 记忆层：存储路径
# ─────────────────────────────────────────────

_HISTORY_FILE = pathlib.Path(__file__).parent / "history.jsonl"


# ─────────────────────────────────────────────
# 记忆层公开契约
# ─────────────────────────────────────────────

def save_step(step: Step) -> str:
    """
    将用户选词步骤持久化到 history.jsonl（JSON Lines，追加写）。

    返回值：step.选择（用户点击的词条），供下一轮编排层使用。

    错误场景：
      · 选择不在候选列表中 → ValueError（快速失败，不写文件）
      · 文件 IO 失败       → 直接上抛 OSError
    """
    if step.选择 not in step.候选:
        raise ValueError(
            f"选择 {step.选择!r} 不在候选列表 {step.候选} 中"
        )

    # 自动填入时间戳（ISO 8601 UTC，结尾 Z）
    ts = step.时间戳 or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    record = {
        "时间戳": ts,
        "当前":   step.当前,
        "终点":   step.终点,
        "内圈":   step.内圈,
        "外圈":   step.外圈,
        "选择":   step.选择,
        "未选":   step.未选,
    }

    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return step.选择


def load_history() -> list[dict]:
    """
    从 history.jsonl 读回全部步骤记录。
    文件不存在时返回空列表，不抛异常。
    """
    if not _HISTORY_FILE.exists():
        return []
    records = []
    with _HISTORY_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────
# HTTP 工具：所有层的唯一 API 入口
# ─────────────────────────────────────────────

def _api_get(params: dict) -> dict:
    """
    向 MediaWiki API 发请求，自动：
      · 主动限速（两次调用间隔 ≥ _REQUEST_INTERVAL 秒）
      · 遇到 429 时按 Retry-After 等待后重试
      · 其余网络异常指数退避，最多 5 次
    """
    elapsed = time.monotonic() - _last_req[0]
    if elapsed < _REQUEST_INTERVAL:
        time.sleep(_REQUEST_INTERVAL - elapsed)

    for attempt in range(5):
        try:
            resp = _SESSION.get(_WIKI_API, params=params, timeout=15)
            _last_req[0] = time.monotonic()
        except requests.RequestException as exc:
            if attempt == 4:
                raise RuntimeError(f"Wikipedia API 请求失败：{exc}") from exc
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
            time.sleep(wait)
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Wikipedia API 请求失败：{exc}") from exc

        return resp.json()

    raise RuntimeError("Wikipedia API 请求失败：超过最大重试次数（5 次）")


# ─────────────────────────────────────────────
# 取数层
# ─────────────────────────────────────────────

# 标准信息框里，认作「概念行」的标签白名单（这类信息框还混着生卒/网站等非概念行，
# 故用白名单而非黑名单）。系列侧栏则相反：默认全要，只黑掉人物行。
_INFOBOX_CONCEPT_LABELS = {
    "fields", "field", "related topics", "related fields",
    "subfields", "domains", "domain", "topics and fields",
}

# 系列侧栏里要排除的「人物/figures」行（按标签关键词匹配，大小写不敏感）
_FIGURE_ROW_PAT = re.compile(
    r"figure|people|person|scientist|researcher|pioneer|notable", re.I
)

# 停用词：明显的产品/公司名（手工维护，可继续补充）。
# 另有两条规则化过滤见 _is_noise：① "List of…" 前缀 ② (software)/(company) 等产品后缀。
# 还有一条「显示文本小写开头」（手写品牌名，如 [[imc FAMOS]]）在 _extract_article_links
# 里只对策展列表（See also/侧栏）生效——因为导语段里 [[map]]→"maps" 的小写是句子流，不是品牌。
_PRODUCT_DENYLIST = {
    "Adobe Inc.", "Adobe Systems", "Adobe Illustrator", "Adobe Photoshop",
    "Microsoft Excel", "Microsoft Power BI", "Tableau Software",
    "Google Charts", "D3.js",
}


def _is_noise(title: str) -> bool:
    """停用词过滤（基于词条标题，全部来源通用）：List of… 列表页、产品名单、产品后缀。"""
    if title in _PRODUCT_DENYLIST:
        return True
    if re.match(r"(?i)^lists? of\b", title):     # "List of …" / "Lists of …"
        return True
    if re.search(                                # "X (software)" / "X (company)" …
        r"(?i)\((software|company|operating system|programming language)\)$", title
    ):
        return True
    return False


def _extract_article_links(html_fragment: str, *, curated: bool = False) -> list[str]:
    """
    从一段渲染后的 HTML 中，按出现顺序抽出指向 Wikipedia 正文文章的链接标题。
    · 只认 /wiki/<Title> 形式（正文链接），忽略 /w/index.php?... 这类编辑链接
    · 跳过带命名空间前缀的链接（File: / Category: / Help: 等），它们是噪音
    · 过滤 _is_noise（List of… / 产品名 / 产品后缀）
    · curated=True（策展列表：See also、侧栏）时，额外剔除显示文本小写开头的手写
      品牌名（如 [[imc FAMOS]]）；导语段用 curated=False，避免误杀句中小写的 [[map]] 等。
    """
    out: list[str] = []
    for m in re.finditer(
        r'<a\b[^>]*href="/wiki/([^"#]+)"[^>]*>(.*?)</a>', html_fragment, re.S
    ):
        title   = unquote(m.group(1)).replace("_", " ").strip()
        display = re.sub(r"<[^>]+>", "", m.group(2)).strip()   # 链接显示文本
        if not title or ":" in title:
            continue
        if _is_noise(title):
            continue
        if curated and display[:1].islower():   # 列表里手写的小写品牌名 → 噪音
            continue
        out.append(title)
    return out


def _extract_see_also(html: str) -> list[str]:
    """抽出 See also 段（<h2 id="See_also"> 到下一个 <h2> 之间）的全部文章链接。"""
    m = re.search(r'id="See_also"', html)
    if not m:
        return []
    rest = html[m.end():]
    nxt = re.search(r"<h2\b", rest)
    segment = rest[: nxt.start()] if nxt else rest
    return _extract_article_links(segment, curated=True)


def _extract_infobox_rows(html: str) -> list[str]:
    """
    抽出信息框/系列侧栏里「概念行」的链接，排除人物/figures 行。
    覆盖两种真实结构：
      (a) 系列侧栏：<th class="sidebar-heading">标签</th> 单独一行，
                    紧跟 <td class="sidebar-content">…链接…</td> 另一行
                    —— 默认全取（Major dimensions / Related topics / Information
                       graphic types / Topics and fields …），只跳过人物行
                       （Important figures 等）。
      (b) 标准信息框：<th class="infobox-label">标签</th><td class="infobox-data">…</td>
                    —— 这类框混着生卒/网站等非概念行，故只取白名单标签
                       （Fields / Related topics / Subfields …）。
    """
    links: list[str] = []

    # (a) 系列侧栏：默认全取概念行，黑掉人物行
    for m in re.finditer(
        r'<th[^>]*class="[^"]*sidebar-heading[^"]*"[^>]*>(.*?)</th>', html, re.S
    ):
        label = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if _FIGURE_ROW_PAT.search(label):          # Important figures → 跳过
            continue
        after = html[m.end():]
        td = re.search(
            r'<td[^>]*class="[^"]*sidebar-content[^"]*"[^>]*>(.*?)</td>',
            after, re.S,
        )
        if td:
            links += _extract_article_links(td.group(1), curated=True)

    # (b) 标准信息框：只取白名单概念标签
    for m in re.finditer(
        r'<th[^>]*class="[^"]*infobox-label[^"]*"[^>]*>(.*?)</th>\s*'
        r'<td[^>]*class="[^"]*infobox-data[^"]*"[^>]*>(.*?)</td>',
        html, re.S,
    ):
        label = re.sub(r"<[^>]+>", "", m.group(1)).strip().lower()
        if label in _INFOBOX_CONCEPT_LABELS:
            links += _extract_article_links(m.group(2), curated=True)

    return links


def _extract_lead(html: str) -> list[str]:
    """
    抽出导语段（lead section）正文里的链接：即第一个 <h2> 之前、各 <p> 段落中的链接。
    只取 <p> 段落，天然避开同处于导语区的信息框/侧栏表格（它们用 <table>/<li>，不是 <p>）。
    """
    m = re.search(r"<h2\b", html)
    lead = html[: m.start()] if m else html
    links: list[str] = []
    for p in re.finditer(r"<p\b[^>]*>(.*?)</p>", lead, re.S):
        links += _extract_article_links(p.group(1))
    return links


def _fetch_neighbors_online(node: str) -> list[str]:
    """
    取数层的「联网 worker」——真正下载整页 HTML 并解析的那一层（慢）。
    不要直接调用它；外部统一走带磁盘缓存的 fetch_neighbors。

    取三处人工策展/概念性的链接，避开导航框人名、图说、产品名噪音：
      · See also 段落
      · 信息框 / 系列侧栏的所有概念行（Major dimensions / Related topics /
        Fields 等），但排除人物 figures 行
      · 导语段（lead section）正文段落

    再过一道停用词过滤（_is_noise）：剔除 "List of…" 列表页、明显产品名、品牌式小写开头标题。

    实现：取渲染后的 HTML（action=parse，自动跟随重定向），再定位上述三处。
    返回结果去重、保序，剔除指向词条自身的链接。
    词条不存在或无解析结果时，返回空列表（不向用户抛原始报错，红线 3）。
    """
    params = {
        "action":        "parse",
        "page":          node,
        "prop":          "text",
        "redirects":     "1",        # 自动跟随重定向（Data visualization → …）
        "format":        "json",
        "formatversion": "2",
    }
    data = _api_get(params)

    parse = data.get("parse")
    if data.get("error") or not parse or "text" not in parse:
        return []

    html     = parse["text"]
    resolved = parse.get("title", node)   # 重定向后的真实标题

    raw = (
        _extract_see_also(html)
        + _extract_infobox_rows(html)
        + _extract_lead(html)
    )

    # 去重保序，并去掉指向自身（原名/重定向后名）的链接
    # （停用词噪音已在 _extract_article_links 里按来源过滤完毕）
    seen: set[str] = set()
    out:  list[str] = []
    for title in raw:
        if title in (node, resolved) or title in seen:
            continue
        seen.add(title)
        out.append(title)
    return out


# ─────────────────────────────────────────────
# 取数层：磁盘缓存（L2，跨进程持久化）
#   _fetch_neighbors_online 每次都要下载整页 HTML，很慢；
#   这里把「词条名 → 邻居列表」永久缓存到本地 JSON 文件，
#   重启进程后依然命中，避免每次测试重等几分钟。
# ─────────────────────────────────────────────

_CACHE_FILE = pathlib.Path(__file__).parent / "neighbors_cache.json"
_nb_cache: dict[str, list[str]] | None = None   # 懒加载的内存镜像；None=尚未从磁盘载入


def _cache_ensure_loaded() -> dict[str, list[str]]:
    """把磁盘缓存懒加载进内存镜像；文件不存在或损坏时退化为空缓存。"""
    global _nb_cache
    if _nb_cache is None:
        try:
            with _CACHE_FILE.open(encoding="utf-8") as f:
                loaded = json.load(f)
            _nb_cache = loaded if isinstance(loaded, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            _nb_cache = {}
    return _nb_cache


def _cache_put(node: str, neighbors: list[str]) -> None:
    """写入内存镜像并原子落盘（先写临时文件再 replace，避免半截文件）。"""
    cache = _cache_ensure_loaded()
    cache[node] = neighbors
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(_CACHE_FILE)


def fetch_neighbors(node: str) -> list[str]:
    """
    取数层公开契约（输入词条名 → 返回邻居列表，语义与 worker 完全一致）。

    外层包了一道磁盘缓存：
      1. 先查本地缓存，命中就直接返回（不上网）；
      2. 未命中才调 _fetch_neighbors_online 上网，并把结果写入本地缓存。

    缓存文件：neighbors_cache.json（与本模块同目录）。
    手动清缓存：删掉该文件即可。
    """
    cache = _cache_ensure_loaded()
    if node in cache:
        return list(cache[node])          # 返回副本，避免调用方改到缓存内部

    neighbors = _fetch_neighbors_online(node)
    _cache_put(node, neighbors)
    return list(neighbors)


# ─────────────────────────────────────────────
# 打分层内部工具（不对外暴露）
# ─────────────────────────────────────────────

@functools.lru_cache(maxsize=512)
def _neighbors_set(node: str) -> frozenset[str]:
    """fetch_neighbors 的缓存版，返回 frozenset 供集合运算。"""
    return frozenset(fetch_neighbors(node))


# 分类缓存：手工 dict，方便批量预填（替代 lru_cache）
_cat_cache: dict[str, frozenset[str]] = {}


def _load_categories_batch(nodes: list[str]) -> None:
    """
    将多个词条的分类批量写入 _cat_cache（每批最多 50 个标题 → 1 次 API 调用）。
    已在缓存中的词条自动跳过。
    """
    to_fetch = [n for n in nodes if n not in _cat_cache]
    if not to_fetch:
        return

    for i in range(0, len(to_fetch), 50):
        batch = to_fetch[i : i + 50]
        params = {
            "action":        "query",
            "prop":          "categories",
            "titles":        "|".join(batch),
            "redirects":     "",
            "cllimit":       "max",
            "clshow":        "!hidden",   # 排除维护性隐藏分类
            "format":        "json",
            "formatversion": "2",
        }
        page_cats:    dict[str, list[str]] = {}
        all_redirects: dict[str, str]      = {}   # from → to

        while True:
            data = _api_get(params)
            # 收集重定向映射（用于将请求标题对齐到 API 返回标题）
            for r in data.get("query", {}).get("redirects", []):
                all_redirects[r["from"]] = r["to"]
            for page in data.get("query", {}).get("pages", []):
                title = page.get("title", "")
                for cat in page.get("categories", []):
                    page_cats.setdefault(title, []).append(cat["title"])
            if "continue" not in data:
                break
            params["clcontinue"] = data["continue"]["clcontinue"]

        for node in batch:
            canonical = all_redirects.get(node, node)
            _cat_cache[node] = frozenset(page_cats.get(canonical, []))


def _fetch_categories(node: str) -> frozenset[str]:
    """返回词条的非隐藏 Wikipedia 分类集合（优先命中 _cat_cache）。"""
    if node not in _cat_cache:
        _load_categories_batch([node])
    return _cat_cache[node]


@functools.lru_cache(maxsize=2048)
def _fetch_cat_parents(cat_title: str) -> frozenset[str]:
    """
    返回分类页面 cat_title 的父分类集合（即该分类页自身所属的上级分类）。
    结果带 lru_cache：同一分类标题全局只取一次。
    """
    params = {
        "action":        "query",
        "prop":          "categories",
        "titles":        cat_title,
        "cllimit":       "max",
        "clshow":        "!hidden",
        "format":        "json",
        "formatversion": "2",
    }
    parents: list[str] = []
    while True:
        data = _api_get(params)
        for page in data.get("query", {}).get("pages", []):
            for cat in page.get("categories", []):
                parents.append(cat["title"])
        if "continue" not in data:
            break
        params["clcontinue"] = data["continue"]["clcontinue"]
    return frozenset(parents)


def _category_proximity(candidate: str, target: str) -> float:
    """
    候选词到目标词的分类树接近度（越近 → 分数越高，返回 [0, 1]）。

    算法：从 target 的分类集合出发，逐层向上遍历父分类，
    看 candidate 的分类何时与之相交：

      层 0  candidate 与 target 直接共享某个分类         → 1.00
      层 1  candidate 的分类 ∩ target 分类的父类 ≠ ∅     → 0.50
      层 2  candidate 的分类 ∩ target 分类的祖父类 ≠ ∅   → 0.25
      未找到                                             → 0.00

    同一 target 的分类层次由 lru_cache 保证全局只计算一次，
    get_rings 循环中所有候选词共享该缓存。
    """
    cats_C = _fetch_categories(candidate)
    cats_T = _fetch_categories(target)

    # 层 0：直接共享分类
    if cats_C & cats_T:
        return 1.0

    # 层 1：target 各直接分类的父分类
    parents_T: frozenset[str] = frozenset()
    for cat in cats_T:
        parents_T |= _fetch_cat_parents(cat)
    if cats_C & parents_T:
        return 0.5

    # 层 2：target 分类的祖父分类（限制展开规模，避免 API 爆炸）
    _CAP = 40   # parents_T 超过此数时跳过层 2
    if len(parents_T) <= _CAP:
        gparents_T: frozenset[str] = frozenset()
        for cat in parents_T:
            gparents_T |= _fetch_cat_parents(cat)
        if cats_C & gparents_T:
            return 0.25

    return 0.0


def _jaccard(a: frozenset, b: frozenset) -> float:
    """两个集合的 Jaccard 相似度，空集返回 0.0（避免除零）。"""
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
      A：候选 与 当前  的邻居 Jaccard 重叠       → 越大说明两词条在图里越近
      B：候选 到 终点  的分类树接近度             → 越大说明候选越靠近目标语义区
         （层 0 = 1.0 / 层 1 = 0.5 / 层 2 = 0.25 / 未找到 = 0.0）

    返回值 ∈ [0.0, 1.0]。
    """
    A = _jaccard(_neighbors_set(候选), _neighbors_set(当前))
    B = _category_proximity(候选, 终点)
    return _W1 * A + _W2 * B


# ─────────────────────────────────────────────
# 编排层公开契约
# ─────────────────────────────────────────────

def get_rings(当前: str, 终点: str) -> dict:
    """
    编排层契约：用 MMR 从 当前 词条的邻居中挑出多样且相关的两圈节点。

    返回：
      {
        "中心": str,         # 当前词条本身
        "内圈": [str, ...],  # 与路径最相关、彼此最多样的词条
        "外圈": [str, ...],  # 补充多样性的词条
      }

    恒成立的不变量（见 CLAUDE.md）：
      len(内圈) <= len(外圈)
      中心词不出现在任何圈内
      全部节点无重复
      所有节点均来自 fetch_neighbors 的真实返回（不发明节点）
    """
    # ① 取数层拿邻居——编排层不直接调 Wikipedia API（红线 1）
    raw        = fetch_neighbors(当前)
    candidates = [c for c in raw if c != 当前]   # 去掉自引用

    if not candidates:
        return {"中心": 当前, "内圈": [], "外圈": []}

    # ② 限制候选数，控制 API 调用量
    candidates = candidates[:_CANDIDATE_CAP]

    # ③ 批量预取所有候选 + 终点 的分类（1 次 API 调用代替 20 次）
    _load_categories_batch(candidates + [终点])

    # ④ 打分层对每个候选打完整分数
    scores: dict[str, float] = {
        c: score_node(c, 当前, 终点) for c in candidates
    }

    # ⑤ MMR 循环：每轮从 remaining 中选 MMR 得分最高的词条
    total     = min(_INNER_SIZE + _OUTER_SIZE, len(candidates))
    selected: list[str] = []
    remaining = list(candidates)

    for _ in range(total):
        if not remaining:
            break

        if not selected:
            # 第一轮：直接选相关性最高的
            best = max(remaining, key=lambda c: scores[c])
        else:
            # 后续轮：MMR = λ × rel − (1−λ) × max_与已选的邻居相似度
            def _mmr(c: str) -> float:
                rel     = scores[c]
                max_sim = max(
                    _jaccard(_neighbors_set(c), _neighbors_set(s))
                    for s in selected       # selected 是同一对象，随循环增长
                )
                return _MMR_LAMBDA * rel - (1 - _MMR_LAMBDA) * max_sim

            best = max(remaining, key=_mmr)

        selected.append(best)
        remaining.remove(best)

    # ⑥ 切圈：inner_n = min(_INNER_SIZE, len//2) 保证 内圈 ≤ 外圈
    #    偶数时两圈相等，奇数时内圈少一个，均满足 ≤
    inner_n = min(_INNER_SIZE, len(selected) // 2)
    内圈 = selected[:inner_n]
    外圈 = selected[inner_n:]

    return {"中心": 当前, "内圈": 内圈, "外圈": 外圈}


# ─────────────────────────────────────────────
# 呈现层：HTML 渲染
# ─────────────────────────────────────────────

def _esc(text: str) -> str:
    """HTML 实体转义（用于把词条安全插入属性值和文本节点）。"""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_rings(
    rings: dict,
    *,
    历史: list[str] | None = None,
    终点: str = "",
) -> str:
    """
    呈现层契约：将 get_rings 的输出渲染为自包含 HTML 页面。

    布局（CSS 绝对定位，坐标用百分比）：
      中心词  —— 画布正中央
      内圈词  —— 小半径（22%）圆周，从 12 点方向均匀排列
      外圈词  —— 大半径（40%）圆周，从 12 点方向均匀排列

    交互：
      点击任意圈内词条 → onclick 调 choose(word)
      choose(word)     → GET /choose?word=<encoded> （配合 run_session）

    参数：
      rings  — get_rings 的返回值
      历史   — 用户已走路径，显示在页面底部
      终点   — 目标词条，显示在右上角
    """
    中心 = rings.get("中心", "")
    内圈 = rings.get("内圈", [])
    外圈 = rings.get("外圈", [])
    历史路径 = list(历史) if 历史 else [中心]

    def _circle_buttons(words: list[str], radius: float, css_class: str) -> str:
        """将词条列表按圆形坐标转为 HTML 按钮字符串。"""
        parts = []
        n = max(len(words), 1)
        for i, word in enumerate(words):
            θ = 2 * math.pi * i / n - math.pi / 2   # 从 12 点方向出发
            x = 50 + radius * math.cos(θ)
            y = 50 + radius * math.sin(θ)
            parts.append(
                f'<button class="{css_class}" '
                f'style="left:{x:.2f}%;top:{y:.2f}%;" '
                f'data-word="{_esc(word)}" '
                f'title="{_esc(word)}">{_esc(word)}</button>'
            )
        return "\n    ".join(parts)

    center_btn  = (
        f'<button class="center-node" style="left:50%;top:50%;" '
        f'title="{_esc(中心)}">{_esc(中心)}</button>'
    )
    inner_btns  = _circle_buttons(内圈, radius=22, css_class="inner-node")
    outer_btns  = _circle_buttons(外圈, radius=40, css_class="outer-node")
    path_html   = " → ".join(_esc(w) for w in 历史路径)

    return f"""\
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Unknown Nodes — {_esc(中心)}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #0f0f1a; color: #e0e0f0;
            font-family: system-ui, -apple-system, sans-serif; overflow: hidden; }}
    .stage {{ position: relative; width: 100vmin; height: 100vmin; margin: 0 auto; }}
    .ring-bg {{ position: absolute; border-radius: 50%;
                border: 1px solid rgba(255,255,255,0.07);
                transform: translate(-50%,-50%); top: 50%; left: 50%; }}
    .ring-inner {{ width: 46%;  height: 46%; }}
    .ring-outer  {{ width: 84%; height: 84%; }}
    button {{
      position: absolute; transform: translate(-50%,-50%);
      border: none; border-radius: 8px; padding: 6px 10px;
      font-size: 0.72rem; cursor: pointer;
      white-space: nowrap; max-width: 130px;
      overflow: hidden; text-overflow: ellipsis;
    }}
    .center-node {{
      background: #7c3aed; color: #fff;
      font-size: 1rem; padding: 12px 18px; border-radius: 12px;
      font-weight: bold; z-index: 3;
      cursor: default; pointer-events: none;
    }}
    .inner-node {{
      background: rgba(99,102,241,0.85); color: #fff; z-index: 2;
      transition: background .15s, transform .15s;
    }}
    .inner-node:hover {{
      background: #6366f1;
      transform: translate(-50%,-50%) scale(1.08);
    }}
    .outer-node {{
      background: rgba(45,55,72,0.90); color: #d1d5db; z-index: 1;
      transition: background .15s, transform .15s;
    }}
    .outer-node:hover {{
      background: rgba(99,102,241,0.55); color: #fff;
      transform: translate(-50%,-50%) scale(1.08);
    }}
    .footer {{
      position: fixed; bottom: 1rem; left: 50%;
      transform: translateX(-50%);
      font-size: 0.70rem; color: #4b5563; white-space: nowrap;
    }}
    .footer em {{ color: #a5b4fc; font-style: normal; }}
    .goal {{
      position: fixed; top: 0.9rem; right: 1.2rem;
      font-size: 0.70rem; color: #4b5563;
    }}
    .goal em {{ color: #34d399; font-style: normal; }}
  </style>
</head>
<body>
  <div class="stage">
    <div class="ring-bg ring-inner"></div>
    <div class="ring-bg ring-outer"></div>
    {center_btn}
    {inner_btns}
    {outer_btns}
  </div>
  <div class="footer">路径：<em>{path_html}</em> → ?</div>
  <div class="goal">目标：<em>{_esc(终点)}</em></div>
  <script>
    function choose(word) {{
      window.location.href = '/choose?word=' + encodeURIComponent(word);
    }}
    // 给每个带 data-word 的按钮绑定点击（避免内联 onclick 的引号转义问题）
    document.querySelectorAll('button[data-word]').forEach(function (btn) {{
      btn.addEventListener('click', function () {{
        choose(btn.getAttribute('data-word'));
      }});
    }});
  </script>
</body>
</html>"""


# ─────────────────────────────────────────────
# 呈现层：本地 Web 会话（串联所有层的完整循环）
# ─────────────────────────────────────────────

def run_session(input_01: str, input_02: str, port: int = 8080) -> None:
    """
    呈现层会话入口：在本地浏览器中从 input_01 交互式地走向 input_02。

    路由：
      GET /          渲染当前词条的两圈（调用 get_rings + render_rings）
      GET /choose    接受 ?word=X → save_step → 重定向 /（进入下一轮）
      GET /history   返回 JSON 格式的完整历史记录

    按 Ctrl+C 退出。
    """
    state: dict = {
        "当前":   input_01,
        "终点":   input_02,
        "路径":   [input_01],
        "_cache": None,   # 避免同一步骤对 get_rings 重复调用
    }

    class _Handler(BaseHTTPRequestHandler):

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)

            # ── /choose：记录选择，推进到下一词 ─────────────────────────
            if parsed.path == "/choose":
                params = parse_qs(parsed.query)
                word   = params.get("word", [None])[0]
                rings  = state["_cache"] or get_rings(state["当前"], state["终点"])
                候选   = rings["内圈"] + rings["外圈"]

                if word and word in 候选:
                    save_step(Step(
                        当前=state["当前"],
                        终点=state["终点"],
                        内圈=rings["内圈"],
                        外圈=rings["外圈"],
                        选择=word,
                    ))
                    state["当前"]  = word
                    state["路径"].append(word)
                    state["_cache"] = None   # 触发下一步重新计算

                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()

            # ── /history：返回历史 JSON ────────────────────────────────
            elif parsed.path == "/history":
                body = json.dumps(
                    load_history(), ensure_ascii=False, indent=2
                ).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)

            # ── /：渲染当前两圈 ───────────────────────────────────────
            else:
                if state["_cache"] is None:
                    state["_cache"] = get_rings(state["当前"], state["终点"])
                html = render_rings(
                    state["_cache"],
                    历史=state["路径"],
                    终点=state["终点"],
                )
                self._send(200, "text/html; charset=utf-8", html.encode("utf-8"))

        def log_message(self, fmt: str, *args) -> None:  # 静默服务器日志
            pass

    url = f"http://localhost:{port}"
    print(f"Unknown Nodes → {url}")
    print(f'从 "{input_01}" 走向 "{input_02}"')
    print("按 Ctrl+C 退出")
    webbrowser.open(url)
    HTTPServer(("", port), _Handler).serve_forever()
