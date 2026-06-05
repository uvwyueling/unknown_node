"""
行为测试 — tests/test_pipeline.py
两种风味，缺一不可（见 CLAUDE.md）。
"""

import pytest
from pipeline import fetch_neighbors


# —— 风味一：例子型（锚定一条我亲眼核实过的真实事实）——

def test_数据可视化_含朝向终点的桥梁词_不含产品噪音():
    # 已在 Wikipedia 亲眼核实（"Data visualization" 重定向到
    # "Data and information visualization"）：
    #   · "Data science" 在系列侧栏 "Major dimensions" 行里——它是朝向终点
    #     （Convolutional neural network）的真实桥梁词，必须被取到；
    #   · "imc FAMOS"（侧栏 See also 段里手写的小写品牌名，词条标题 "Imc FAMOS"）
    #     是产品噪音，必须被停用词过滤剔除。
    result = fetch_neighbors("Data visualization")
    assert "Data science" in result          # 桥梁词，必须有
    assert "Imc FAMOS" not in result          # 产品噪音，必须没有
    assert "imc FAMOS" not in result


def test_数据可视化_的相关主题里有回归分析():
    # 已在 Wikipedia 亲眼核实：
    # "Data visualization" 重定向到 "Data and information visualization"，
    # 其右侧系列侧栏的 "Related topics" 行里含 "Regression analysis"。
    result = fetch_neighbors("Data visualization")
    assert "Regression analysis" in result


# —— 风味二：不变量型（对任何输入都必须成立，不需要我懂领域）——

def test_fetch_neighbors_无噪音无重复无自指():
    # 不变量：取数层产出应是干净的——
    #   无命名空间噪音（File:/Category:/人名导航框那类）、无重复、不含词条自身。
    result = fetch_neighbors("Data visualization")
    assert result, "策展链接不应为空"
    assert all(":" not in t for t in result)                 # 无命名空间前缀
    assert len(set(result)) == len(result)                   # 无重复
    assert "Data and information visualization" not in result  # 不含自身（重定向后名）


# ── 磁盘缓存测试（纯逻辑，不联网）────────────────────────────────────────────

def test_fetch_neighbors_磁盘缓存_第二次不再上网(monkeypatch, tmp_path):
    # 例子型：第一次未命中 → 上网一次；第二次命中磁盘 → 不再上网，结果一致。
    import pipeline
    monkeypatch.setattr(pipeline, "_CACHE_FILE", tmp_path / "nb.json")
    monkeypatch.setattr(pipeline, "_nb_cache", None)   # 重置内存镜像，强制从临时文件载入

    上网次数 = {"n": 0}
    def 假上网(node):
        上网次数["n"] += 1
        return ["Regression analysis", "Big data"]
    monkeypatch.setattr(pipeline, "_fetch_neighbors_online", 假上网)

    r1 = pipeline.fetch_neighbors("Data visualization")   # miss → 上网
    r2 = pipeline.fetch_neighbors("Data visualization")   # hit  → 磁盘

    assert r1 == r2 == ["Regression analysis", "Big data"]
    assert 上网次数["n"] == 1                       # 只上网了一次
    assert (tmp_path / "nb.json").exists()          # 已落盘


def test_fetch_neighbors_磁盘缓存_重启进程后仍命中(monkeypatch, tmp_path):
    # 不变量：缓存必须跨进程存活——清空内存镜像（模拟重启）后，纯磁盘也能命中，
    #         此时即使"上网"会爆炸，也不该被触发。
    import pipeline
    monkeypatch.setattr(pipeline, "_CACHE_FILE", tmp_path / "nb.json")
    monkeypatch.setattr(pipeline, "_nb_cache", None)
    monkeypatch.setattr(pipeline, "_fetch_neighbors_online",
                        lambda node: ["A", "B", "C"])
    first = pipeline.fetch_neighbors("X")           # 落盘

    # 模拟重启：内存镜像清空，并让"上网"直接失败
    monkeypatch.setattr(pipeline, "_nb_cache", None)
    def 不准上网(node):
        raise AssertionError("缓存应命中，不该再上网")
    monkeypatch.setattr(pipeline, "_fetch_neighbors_online", 不准上网)

    second = pipeline.fetch_neighbors("X")          # 纯磁盘命中
    assert first == second == ["A", "B", "C"]

# ── score_node 测试 ──────────────────────────────────────────────────────────

def test_score_node_数据分析比陶器更接近目标():
    # 从 Data visualization 出发、目标 Machine learning：
    # Data analysis 与两者都强相关，Pottery 与两者几乎无关。
    from pipeline import score_node
    score_相关 = score_node("Data analysis", "Data visualization", "Machine learning")
    score_无关 = score_node("Pottery", "Data visualization", "Machine learning")
    assert score_相关 > score_无关


def test_score_node_返回值在合法区间():
    # 不变量：任何输入的得分都必须在 [0, 1]
    from pipeline import score_node
    s = score_node("Data analysis", "Data visualization", "Machine learning")
    assert 0.0 <= s <= 1.0


# ── save_step / load_history 测试 ───────────────────────────────────────────

def test_save_step_写入后能读回来(monkeypatch, tmp_path):
    # 例子型：save_step 返回选择，load_history 能读回完整记录，时间戳格式正确
    import pipeline
    from pipeline import Step, save_step, load_history
    monkeypatch.setattr(pipeline, "_HISTORY_FILE", tmp_path / "history.jsonl")

    step = Step(
        当前 = "Data visualization",
        终点 = "Machine learning",
        内圈 = ["Data analysis", "Chart"],
        外圈 = ["Statistics", "Python", "Bar chart"],
        选择 = "Data analysis",
    )
    返回值 = save_step(step)

    assert 返回值 == "Data analysis"

    records = load_history()
    assert len(records) == 1
    r = records[0]
    assert r["选择"] == "Data analysis"
    assert r["当前"] == "Data visualization"
    assert r["终点"] == "Machine learning"
    # 时间戳格式：ISO 8601 UTC，结尾 Z
    assert r["时间戳"].endswith("Z") and "T" in r["时间戳"]
    # 选中项不在未选里
    assert "Data analysis" not in r["未选"]


def test_save_step_候选守恒(monkeypatch, tmp_path):
    # 不变量：未选 ∪ {选择} == 候选（无遗漏）；未选 ∩ {选择} == ∅（无重复）
    import pipeline
    from pipeline import Step, save_step, load_history
    monkeypatch.setattr(pipeline, "_HISTORY_FILE", tmp_path / "history.jsonl")

    step = Step(
        当前 = "A",
        终点 = "Z",
        内圈 = ["B", "C"],
        外圈 = ["D", "E", "F"],
        选择 = "D",
    )
    save_step(step)
    r = load_history()[0]

    assert r["选择"] not in r["未选"]
    assert set(r["未选"]) | {r["选择"]} == set(r["内圈"] + r["外圈"])


# ── render_rings 测试 ────────────────────────────────────────────────────────

def test_render_rings_包含所有词条():
    # 例子型：渲染 HTML 必须出现中心词、内圈、外圈的每一个词，以及目标词
    from pipeline import render_rings
    rings = {
        "中心": "Data visualization",
        "内圈": ["Data analysis", "Chart"],
        "外圈": ["Statistics", "Python", "Bar chart"],
    }
    html = render_rings(rings, 终点="Machine learning")
    assert "Data visualization" in html
    assert "Machine learning"   in html
    for word in rings["内圈"] + rings["外圈"]:
        assert word in html, f"词条 {word!r} 未出现在 HTML 中"


def test_render_rings_结构完整且有点击回调():
    # 不变量：输出是完整 HTML 文档；每个圈内词条都在 HTML 里；choose() 回调存在
    from pipeline import render_rings
    rings = {"中心": "A", "内圈": ["B", "C"], "外圈": ["D", "E", "F"]}
    html  = render_rings(rings)
    assert "<!DOCTYPE html>" in html
    assert "</html>"          in html
    assert "choose("          in html        # 点击回调已注入
    for word in ["A", "B", "C", "D", "E", "F"]:
        assert word in html, f"词条 {word!r} 未出现在 HTML 中"


def test_render_rings_点击绑定不被引号破坏():
    # 回归守卫：曾用内联 onclick="choose("带空格的词")"，里层双引号会提前闭合属性，
    # 导致点击失效。现改为 data-word + 事件委托。本测试钉死这个修复。
    from pipeline import render_rings
    rings = {"中心": "Center", "内圈": ["Visual perception"], "外圈": ["Big data"]}
    html  = render_rings(rings)

    # 1) 不再有内联 onclick，也不能出现断裂的 choose("…
    assert "onclick=" not in html
    assert 'choose("' not in html
    # 2) 可点击词条都带 data-word，且空格词条原样保留在属性里
    assert 'data-word="Visual perception"' in html
    assert 'data-word="Big data"' in html
    # 3) 脚本里有事件委托绑定
    assert "addEventListener" in html
    assert "button[data-word]" in html


def test_get_rings_节点均来自Wikipedia真实链接():
    # 例子型：CLAUDE.md 红线 2——编排层不能发明节点，
    # 所有返回词条必须是 fetch_neighbors 的真实子集。
    from pipeline import fetch_neighbors, get_rings
    result   = get_rings("Data visualization", "Machine learning")
    真实邻居  = set(fetch_neighbors("Data visualization"))
    for 节点 in result["内圈"] + result["外圈"]:
        assert 节点 in 真实邻居, f"发明了节点：{节点!r}"


def test_get_rings_的不变量():
    # 不变量型：对任何输入，结构承诺必须成立
    from pipeline import get_rings
    result = get_rings("Data visualization", "Convolutional neural network")

    # 形状：该有的 key 都在
    assert "内圈" in result and "外圈" in result and "中心" in result

    内圈, 外圈, 中心 = result["内圈"], result["外圈"], result["中心"]

    # 中心词不在任何一圈里
    assert 中心 not in 内圈 and 中心 not in 外圈
    # 两圈之间、各圈内部，都不重复
    全部 = 内圈 + 外圈
    assert len(set(全部)) == len(全部)
    # 内圈不大于外圈（这是对自身设计的承诺）
    assert len(内圈) <= len(外圈)
