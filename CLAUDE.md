# 🔍 Unknown nodes - Project Constitution

## 产品宣言
恢复用户主观能动性：在算法推荐时代，把找答案的指南针还给用户。
系统只把菜单往 input_02 那边倾斜，通往 input_02 的路径永远由用户来决定。

## 项目概述
用户输入已知的领域词（input_01）和想探索的新领域词（input_02） 
→ 基于 Wikipedia 真实链接，生成可视化的两圈相关词条  
→ 用户逐步选词，亲自走出从 input_01 到 input_02 的路径。
核心承诺：所有词条都是 Wikipedia 上真实存在的连接，不是 AI 凭空生成的。

## 技术栈
- 后端（五层逻辑）：Python
- 前端（呈现层）：HTML / JS
- 数据源：MediaWiki API（实时取「词条链接的页面」+「See also」，不下载 dump）


## 五层架构（本表是唯一真相源；改架构，先改这张表）
| 层 | 内部操作 | 函数契约 | 传出 → 下一层 |
|--|--|--|--|
| 取数层 | 调 MediaWiki API（取渲染后 HTML），取三处链接：① See also 段；② 信息框/系列侧栏的所有**概念行**（Major dimensions / Related topics / Fields 等，**排除人物 figures 行**）；③ 导语段(lead)正文。再过一道停用词过滤（剔除 `List of…` 前缀、明显产品名、`(software)`/`(company)` 后缀，及策展列表里手写的小写品牌名）。**不取全部页面链接。** | `fetch_neighbors(node) -> [str, ...]` | 邻居列表 |
| 打分层 | 每个邻居算 rel = w1×A + w2×B（A=候选与当前的分类 Jaccard 重叠；B=到 input_02 的分类树距离）。两维都只用已批量缓存的分类，不逐个下载候选整页 | `score_node(候选, 当前, 终点) -> float` | [{词, rel}, ...] |
| 编排层 | 用 MMR 循环挑选（多样性在此发力，候选间相似度同样用分类 Jaccard），切内圈 / 外圈。全部邻居都进入打分，无候选截断 | `get_rings(当前, 终点) -> {中心, 内圈:[...], 外圈:[...]}` | 两圈 |
| 记忆层 | 记录：给了哪些候选、用户选了什么、没选什么 | `save_step(step)` | 用户的选择 |
| 呈现层 | 渲染中心 + 两圈，接收点击 | 前端组件（非 Python） | 被点的词 → 取数层（第 n+1 次循环）|

注：input_02（终点）作为参数贯穿打分层与编排层；取数层不使用它，只向下传递。


## 四条红线（不可违反）
1. 层不许越界：上层只能通过下层的函数契约拿数据。`get_rings` 不许直接连 Wikipedia，必须通过 `fetch_neighbors`。
2. 不许发明节点：打分层、编排层只能操作取数层给出的真实词条。（将来打分层接 LLM 时，必须用代码核对 LLM 返回的每个词都在传入列表里，不在的丢弃——这道栅栏放在编排层入口。）
3. 外部输入先验证再用：Wikipedia 返回可能为空 / 报错 / 上百条，处理后再用；错误包装成友好提示，不把原始报错丢给用户。
4. 时间戳一律存 UTC 的 ISO 8601 字符串(datetime.now(timezone.utc).isoformat());本地时间只在呈现层显示时转换。

## PEV 工作流
1. Plan：列出输入 / 输出 / 错误场景
2. Execute：一次只实现一个函数契约，不许一口气写整条 pipeline
3. Verify（两道闸，都要过）：
   - 行为：跑 `pytest`，行为测试必须通过。
     例：`fetch_neighbors("Data visualization")` 的返回里必须含 `"Data science"`（朝向终点的真实桥梁词），且必须不含 `"Imc FAMOS"`（产品噪音）——已在 Wikipedia 亲眼核实。
   - 构建：前端跑 `npm run build`，失败则修复直到通过。
   - build 通过只是底线，行为测试通过才算"做对"。

## 行为测试（住在 tests/test_pipeline.py；Verify 的「行为闸」）

测试有两种风味，下面各一个，缺一不可：

```python
# —— 风味一：例子型（锚定一条我亲眼核实过的真实事实）——
def test_数据可视化_含桥梁词_不含产品噪音():
    # 已在 Wikipedia 亲眼核实（"Data visualization" 重定向到 "Data and information visualization"）：
    # "Data science" 在侧栏 Major dimensions 行（朝向终点的桥梁），必须取到；
    # "Imc FAMOS" 是产品噪音，必须被停用词过滤剔除。
    result = fetch_neighbors("Data visualization")
    assert "Data science" in result
    assert "Imc FAMOS" not in result


# —— 风味二：不变量型（对任何输入都必须成立，不需要我懂领域）——
def test_get_rings_的不变量():
    result = get_rings("Data visualization", "Convolutional neural network")

    # 形状：该有的 key 都在
    assert "内圈" in result and "外圈" in result

    内圈, 外圈, 中心 = result["内圈"], result["外圈"], result["中心"]

    # 中心词不在任何一圈里
    assert 中心 not in 内圈 and 中心 not in 外圈
    # 两圈之间、各圈内部，都不重复
    全部 = 内圈 + 外圈
    assert len(set(全部)) == len(全部)
    # 内圈不大于外圈（这是对自身设计的承诺）
    assert len(内圈) <= len(外圈)
```

## 禁止事项
- 硬编码 API 密钥
- 跳过 Verify 直接标记完成
- 让 Claude Code 一次性生成整条 pipeline（必须逐个函数、逐个验）