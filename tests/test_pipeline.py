"""
行为测试 — tests/test_pipeline.py
两种风味，缺一不可（见 CLAUDE.md）。
"""

import pytest
from pipeline import fetch_neighbors


# —— 风味一：例子型（锚定一条我亲眼核实过的真实事实）——

def test_数据可视化_应该连着数据分析():
    # 已在 Wikipedia 的 "Data visualization" 页面亲眼核实这条链接为真
    result = fetch_neighbors("Data visualization")
    assert "Data analysis" in result


# —— 风味二：不变量型（对任何输入都必须成立，不需要我懂领域）——

@pytest.mark.skip(reason="get_rings 尚未实现")
def test_get_rings_的不变量():
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
