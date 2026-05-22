"""memory/quality_checker.py — 语义记忆检索质量评估。

三个核心指标：
1. 相关度：字符级 Jaccard 相似度，支持中文（向量嵌入可用时可替换为 cosine 相似度）。
2. 时间衰减：基于 Ebbinghaus 遗忘曲线的时间衰减因子。
3. 完整度：检索结果对查询关键词的覆盖率。
"""
from __future__ import annotations

import math
import re
from datetime import datetime, UTC
from typing import Any


def calculate_relevance(query: str, retrieved_text: str) -> float:
    """计算相关度分（字符级 Jaccard 相似度，支持中文）。

    原 \w+ 正则无法分割中文导致相似度恒为 0，已改为字符级集合匹配。
    """
    if not query.strip() or not retrieved_text.strip():
        return 0.0

    # 使用字符级集合，有效支持中文文本匹配
    q_tokens = set(query.lower())
    r_tokens = set(retrieved_text.lower())

    # 过滤空白字符
    q_tokens.discard(' ')
    r_tokens.discard(' ')

    if not q_tokens or not r_tokens:
        return 0.0
        
    intersection = len(q_tokens & r_tokens)
    union = len(q_tokens | r_tokens)
    
    return intersection / union if union > 0 else 0.0


def calculate_recency_decay(created_at_iso: str, decay_lambda: float = 0.1, activation: float = 1.0) -> float:
    """基于改进型双曲线衰减计算时间衰减因子。

    相比指数衰减，双曲线衰减对长期记忆更友好，保留更多历史重要节点。
    引入 activation 补偿：高激活/高频访问节点衰减更慢。
    公式：activation / (1 + decay_lambda * days_since)
    """
    try:
        created = datetime.fromisoformat(created_at_iso)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        
        now = datetime.now(UTC)
        days_since = max(0.0, (now - created).total_seconds() / 86400)
        
        # 双曲线衰减 + 激活补偿
        return activation / (1.0 + decay_lambda * days_since)
    except Exception:
        return 0.5  # 时间格式解析失败时返回中性默认值


def check_completeness(query: str, retrieved_memories: list[dict[str, Any]]) -> dict[str, Any]:
    """检查检索结果对查询关键词的覆盖率。

    返回包含覆盖率和未覆盖关键词的字典。
    """
    if not query.strip():
        return {"coverage": 1.0, "missing_keywords": []}

    # 提取查询中的有效关键词（可按需添加停用词过滤）
    query_keywords = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', query.lower()))

    if not query_keywords:
        return {"coverage": 1.0, "missing_keywords": []}

    # 汇总检索结果中所有 token
    retrieved_tokens: set[str] = set()
    for mem in retrieved_memories:
        text = f"{mem.get('title', '')} {mem.get('body', '')}"
        retrieved_tokens.update(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', text.lower()))

    # 计算覆盖率
    covered_keywords = query_keywords & retrieved_tokens
    missing_keywords = list(query_keywords - covered_keywords)

    coverage_ratio = len(covered_keywords) / len(query_keywords)
    
    return {
        "coverage": coverage_ratio,
        "missing_keywords": missing_keywords,
        "total_keywords": len(query_keywords),
        "covered_count": len(covered_keywords)
    }


def evaluate_retrieval_quality(
    query: str,
    retrieved_memories: list[dict[str, Any]],
    decay_lambda: float = 0.1,
    *,
    w_rel: float = 0.45,
    w_comp: float = 0.35,
    w_rec: float = 0.20,
) -> dict[str, Any]:
    """综合评估检索质量：相关度 + 完整度 + 时间新近度加权合成。

    优化权重：降低纯时间衰减的绝对主导，提升完整度权重以保留上下文。
    默认：w_rel=0.45, w_comp=0.35, w_rec=0.20。支持动态传入。
    """
    if not retrieved_memories:
        return {
            "overall_score": 0.0,
            "relevance": 0.0,
            "avg_recency": 0.0,
            "completeness": 0.0,
            "details": "无检索结果",
        }
        
    # 1. 相关度（各结果平均）
    relevances: list[float] = []
    for mem in retrieved_memories:
        text = f"{mem.get('title', '')} {mem.get('body', '')}"
        rel = calculate_relevance(query, text)
        relevances.append(rel)
    avg_relevance = sum(relevances) / len(relevances) if relevances else 0.0

    # 2. 时间新近度（衰减因子平均，引入激活值补偿）
    recencies: list[float] = []
    for mem in retrieved_memories:
        created_at = str(mem.get("created_at", ""))
        if created_at:
            activation = float(mem.get("activation", 1.0))
            recencies.append(calculate_recency_decay(created_at, decay_lambda, activation))
    avg_recency = sum(recencies) / len(recencies) if recencies else 0.0

    # 3. 完整度
    completeness_data = check_completeness(query, retrieved_memories)
    completeness_score = float(completeness_data["coverage"])

    # 加权合成：相关度 > 完整度 > 时间新近度
    overall_score = (w_rel * avg_relevance) + (w_comp * completeness_score) + (w_rec * avg_recency)
    
    return {
        "overall_score": round(overall_score, 4),
        "metrics": {
            "relevance": round(avg_relevance, 4),
            "recency": round(avg_recency, 4),
            "completeness": round(completeness_score, 4)
        },
        "completeness_details": completeness_data,
        "result_count": len(retrieved_memories)
    }
