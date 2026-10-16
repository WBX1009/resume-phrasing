from __future__ import annotations


RISK_TAG_CN = {
    "night_activity": "夜间交易占比偏高",
    "high_frequency_window": "短时间集中交易",
    "near_threshold_amount": "接近大额阈值交易",
    "many_counterparties": "对手方数量较多",
    "structured_in_scattered_out": "呈现整进散出结构特征",
    "self_transfer_high_density": "同名账户间转账密度较高",
    "same_day_same_amount": "同日同额转账模式",
}


TOOL_NAME_CN = {
    "entity_model_slice": "主体定向切片核查",
    "relation_neighbor_subgraph": "关系邻域探查",
    "fund_flow_path_probe": "资金路径探查",
    "same_amount_window_probe": "同日同额窗口核查",
    "counterparty_overlap_probe": "跨模型对手方重合核查",
}


STATUS_CN = {
    "enhanced": "增强置信度",
    "weakened": "削弱该线索",
    "unresolved": "暂未获得进一步支撑",
    "skipped": "未补证",
    "candidate": "候选线索",
    "success": "成功",
    "fallback": "已降级",
}


AGGREGATION_BASIS_CN = {
    "full_result_set": "完整缓存结果集",
    "entity_dictionary": "实体字典",
    "no_data": "无结果数据",
    "full cached result set; sample rows are only for field profile preview and evidence snippets": (
        "以完整缓存结果集为统计基础；样本行仅用于字段画像预览和证据片段展示"
    ),
}


DEMO_LEVEL_CN = {
    "source": "基础来源",
    "no-data": "无数据分支",
}


def cn_risk_tag(tag: object) -> str:
    value = str(tag or "")
    return RISK_TAG_CN.get(value, value)


def cn_tool_name(name: object) -> str:
    value = str(name or "")
    return TOOL_NAME_CN.get(value, value)


def cn_status(status: object) -> str:
    value = str(status or "")
    return STATUS_CN.get(value, value)


def cn_aggregation_basis(value: object) -> str:
    text = str(value or "")
    return AGGREGATION_BASIS_CN.get(text, text)


def cn_demo_level(value: object) -> str:
    text = str(value or "")
    return DEMO_LEVEL_CN.get(text, text)


def cn_risk_tags(tags: list[object]) -> list[str]:
    return [cn_risk_tag(tag) for tag in tags if tag]
