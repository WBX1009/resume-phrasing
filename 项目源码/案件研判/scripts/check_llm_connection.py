from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config


def _mask(value: str | None) -> str:
    if not value:
        return "missing"
    return f"present(len={len(value)})"


def main() -> int:
    config = load_config()
    llm = config.llm

    print("=" * 60)
    print("LLM 连接诊断")
    print("=" * 60)
    print(f"provider:       {llm.provider}")
    print(f"api_key:        {_mask(llm.api_key)}")
    print(f"base_url:       {llm.base_url or 'missing'}")
    print(f"model:          {llm.model}")
    print(f"max_tokens:     {llm.max_tokens}")
    print(f"temperature:    {llm.temperature}")
    print(f"is_complete:    {llm.is_complete}")
    print()

    if not llm.is_complete:
        print("FAIL: LLM 配置不完整，请检查 .env 中的以下变量：")
        print("  QIANFAN_API_KEY (或 OPENAI_API_KEY)")
        print("  QIANFAN_OPENAI_BASE_URL (或 OPENAI_BASE_URL)")
        print("  QIANFAN_DEFAULT_MODEL (或 OPENAI_MODEL_NAME)")
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("FAIL: openai 包未安装。请执行: pip install openai")
        return 2

    client = OpenAI(
        api_key=llm.api_key,
        base_url=llm.base_url,
        timeout=60.0,
    )

    # 第1步：基础连通性测试（低 token，快速失败）
    print("--- 测试 1: 基础连通性 (max_tokens=256) ---")
    try:
        start = time.time()
        resp = client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": "回复 OK"}],
            max_tokens=256,
            temperature=0,
        )
        elapsed = time.time() - start
        content = resp.choices[0].message.content or ""
        reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
        usage = resp.usage

        print(f"  状态:       成功")
        print(f"  耗时:       {elapsed:.1f}s")
        print(f"  model:      {resp.model}")
        print(f"  content:    '{content[:200]}'")
        if reasoning:
            print(f"  reasoning_content 长度: {len(reasoning)}")
        if usage:
            print(f"  prompt_tokens:     {usage.prompt_tokens}")
            print(f"  completion_tokens: {usage.completion_tokens}")
            if hasattr(usage, "reasoning_tokens"):
                print(f"  reasoning_tokens:  {usage.reasoning_tokens}")
            print(f"  total_tokens:      {usage.total_tokens}")
    except Exception as exc:
        print(f"  状态: 失败")
        print(f"  错误类型: {type(exc).__name__}")
        print(f"  错误信息: {exc}")
        print()
        print("可能原因：")
        print("  1. API Key 无效或过期")
        print("  2. base_url 不正确")
        print("  3. 网络不通（防火墙/代理）")
        print("  4. model 名称不被该端点支持")
        return 2

    print()

    # 第2步：接近生产场景的测试
    print(f"--- 测试 2: 模拟报告生成场景 (max_tokens={llm.max_tokens}) ---")
    test_prompt = (
        "你是模型结果研判报告助手。只能使用给定数据中的事实。"
        "请输出 JSON，键为 summary，值为一句简短的中文摘要。"
    )
    test_input = (
        '{"case_id":13,"models":['
        '{"model_id":43,"model_name":"★大额存取现","focus_entities":['
        '{"entity_name":"曹小红","metrics":{"record_count":113,"amount_sum":76294380.0}}'
        "]}]}"
    )

    try:
        start = time.time()
        resp = client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": test_prompt},
                {"role": "user", "content": test_input},
            ],
            max_tokens=llm.max_tokens,
            temperature=llm.temperature,
        )
        elapsed = time.time() - start
        content = resp.choices[0].message.content or ""
        reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
        usage = resp.usage

        print(f"  状态:       成功")
        print(f"  耗时:       {elapsed:.1f}s")
        if reasoning:
            print(f"  reasoning_content 长度: {len(reasoning)}")
        print(f"  content:    '{content[:300]}'")
        if usage:
            print(f"  prompt_tokens:     {usage.prompt_tokens}")
            print(f"  completion_tokens: {usage.completion_tokens}")
            if hasattr(usage, "reasoning_tokens"):
                print(f"  reasoning_tokens:  {usage.reasoning_tokens}")
            print(f"  total_tokens:      {usage.total_tokens}")

        if not content.strip():
            print()
            print("WARNING: message.content 为空！")
            print("  这通常是因为 reasoning_tokens 消耗了全部 max_tokens 预算。")
            print("  解决方法：增大 .env 中的 LLM_DEFAULT_MAX_TOKENS（如 8192 或 16384）。")
            print("  或者：如果不需要推理过程，检查千帆 API 是否支持关闭 reasoning。")
            if reasoning:
                print(f"  reasoning_content 预览: '{str(reasoning)[:500]}'")

        print()
        print("=" * 60)
        print("诊断通过：LLM 配置正确，连接可用。")
        print("如果测试2的 content 为空，请增大 LLM_DEFAULT_MAX_TOKENS 后重试。")
        return 0

    except Exception as exc:
        print(f"  状态: 失败")
        print(f"  错误类型: {type(exc).__name__}")
        print(f"  错误信息: {exc}")
        print()
        err_str = str(exc).lower()
        if "timeout" in err_str:
            print("诊断：请求超时。请检查：")
            print("  1. 网络是否能访问 base_url")
            print("  2. 是否需要配置代理")
            print("  3. 当前 timeout 为 60s，如果模型本身响应慢，可联系千帆确认")
        elif "401" in err_str or "unauthorized" in err_str:
            print("诊断：认证失败。请检查 API Key 是否正确。")
        elif "404" in err_str or "not found" in err_str:
            print("诊断：端点或模型不存在。请检查 base_url 和 model 名称。")
        elif "429" in err_str:
            print("诊断：请求频率超限。请稍后重试。")
        else:
            print("诊断：未知错误，请根据上述错误信息排查。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())