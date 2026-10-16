import json
import os
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
import streamlit as st


API_BASE_URL = os.getenv("DEEPASSIST_API_BASE", "http://localhost:8000/api").rstrip("/")
DEFAULT_COLLECTION = "tech_docs_kb"
RAG_COLLECTION = "__all__"
REQUEST_TIMEOUT = int(os.getenv("DEEPASSIST_UI_TIMEOUT", "180"))


st.set_page_config(page_title="DeepAssist", page_icon="D", layout="wide")

st.markdown(
    """
<style>
    #MainMenu, footer, .stAppDeployButton {visibility: hidden;}
    .block-container {padding: 1.4rem 2rem 1rem 2rem; max-width: 100%;}
    section[data-testid="stSidebar"] {background: #f7f8fb; border-right: 1px solid #e9edf3;}
    section[data-testid="stSidebar"] .block-container {padding: 1.2rem 1.1rem;}
    div[data-testid="stVerticalBlock"] {gap: 0.55rem;}
    .brand {display: flex; align-items: center; gap: .6rem; font-weight: 800; font-size: 1.25rem; color: #111827;}
    .brand-mark {width: 30px; height: 30px; border: 1px solid #111827; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: .78rem; font-weight: 800;}
    .muted {color: #6b7280; font-size: .9rem;}
    .tiny {color: #8b95a5; font-size: .78rem;}
    .page-title {font-size: 2rem; line-height: 1.15; font-weight: 850; color: #111827; margin: 0;}
    .page-subtitle {color: #6b7280; margin-top: .4rem;}
    .section-title {font-size: 1.18rem; font-weight: 800; color: #111827; margin: 0 0 .4rem 0;}
    .mode-card {border: 1px solid #e6eaf0; border-radius: 8px; padding: 1.35rem 1.45rem; background: #fff; transition: border-color .15s ease, box-shadow .15s ease;}
    .mode-card:hover {border-color: #9bbcff; box-shadow: 0 10px 26px rgba(36, 99, 235, .08);}
    .mode-row {display: flex; align-items: center; gap: 1rem;}
    .mode-icon {width: 52px; height: 52px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #0f172a; font-weight: 850; letter-spacing: 0;}
    .mode-title {font-size: 1.12rem; font-weight: 800; color: #111827; margin-bottom: .25rem;}
    .mode-desc {color: #6b7280; font-size: .95rem;}
    .panel {border: 1px solid #e6eaf0; border-radius: 8px; padding: 1.1rem 1.25rem; background: #fff;}
    .header-card {border: 1px solid #e6eaf0; border-radius: 8px; padding: 1.25rem 1.35rem; background: #fff;}
    .metric-pill {display:inline-flex; align-items:center; gap:.35rem; border:1px solid #e6eaf0; border-radius:999px; padding:.22rem .55rem; color:#526071; font-size:.78rem; margin-right:.35rem;}
    .trace-box {border-left: 3px solid #d6e4ff; padding-left: .75rem; color: #526071; font-size: .88rem;}
    .status-ok {color:#0f8a5f; font-weight:700;}
    .status-blocked {color:#b45309; font-weight:700;}
    .status-bad {color:#b91c1c; font-weight:700;}
    .error-card {border:1px solid #fecaca; border-left:4px solid #ef4444; border-radius:8px; padding:.85rem 1rem; background:#fff7f7;}
    .stButton button {border-radius: 8px; min-height: 2.35rem; white-space: normal; line-height: 1.2;}
    section[data-testid="stSidebar"] .stButton button {font-size:.88rem; padding:.35rem .45rem;}
    .stTextInput input, .stSelectbox div[data-baseweb="select"], .stNumberInput input {border-radius: 8px;}
    div[data-testid="stChatMessage"] {border-radius: 8px;}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "nav_tab": "assistant",
        "session_id": f"sess_{uuid.uuid4().hex[:8]}",
        "chat_mode": "conversation",
        "messages": [],
        "show_home": True,
        "selected_model": "deepseek-chat",
        "temperature": 0.7,
        "top_p": 1.0,
        "history_rounds": 10,
        "use_user_memory": False,
        "show_settings": False,
        "kb_collection": DEFAULT_COLLECTION,
        "rag_collection_scope": RAG_COLLECTION,
        "kb_confirm_delete_all": False,
        "runtime_capabilities": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.selected_model = {
        "DeepSeek Chat": "deepseek-chat",
        "DeepSeek Reasoner": "deepseek-reasoner",
        "智谱 GLM-4.6": "deepseek-chat",
    }.get(str(st.session_state.selected_model), str(st.session_state.selected_model))


def api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def request_json(method: str, path: str, **kwargs) -> Dict[str, Any]:
    try:
        response = requests.request(method, api_url(path), timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        return {"status": "error", "message": str(exc), "data": []}

    try:
        payload = response.json()
    except ValueError:
        payload = {"status": "error", "message": response.text, "data": []}
    if response.status_code >= 400:
        payload.setdefault("status", "error")
        payload.setdefault("message", f"HTTP {response.status_code}")
    return payload


def fetch_sessions() -> List[Dict[str, Any]]:
    payload = request_json("GET", "/chat/sessions")
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def fetch_history(session_id: str) -> List[Dict[str, Any]]:
    payload = request_json("GET", f"/chat/history/{quote(session_id, safe='')}")
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def delete_session(session_id: str) -> Dict[str, Any]:
    return request_json("DELETE", f"/chat/sessions/{quote(session_id, safe='')}")


def fetch_kb_collections() -> Dict[str, Any]:
    return request_json("GET", "/kb/collections")


def fetch_kb_files(collection_name: Optional[str] = None) -> Dict[str, Any]:
    return request_json(
        "GET",
        "/kb/files",
        params={"collection_name": collection_name or st.session_state.kb_collection},
    )


def fetch_kb_health(refresh: bool = False) -> Dict[str, Any]:
    return request_json(
        "GET",
        "/kb/health",
        params={"refresh": refresh},
    )


def fetch_runtime_capabilities() -> Dict[str, Any]:
    return request_json("GET", "/runtime/capabilities")


def fetch_runtime_diagnostics(refresh: bool = False) -> Dict[str, Any]:
    return request_json("GET", "/runtime/diagnostics", params={"refresh": refresh})


def delete_kb_file(source_file: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
    encoded = quote(source_file, safe="")
    return request_json(
        "DELETE",
        f"/kb/files/{encoded}",
        params={"collection_name": collection_name or st.session_state.kb_collection},
    )


def upload_kb_file(uploaded_file, collection_name: Optional[str] = None) -> Dict[str, Any]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "text/markdown",
        )
    }
    return request_json(
        "POST",
        "/kb/upload",
        params={"collection_name": collection_name or st.session_state.kb_collection},
        files=files,
    )


def create_new_chat() -> None:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.session_state.show_home = True
    st.session_state.chat_mode = "conversation"


def load_session(session_id: str) -> None:
    history = fetch_history(session_id)
    st.session_state.session_id = session_id
    st.session_state.messages = history
    st.session_state.chat_mode = "conversation"
    st.session_state.show_home = False
    st.session_state.nav_tab = "history"


def summarize(text: str, length: int = 34) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "新会话"
    return text if len(text) <= length else f"{text[:length]}..."


def backend_model_name() -> str:
    selected = str(st.session_state.get("selected_model", "deepseek-chat"))
    legacy_map = {
        "DeepSeek Chat": "deepseek-chat",
        "DeepSeek Reasoner": "deepseek-reasoner",
        "智谱 GLM-4.6": "deepseek-chat",
    }
    return legacy_map.get(selected, selected or "deepseek-chat")


def get_runtime_capabilities(refresh: bool = False) -> Dict[str, Any]:
    cached = st.session_state.get("runtime_capabilities")
    if refresh or not isinstance(cached, dict) or not cached:
        payload = fetch_runtime_capabilities()
        data = payload.get("data", {})
        st.session_state.runtime_capabilities = data if isinstance(data, dict) else {}
    return st.session_state.get("runtime_capabilities", {})


def runtime_models() -> List[Dict[str, Any]]:
    models = get_runtime_capabilities().get("models", [])
    if isinstance(models, list) and models:
        return [item for item in models if isinstance(item, dict) and item.get("id")]
    return [
        {"id": "deepseek-chat", "label": "DeepSeek Chat", "supported": True},
        {"id": "deepseek-reasoner", "label": "DeepSeek Reasoner", "supported": True},
    ]


def runtime_tools() -> List[Dict[str, Any]]:
    tools = get_runtime_capabilities().get("tools", [])
    return tools if isinstance(tools, list) else []


def runtime_kb_status() -> Dict[str, Any]:
    kb = get_runtime_capabilities().get("knowledge_base", {})
    return kb if isinstance(kb, dict) else {}


def runtime_mode_specs() -> Dict[str, Dict[str, Any]]:
    items = get_runtime_capabilities().get("modes", [])
    if not isinstance(items, list):
        return {}
    return {
        str(item.get("id")): item
        for item in items
        if isinstance(item, dict) and item.get("id")
    }


def runtime_kb_scope_options() -> List[Dict[str, Any]]:
    kb = get_runtime_capabilities().get("knowledge_base", {})
    if not isinstance(kb, dict):
        return []
    options = kb.get("rag_scope_options", [])
    return options if isinstance(options, list) else []


def selected_model_label() -> str:
    current = backend_model_name()
    for item in runtime_models():
        if item.get("id") == current:
            return str(item.get("label") or current)
    return current


def current_rag_scope() -> str:
    scope = str(st.session_state.get("rag_collection_scope", RAG_COLLECTION) or RAG_COLLECTION)
    options = {
        str(item.get("id"))
        for item in runtime_kb_scope_options()
        if isinstance(item, dict) and item.get("id")
    }
    if options and scope not in options:
        return RAG_COLLECTION
    return scope


def current_rag_scope_label() -> str:
    scope = current_rag_scope()
    for item in runtime_kb_scope_options():
        if isinstance(item, dict) and item.get("id") == scope:
            return str(item.get("label") or scope)
    return scope


def runtime_payload() -> Dict[str, Any]:
    return {
        "model_name": backend_model_name(),
        "temperature": float(st.session_state.temperature),
        "top_p": float(st.session_state.top_p),
        "history_rounds": int(st.session_state.history_rounds),
        "use_user_memory": bool(st.session_state.use_user_memory),
    }


def request_contract_preview(prompt: str = "<user query>") -> Dict[str, Any]:
    return {
        "endpoint": "/api/conversation/stream",
        "payload": {
            "session_id": st.session_state.session_id,
            "query": prompt,
            "collection_name": current_rag_scope(),
            **runtime_payload(),
        },
    }


def render_brand() -> None:
    st.markdown(
        """
<div class="brand">
  <div class="brand-mark">DA</div>
  <div>DeepAssist</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        render_brand()
        st.divider()

        tab_cols = st.columns(3)
        tabs = [("assistant", "对话"), ("history", "历史"), ("kb", "知识库")]
        for col, (key, label) in zip(tab_cols, tabs):
            if col.button(
                label,
                use_container_width=True,
                type="primary" if st.session_state.nav_tab == key else "secondary",
            ):
                st.session_state.nav_tab = key
                st.rerun()

        st.divider()

        if st.session_state.nav_tab in {"assistant", "history"}:
            if st.button("新建会话", use_container_width=True, type="primary"):
                create_new_chat()
                st.rerun()

        if st.session_state.nav_tab == "assistant":
            render_sidebar_agent_controls()
        elif st.session_state.nav_tab == "history":
            render_sidebar_history()
        else:
            render_sidebar_kb_v2()


def render_sidebar_agent_controls() -> None:
    model_items = runtime_models()
    model_ids = [str(item.get("id")) for item in model_items]
    current = backend_model_name()
    if current not in model_ids and model_ids:
        st.session_state.selected_model = model_ids[0]

    st.caption(f"当前会话：`{st.session_state.session_id}`")

    with st.expander("模型与生成参数", expanded=False):
        st.selectbox(
            "模型",
            model_ids,
            key="selected_model",
            format_func=lambda model_id: next(
                (
                    str(item.get("label"))
                    for item in model_items
                    if item.get("id") == model_id
                ),
                model_id,
            ),
        )
        st.slider("Temperature", 0.0, 2.0, key="temperature", step=0.1)
        st.slider("Top P", 0.0, 1.0, key="top_p", step=0.05)
        st.slider("历史轮数", 1, 30, key="history_rounds")
        st.checkbox("启用长期记忆", key="use_user_memory")

    kb_status = runtime_kb_status()
    rag_badge = "可用" if kb_status.get("rag_available") else "需检查"
    with st.expander(f"知识库范围：{current_rag_scope_label()} / {rag_badge}", expanded=False):
        scope_options = runtime_kb_scope_options()
        scope_ids = [str(item.get("id")) for item in scope_options if isinstance(item, dict)]
        if scope_ids:
            if current_rag_scope() not in scope_ids:
                st.session_state.rag_collection_scope = scope_ids[0]
            st.selectbox(
                "检索范围",
                scope_ids,
                key="rag_collection_scope",
                format_func=lambda scope_id: next(
                    (
                        str(item.get("label"))
                        for item in scope_options
                        if item.get("id") == scope_id
                    ),
                    scope_id,
                ),
            )
        if kb_status.get("rag_available"):
            st.markdown("<span class='status-ok'>知识库问答链路可用</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-bad'>知识库问答链路需要诊断</span>", unsafe_allow_html=True)
            st.caption(str(kb_status.get("rag_unavailable_reason") or "RAG 状态未知"))

    with st.expander("工具与权限状态", expanded=False):
        tools = runtime_tools()
        if not tools:
            st.info("当前没有工具元数据。")
        else:
            for tool in tools:
                name = str(tool.get("name") or "tool")
                description = str(tool.get("description") or "")
                category = str(tool.get("category") or "unknown")
                enabled = bool(tool.get("enabled"))
                state = "可用" if enabled else "已禁用"
                css_class = "status-ok" if enabled else "status-blocked"
                args = tool.get("parameters", []) or []
                required = [str(item.get("name")) for item in args if item.get("required")]
                required_text = ", ".join(required) if required else "无"
                st.markdown(
                    f"- `{name}` · `{category}` · <span class='{css_class}'>{state}</span>",
                    unsafe_allow_html=True,
                )
                if not enabled:
                    st.caption(str(tool.get("blocked_reason") or "被当前工具策略阻止"))
                else:
                    st.caption(f"{description} | 必填参数：{required_text}")

    with st.expander("运行时诊断", expanded=False):
        if st.button("刷新诊断", use_container_width=True):
            st.session_state.runtime_capabilities = {}
            diag = fetch_runtime_diagnostics(refresh=True)
        else:
            diag = fetch_runtime_diagnostics(refresh=False)
        data = diag.get("data", {}) if isinstance(diag, dict) else {}
        rag = data.get("rag", {}) if isinstance(data, dict) else {}
        st.caption(f"RAG：{'可用' if rag.get('available') else '不可用'}")
        if rag.get("unavailable_reason"):
            st.caption(str(rag.get("unavailable_reason")))


def render_sidebar_history() -> None:
    sessions = fetch_sessions()
    st.caption("历史会话")
    if not sessions:
        st.markdown('<div class="muted">暂无聊天记录</div>', unsafe_allow_html=True)
        return

    for session in sessions:
        session_id = session.get("session_id", "")
        title = summarize(session.get("title", ""), 18)
        cols = st.columns([0.78, 0.22])
        if cols[0].button(title, key=f"open_{session_id}", use_container_width=True):
            load_session(session_id)
            st.rerun()
        if cols[1].button("Del", key=f"delete_{session_id}", use_container_width=True):
            result = delete_session(session_id)
            if result.get("status") == "success":
                if st.session_state.session_id == session_id:
                    create_new_chat()
                st.rerun()
            else:
                st.error(result.get("message", "删除失败"))


def render_sidebar_kb() -> None:
    collections_payload = fetch_kb_collections()
    collections = collections_payload.get("data", [])
    collections = collections if isinstance(collections, list) else []
    collection_names = [item.get("collection_name") for item in collections if isinstance(item, dict)]
    collection_names = [name for name in collection_names if name]
    if st.session_state.kb_collection not in collection_names and collection_names:
        st.session_state.kb_collection = collection_names[0]

    selected = st.session_state.kb_collection
    selected_meta = next(
        (item for item in collections if item.get("collection_name") == selected),
        {},
    )

    st.caption("知识库索引")
    if collection_names:
        st.selectbox(
            "当前管理集合",
            collection_names,
            key="kb_collection",
            label_visibility="collapsed",
        )
        selected = st.session_state.kb_collection
        selected_meta = next(
            (item for item in collections if item.get("collection_name") == selected),
            {},
        )
    st.markdown(
        f"""
<div class="panel">
  <div style="font-weight:800;">{selected}</div>
  <div class="tiny">{selected_meta.get('file_count', 0)} 个文件，{selected_meta.get('chunk_count', 0)} 个文本块</div>
  <div style="margin-top:.45rem;">
    <span class="metric-pill">{" + ".join(selected_meta.get('stores', [])) or "未连接"}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption("问答默认使用全部知识库检索；这里仅用于维护单个集合。")
    if st.button("新建知识库", use_container_width=True):
        st.info("多知识库命名空间入口已预留；当前可管理已有集合和默认 tech_docs_kb。")


def render_sidebar_kb_v2() -> None:
    collections_payload = fetch_kb_collections()
    health_payload = fetch_kb_health(refresh=False)

    collections = collections_payload.get("data", [])
    collections = collections if isinstance(collections, list) else []
    health_report = health_payload.get("data", {})
    health_report = health_report if isinstance(health_report, dict) else {}
    health_collections = health_report.get("collections", [])
    health_collections = health_collections if isinstance(health_collections, list) else []

    collection_names = [
        item.get("collection_name")
        for item in collections
        if isinstance(item, dict) and item.get("collection_name")
    ]
    if st.session_state.kb_collection not in collection_names and collection_names:
        st.session_state.kb_collection = collection_names[0]

    st.caption("Knowledge Base")
    if collection_names:
        st.selectbox(
            "Current collection",
            collection_names,
            key="kb_collection",
            label_visibility="collapsed",
        )

    selected = st.session_state.kb_collection
    selected_meta = next(
        (item for item in collections if item.get("collection_name") == selected),
        {},
    )
    selected_health = next(
        (item for item in health_collections if item.get("collection_name") == selected),
        {},
    )

    stores = " + ".join(selected_meta.get("stores", [])) or "Unavailable"
    file_count = int(selected_meta.get("file_count", 0))
    chunk_count = int(selected_meta.get("chunk_count", 0))
    health_status = "Healthy" if selected_health.get("healthy") else "Needs attention"
    checked_at = health_report.get("checked_at", "--")
    errors = selected_health.get("errors", []) or []

    st.markdown(
        f"""
<div class="panel">
  <div style="font-weight:800;">{selected or "No collection selected"}</div>
  <div class="tiny">{file_count} files, {chunk_count} chunks</div>
  <div style="margin-top:.45rem;">
    <span class="metric-pill">{stores}</span>
    <span class="metric-pill">Health: {health_status}</span>
  </div>
  <div class="tiny" style="margin-top:.45rem;">Checked: {checked_at}</div>
  <div class="tiny">Health issues: {len(errors)}</div>
  <div class="tiny">Current Knowledge Q&A scope: {current_rag_scope_label()}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    refresh_disabled = not bool(collection_names)
    if st.button("Refresh KB health", use_container_width=True, disabled=refresh_disabled):
        refreshed = fetch_kb_health(refresh=True)
        if refreshed.get("status") in {"success", "partial_success"}:
            st.success("Knowledge-base health report updated.")
        else:
            st.error(refreshed.get("message", "Knowledge-base health refresh failed."))
        st.rerun()

    with st.expander("Health details", expanded=False):
        st.caption("RAG and Agent retrieval search across all collections by default.")
        if selected_health:
            st.json(selected_health)
        elif collection_names:
            st.info("No cached health data for the selected collection yet.")
        else:
            st.info("No knowledge-base collections found.")


def render_home() -> None:
    st.markdown('<div style="height: 10vh;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="page-title" style="text-align:center;">DeepAssist 智能助理</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle" style="text-align:center;">直接输入需求，系统会自动选择普通问答、知识库检索、工具执行或澄清。</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height: 1.6rem;"></div>', unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.35, 1])
    with center:
        st.markdown(
            """
<div class="mode-card">
  <div class="mode-row">
    <div class="mode-icon" style="background:#1ed6a5;">DA</div>
    <div>
      <div class="mode-title">统一对话入口</div>
      <div class="mode-desc">一个输入框完成问答、知识库检索、工具调用和意图澄清。</div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("开始对话", type="primary", use_container_width=True):
            st.session_state.show_home = False
            st.session_state.chat_mode = "conversation"
            st.rerun()


def render_chat_header() -> None:
    cols = st.columns([0.62, 0.38])
    scope_badge = f'<span class="metric-pill">知识库范围：{current_rag_scope_label()}</span>'
    with cols[0]:
        st.markdown(
            f"""
<div class="header-card">
  <div class="section-title">统一对话</div>
  <div class="muted">系统会按当前输入动态选择普通问答、知识库检索、智能体工具或澄清问题。</div>
  <div style="margin-top:.55rem;">
    <span class="metric-pill">会话：{st.session_state.session_id}</span>
    <span class="metric-pill">模型：{selected_model_label()}</span>
    {scope_badge}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with cols[1]:
        button_cols = st.columns(3)
        if button_cols[0].button("设置", use_container_width=True):
            st.session_state.show_settings = not st.session_state.show_settings
        if button_cols[1].button("首页", use_container_width=True):
            st.session_state.show_home = True
            st.rerun()
        if button_cols[2].button("新建", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()

    if st.session_state.show_settings:
        render_settings_panel()


def render_settings_panel() -> None:
    with st.container(border=True):
        st.markdown("#### 运行时请求")
        st.caption("这些设置会进入统一对话接口；旧 quick/RAG/agent 接口仅保留给调试。")

        left, right = st.columns([0.48, 0.52])
        with left:
            mode_specs = runtime_mode_specs()
            mode_meta = mode_specs.get("conversation", {})
            st.markdown("**当前入口**：`conversation`")
            if mode_meta:
                st.caption(str(mode_meta.get("description") or ""))
                st.code(str(mode_meta.get("endpoint") or "n/a"), language="text")

            request_preview = request_contract_preview()
            st.markdown("**请求预览**")
            st.json(request_preview)

        with right:
            model_items = runtime_models()
            model_ids = [str(item.get("id")) for item in model_items]
            st.selectbox(
                "模型",
                model_ids,
                key="selected_model",
                format_func=lambda model_id: next(
                    (
                        str(item.get("label"))
                        for item in model_items
                        if item.get("id") == model_id
                    ),
                    model_id,
                ),
            )
            st.slider("Temperature", 0.0, 2.0, key="temperature", step=0.1)
            st.slider("Top P", 0.0, 1.0, key="top_p", step=0.05)
            st.slider("历史轮数", 1, 30, key="history_rounds")
            st.checkbox("启用长期记忆", key="use_user_memory")
            scope_options = runtime_kb_scope_options()
            scope_ids = [str(item.get("id")) for item in scope_options if isinstance(item, dict)]
            if scope_ids:
                st.selectbox(
                        "知识库范围",
                    scope_ids,
                    key="rag_collection_scope",
                    format_func=lambda scope_id: next(
                        (
                            str(item.get("label"))
                            for item in scope_options
                            if item.get("id") == scope_id
                        ),
                        scope_id,
                    ),
                )


def render_message(message: Dict[str, Any], index: Optional[int] = None) -> None:
    role = message.get("role")
    content = message.get("content", "")

    if role in {"user", "assistant"}:
        with st.chat_message(role):
            st.markdown(content or "")
            events = message.get("events") or []
            if events:
                render_event_trace_v2(events)
            render_pending_action_controls(message, index)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                with st.expander("Tool call request", expanded=False):
                    st.json(tool_calls)
    elif role == "tool":
        with st.chat_message("assistant"):
            with st.expander(f"Tool result: {message.get('name') or 'tool'}", expanded=False):
                st.markdown(content or "")


def render_pending_action_controls(
    message: Dict[str, Any],
    index: Optional[int] = None,
) -> None:
    pending_action = message.get("pending_action")
    if not isinstance(pending_action, dict):
        return
    action_id = str(pending_action.get("action_id") or "").strip()
    if not action_id or pending_action.get("status") != "pending":
        return

    tool_name = pending_action.get("tool_name") or pending_action.get("name") or "tool"
    st.warning(f"Approval required before executing `{tool_name}`.")
    with st.expander("Pending tool arguments", expanded=False):
        st.json(pending_action.get("args", {}))
    cols = st.columns([0.22, 0.22, 0.56])
    key_prefix = f"{index if index is not None else action_id}_{action_id}"
    if cols[0].button("Approve and resume", key=f"approve_{key_prefix}", type="primary"):
        continuation = stream_pending_action_decision(action_id, approved=True)
        merge_continuation_message(message, continuation)
        pending_action["status"] = "approved"
        st.rerun()
    if cols[1].button("Reject", key=f"reject_{key_prefix}"):
        continuation = stream_pending_action_decision(action_id, approved=False)
        merge_continuation_message(message, continuation)
        pending_action["status"] = "rejected"
        st.rerun()


def merge_continuation_message(
    message: Dict[str, Any],
    continuation: Dict[str, Any],
) -> None:
    current = str(message.get("content") or "")
    new_content = str(continuation.get("content") or "")
    if new_content:
        waiting_prefix = current.startswith("Waiting for approval") or current.startswith("等待人工审批")
        message["content"] = new_content if waiting_prefix else f"{current}\n\n{new_content}".strip()
    message.setdefault("events", []).extend(continuation.get("events") or [])


def render_event_trace(events: List[Dict[str, Any]]) -> None:
    with st.expander("运行轨迹", expanded=False):
        for event in events:
            event_name = event.get("event")
            data = event.get("data", {}) or {}
            if event_name == "supervisor_route":
                st.markdown(
                    f"- 路由：`{data.get('worker_kind') or data.get('worker')}`，"
                    f"意图 `{data.get('intent')}`，置信度 `{data.get('confidence')}`"
                )
            elif event_name == "retrieval_trace":
                st.markdown(
                    f"- 检索：命中 `{data.get('hit_count', 0)}`，"
                    f"候选 `{data.get('candidate_k', 0)}`，融合 `{data.get('fusion', 'n/a')}`"
                )
            elif event_name == "citation_trace":
                st.markdown(f"- 引用打包：`{len(data.get('citations', []))}` 条引用")
            elif event_name == "tool_call":
                st.markdown(f"- 工具调用：`{event.get('name')}`")
                st.json(event.get("args", {}))
            elif event_name == "tool_result":
                ok = data.get("success")
                st.markdown(f"- 工具结果：`{event.get('name')}`，success=`{ok}`")
                if data.get("error"):
                    st.caption(data.get("error"))
            elif event_name == "answer_guard":
                st.markdown(
                    f"- 答案守卫：grounded=`{data.get('grounded')}`，"
                    f"warnings=`{data.get('warnings', [])}`"
                )
            elif event_name == "self_correction":
                st.markdown(f"- 自修正：{event.get('message') or data.get('error')}")
            elif event_name == "reasoning":
                st.markdown("- 模型推理片段")
                st.code(event.get("content", ""), language="text")


def compact_event_summary(events: List[Dict[str, Any]]) -> str:
    if not events:
        return ""
    parts: List[str] = []
    for event in events[-8:]:
        name = event.get("event")
        data = event.get("data", {}) or {}
        if name == "supervisor_route":
            parts.append(f"route:{data.get('worker_kind') or data.get('worker')}")
        elif name == "retrieval_trace":
            parts.append(f"retrieval:{data.get('hit_count', 0)}")
        elif name == "citation_trace":
            parts.append(f"citations:{len(data.get('citations', []))}")
        elif name == "tool_call":
            parts.append(f"tool:{event.get('name')}")
        elif name == "tool_result":
            parts.append(f"result:{event.get('name')}")
        elif name == "answer_guard":
            parts.append("guard")
        elif name == "error":
            parts.append("error")
    return " -> ".join(parts)


def render_event_trace_v2(events: List[Dict[str, Any]]) -> None:
    with st.expander("运行链路", expanded=False):
        for event in events:
            event_name = event.get("event")
            data = event.get("data", {}) or {}
            if event_name == "routing_decision":
                st.markdown(
                    f"- 本轮路由决策：`{data.get('route')}` "
                    f"置信度=`{data.get('confidence')}` 来源策略=`{data.get('source_policy')}`"
                )
                capabilities = data.get("required_capabilities", []) or []
                if capabilities:
                    st.caption(f"所需能力：{', '.join(str(item) for item in capabilities)}")
                if data.get("reason"):
                    st.caption(str(data.get("reason")))
            elif event_name == "capability_status":
                st.markdown(f"- 能力状态：`{data.get('capability', 'unknown')}`")
                if data.get("reason"):
                    st.caption(str(data.get("reason")))
            elif event_name == "rag_unavailable":
                st.markdown("- 知识库问答不可用")
                st.caption(str(data.get("reason") or data.get("component") or "RAG 组件未就绪"))
                if data.get("next_action"):
                    st.caption(str(data.get("next_action")))
            elif event_name == "tool_policy_blocked":
                st.markdown(
                    f"- 工具被策略阻止：`{data.get('tool_name')}` "
                    f"类别=`{data.get('category')}`"
                )
                st.caption(str(data.get("blocked_reason") or "当前策略不允许该工具"))
                if data.get("how_to_enable_for_local_demo"):
                    st.caption(str(data.get("how_to_enable_for_local_demo")))
            elif event_name == "pending_action":
                pending = data.get("pending_action", {}) or {}
                st.markdown(
                    f"- 等待人工审批：`{pending.get('tool_name') or data.get('tool_name')}` "
                    f"类别=`{pending.get('category') or data.get('category')}`"
                )
                if pending.get("action_id"):
                    st.caption(f"action_id={pending.get('action_id')}")
            elif event_name == "approval_result":
                st.markdown(
                    f"- 审批结果：`{data.get('status', 'unknown')}` "
                    f"action_id=`{data.get('action_id', '')}`"
                )
            elif event_name == "agent_runtime_node":
                st.markdown(
                    f"- Runtime node: `{data.get('node', 'unknown')}` "
                    f"status=`{data.get('status', 'running')}` "
                    f"fallback=`{data.get('fallback_without_langgraph')}`"
                )
                if data.get("fallback_reason"):
                    st.caption(str(data.get("fallback_reason")))
            elif event_name == "supervisor_route":
                st.markdown(
                    f"- 智能体内部调度：`{data.get('worker_kind') or data.get('worker')}` "
                    f"意图=`{data.get('intent')}` 置信度=`{data.get('confidence')}`"
                )
            elif event_name == "context_window_trace":
                st.markdown(
                    f"- 上下文窗口：预算=`{data.get('budget', 0)}` "
                    f"选中轮次=`{data.get('selected_turn_count', 0)}` "
                    f"丢弃轮次=`{data.get('dropped_turn_count', 0)}` "
                    f"记忆=`{data.get('recalled_memory_count', 0)}`"
                )
                recalled_memories = data.get("recalled_memories", []) or []
                if recalled_memories:
                    st.markdown("记忆召回")
                    for memory in recalled_memories:
                        st.markdown(
                            f"- [{memory.get('category', 'memory')}] {memory.get('content', '')}"
                        )
                selected_turns = data.get("selected_turns", []) or []
                if selected_turns:
                    st.markdown("选中的历史轮次")
                    for turn in selected_turns[:4]:
                        st.markdown(
                            f"- `{turn.get('turn_id')}` `{turn.get('priority_band')}` "
                            f"score=`{turn.get('priority_score')}` {turn.get('preview', '')}"
                        )
                dropped_turns = data.get("dropped_turns", []) or []
                if dropped_turns:
                    st.markdown("被压缩或丢弃的轮次")
                    for turn in dropped_turns[:3]:
                        st.markdown(
                            f"- `{turn.get('turn_id')}` `{turn.get('priority_band')}` "
                            f"{turn.get('preview', '')}"
                        )
                summary = data.get("summary")
                if isinstance(summary, dict):
                    st.markdown(
                        f"- 已注入摘要：消息=`{summary.get('dropped_message_count', 0)}` "
                        f"轮次=`{summary.get('dropped_turn_count', 0)}`"
                    )
            elif event_name == "multi_agent_plan":
                st.markdown(
                    f"- 多智能体计划：任务=`{data.get('task_count', 0)}` "
                    f"模式=`{data.get('mode', 'sequential_collaboration')}` "
                    f"复杂度=`{data.get('complexity', 'medium')}`"
                )
                for task in (data.get("tasks", []) or [])[:4]:
                    st.caption(
                        f"{task.get('task_id')} -> {task.get('worker')} | {task.get('title')}"
                    )
            elif event_name == "collaborator_trace":
                st.markdown(
                    f"- Collaborator: phase=`{data.get('phase', 'unknown')}` "
                    f"worker=`{data.get('worker', 'unknown')}` "
                    f"task=`{data.get('task_id', '')}`"
                )
                if data.get("output_preview"):
                    st.caption(data.get("output_preview"))
            elif event_name == "task_recovery":
                st.markdown(
                    f"- Task recovery: worker=`{data.get('route_worker', 'unknown')}` "
                    f"status=`{data.get('status', 'running')}`"
                )
                if data.get("payload_keys"):
                    st.caption(f"Recovered snapshot keys: {data.get('payload_keys')}")
            elif event_name == "retrieval_trace":
                st.markdown(
                    f"- 知识库检索：命中=`{data.get('hit_count', 0)}` "
                    f"候选=`{data.get('candidate_k', 0)}` 融合=`{data.get('fusion', 'n/a')}`"
                )
                diagnostics = data.get("metadata", {}).get("diagnostics", {}) if isinstance(data.get("metadata"), dict) else {}
                if diagnostics:
                    st.caption(
                        "Reason: "
                        f"{diagnostics.get('reason_code', 'ok')} | action={diagnostics.get('suggested_action', 'proceed_with_rag')}"
                    )
                    if diagnostics.get("rewrite_notes"):
                        st.caption(f"Rewrite: {diagnostics.get('rewrite_notes')}")
            elif event_name == "citation_trace":
                st.markdown(f"- 引用片段：`{len(data.get('citations', []))}`")
                citations = data.get("citations", []) or []
                if citations:
                    with st.expander("查看引用原文", expanded=False):
                        for citation in citations:
                            ref_id = citation.get("ref_id", "C?")
                            source = citation.get("source_file") or citation.get("source") or "unknown"
                            score = citation.get("score")
                            label = f"{ref_id} · {source}"
                            if score is not None:
                                label += f" · score={score:.4f}" if isinstance(score, (int, float)) else f" · score={score}"
                            st.markdown(f"**{label}**")
                            st.caption(str(citation.get("content", ""))[:1200])
            elif event_name == "tool_call":
                st.markdown(f"- 工具调用：`{event.get('name')}`")
                st.json(event.get("args", {}))
            elif event_name == "tool_result":
                ok = data.get("success")
                st.markdown(f"- 工具返回：`{event.get('name')}` 成功=`{ok}`")
                if data.get("error"):
                    st.caption(data.get("error"))
            elif event_name == "plan_assessment":
                st.markdown(
                    f"- Plan assessment: tools=`{data.get('tool_call_count', 0)}` "
                    f"mode=`{data.get('recommended_mode', 'execute')}` "
                    f"warnings=`{data.get('warnings', [])}`"
                )
                if data.get("duplicate_signature_count", 0):
                    st.caption(
                        f"Duplicate signatures: {data.get('duplicate_signature_count', 0)}"
                    )
            elif event_name == "failure_recovery":
                st.markdown(
                    f"- Recovery: action=`{data.get('action', 'fallback_answer')}` "
                    f"reason=`{data.get('reason', 'tool_failure')}`"
                )
                if data.get("instruction"):
                    st.caption(data.get("instruction"))
            elif event_name == "answer_guard":
                st.markdown(
                    f"- Answer guard: grounded=`{data.get('grounded')}` "
                    f"warnings=`{data.get('warnings', [])}`"
                )
                if data.get("recommended_action"):
                    st.caption(
                        f"Guard action: {data.get('recommended_action')} | reason={data.get('reason', '')}"
                    )
            elif event_name == "self_correction":
                st.markdown(f"- Self correction: {event.get('message') or data.get('error')}")
                if data:
                    st.markdown(
                        f"- Strategy: `{data.get('repair_strategy', 'n/a')}` "
                        f"retryable=`{data.get('retryable')}` "
                        f"remaining_budget=`{data.get('remaining_self_corrections')}`"
                    )
                    if data.get("diagnosis"):
                        st.caption(f"Diagnosis: {data.get('diagnosis')}")
                    if data.get("suggested_tool"):
                        st.caption(f"Suggested tool: {data.get('suggested_tool')}")
            elif event_name == "reasoning":
                st.markdown("- Reasoning snippet")
                st.code(event.get("content", ""), language="text")


def compact_event_summary_v2(events: List[Dict[str, Any]]) -> str:
    if not events:
        return ""
    parts: List[str] = []
    for event in events[-8:]:
        name = event.get("event")
        data = event.get("data", {}) or {}
        if name == "routing_decision":
            parts.append(f"route:{data.get('route', 'unknown')}")
        elif name == "rag_unavailable":
            parts.append("rag:unavailable")
        elif name == "tool_policy_blocked":
            parts.append(f"blocked:{data.get('tool_name', 'tool')}")
        elif name == "pending_action":
            pending = data.get("pending_action", {}) or {}
            parts.append(f"pending:{pending.get('tool_name') or data.get('tool_name', 'tool')}")
        elif name == "approval_result":
            parts.append(f"approval:{data.get('status', 'done')}")
        elif name == "agent_runtime_node":
            parts.append(f"node:{data.get('node', 'runtime')}:{data.get('status', '')}")
        elif name == "capability_status":
            parts.append(f"cap:{data.get('capability', 'unknown')}")
        elif name == "supervisor_route":
            parts.append(f"route:{data.get('worker_kind') or data.get('worker')}")
        elif name == "context_window_trace":
            parts.append(
                f"context:{data.get('selected_turn_count', 0)}/{data.get('budget', 0)}"
            )
        elif name == "multi_agent_plan":
            parts.append(f"plan2:{data.get('task_count', 0)}")
        elif name == "collaborator_trace":
            parts.append(f"collab:{data.get('worker', 'unknown')}")
        elif name == "task_recovery":
            parts.append(f"resume:{data.get('route_worker', 'unknown')}")
        elif name == "retrieval_trace":
            parts.append(f"retrieval:{data.get('hit_count', 0)}")
        elif name == "citation_trace":
            parts.append(f"citations:{len(data.get('citations', []))}")
        elif name == "tool_call":
            parts.append(f"tool:{event.get('name')}")
        elif name == "tool_result":
            parts.append(f"result:{event.get('name')}")
        elif name == "plan_assessment":
            parts.append(f"plan:{data.get('recommended_mode', 'execute')}")
        elif name == "failure_recovery":
            parts.append(f"recover:{data.get('action', 'fallback')}")
        elif name == "answer_guard":
            parts.append("guard")
        elif name == "error":
            parts.append("error")
        elif name == "finish":
            parts.append(f"finish:{data.get('status', 'done')}")
    return " -> ".join(parts)


def stream_backend_answer(prompt: str) -> Dict[str, Any]:
    contract = request_contract_preview(prompt)
    endpoint = str(contract.get("endpoint") or "/api/conversation/stream")
    payload = contract.get("payload", {})
    path = endpoint.removeprefix("/api")

    answer = ""
    events: List[Dict[str, Any]] = []
    pending_action: Optional[Dict[str, Any]] = None
    text_box = st.empty()
    trace_box = st.empty()

    try:
        with requests.post(
            api_url(path),
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            if response.status_code != 200:
                text = f"后端请求失败：HTTP {response.status_code}\n\n{response.text[:800]}"
                text_box.error(text)
                return {"role": "assistant", "content": text, "events": events}

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_name = event.get("event")
                if event_name == "done":
                    break

                if event_name in {
                    "routing_decision",
                    "agent_runtime_node",
                    "capability_status",
                    "rag_unavailable",
                    "tool_policy_blocked",
                    "pending_action",
                    "approval_result",
                    "supervisor_route",
                    "context_window_trace",
                    "multi_agent_plan",
                    "collaborator_trace",
                    "task_recovery",
                    "retrieval_trace",
                    "citation_trace",
                    "tool_call",
                    "tool_result",
                    "plan_assessment",
                    "failure_recovery",
                    "answer_guard",
                    "self_correction",
                    "reasoning",
                    "error",
                    "finish",
                }:
                    events.append(event)

                if event_name == "message_delta":
                    answer += event.get("content", "")
                elif event_name == "final_answer":
                    answer = event.get("content", "") or answer
                elif event_name == "pending_action":
                    pending_action = (event.get("data") or {}).get("pending_action") or {}
                    answer = answer or "等待人工审批后执行该高风险工具。"
                elif event_name == "status" and not answer:
                    text_box.markdown(f"<span class='muted'>{event.get('message', '')}</span>", unsafe_allow_html=True)
                elif event_name == "error":
                    answer += f"\n\n**错误**：{event.get('message', '')}"

                if answer:
                    text_box.markdown(answer + "▌")
                summary = compact_event_summary_v2(events)
                if summary:
                    trace_box.markdown(f"<div class='trace-box'>{summary}</div>", unsafe_allow_html=True)

    except requests.RequestException as exc:
        answer = f"无法连接后端服务：{exc}"
        text_box.error(answer)

    text_box.markdown(answer or "没有收到模型输出。")
    message = {"role": "assistant", "content": answer or "没有收到模型输出。", "events": events}
    if pending_action:
        pending_action.setdefault("status", "pending")
        message["pending_action"] = pending_action
    return message


def stream_pending_action_decision(
    action_id: str,
    approved: bool,
) -> Dict[str, Any]:
    endpoint = (
        f"/conversation/pending-actions/{quote(action_id, safe='')}/approve/stream"
        if approved
        else f"/conversation/pending-actions/{quote(action_id, safe='')}/reject/stream"
    )
    payload = {
        "session_id": st.session_state.session_id,
        "decided_by": "local_user",
        "reason": "Approved from Streamlit UI" if approved else "Rejected from Streamlit UI",
        "model_name": st.session_state.selected_model,
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
    }
    answer = ""
    events: List[Dict[str, Any]] = []
    text_box = st.empty()
    trace_box = st.empty()

    try:
        with requests.post(
            api_url(endpoint),
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            if response.status_code != 200:
                text = f"Approval resume request failed: HTTP {response.status_code}\n\n{response.text[:800]}"
                text_box.error(text)
                return {"role": "assistant", "content": text, "events": events}

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_name = event.get("event")
                if event_name == "done":
                    break
                if event_name in {
                    "approval_result",
                    "agent_runtime_node",
                    "tool_call",
                    "tool_result",
                    "plan_assessment",
                    "failure_recovery",
                    "self_correction",
                    "reasoning",
                    "pending_action",
                    "error",
                    "finish",
                }:
                    events.append(event)

                if event_name == "message_delta":
                    answer += event.get("content", "")
                elif event_name == "final_answer":
                    answer = event.get("content", "") or answer
                elif event_name == "error":
                    answer += f"\n\n**错误**：{event.get('message', '')}"

                if answer:
                    text_box.markdown(answer + "▌")
                summary = compact_event_summary_v2(events)
                if summary:
                    trace_box.markdown(f"<div class='trace-box'>{summary}</div>", unsafe_allow_html=True)
    except requests.RequestException as exc:
        answer = f"无法连接后端服务：{exc}"
        text_box.error(answer)

    text_box.markdown(answer or "审批已处理，但没有收到后续模型输出。")
    return {
        "role": "assistant",
        "content": answer or "审批已处理，但没有收到后续模型输出。",
        "events": events,
    }


def render_chat_workspace() -> None:
    if st.session_state.show_home:
        render_home()
        return

    render_chat_header()
    st.divider()

    for index, message in enumerate(st.session_state.messages):
        render_message(message, index=index)


def render_kb_workspace() -> None:
    collection_name = st.session_state.kb_collection
    payload = fetch_kb_files(collection_name)
    files = payload.get("data", []) if isinstance(payload.get("data", []), list) else []
    total_chunks = sum(int(item.get("chunk_count", 0)) for item in files if isinstance(item, dict))

    st.markdown('<p class="page-title">Knowledge Base</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Manage the offline Markdown knowledge-base collections used by Knowledge Q&A and Agent retrieval.</p>',
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([0.42, 0.58])
    with top_left:
        st.markdown(
            f"""
<div class="header-card">
  <div class="section-title">Indexed collection</div>
  <div class="muted">Collection: {collection_name}</div>
  <div style="margin-top:.65rem;">
    <span class="metric-pill">{len(files)} files</span>
    <span class="metric-pill">{total_chunks} chunks</span>
    <span class="metric-pill">Chroma + Whoosh</span>
    <span class="metric-pill">RAG scope default: {current_rag_scope_label()}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with top_right:
        with st.container(border=True):
            st.markdown("#### Upload structured Markdown")
            uploaded_file = st.file_uploader(
                "Choose a Markdown file",
                type=["md"],
                accept_multiple_files=False,
                label_visibility="collapsed",
            )
            upload_cols = st.columns([0.25, 0.75])
            if upload_cols[0].button("Ingest", type="primary", use_container_width=True, disabled=uploaded_file is None):
                result = upload_kb_file(uploaded_file, collection_name)
                if result.get("status") == "success":
                    st.success(result.get("message", "Upload succeeded"))
                    st.rerun()
                else:
                    st.error(result.get("message", "Upload failed"))
            upload_cols[1].caption("Only cleaned, structured Markdown files are accepted in the current pipeline.")

    st.write("")
    with st.container(border=True):
        st.markdown("#### Indexed files")
        if not files:
            st.info("This collection does not contain indexed files yet.")
        else:
            header = st.columns([0.45, 0.18, 0.2, 0.17])
            header[0].markdown("**Source file**")
            header[1].markdown("**Chunks**")
            header[2].markdown("**Store consistency**")
            header[3].markdown("**Action**")
            st.divider()
            for item in files:
                source_file = item.get("source_file", "")
                consistent = bool(item.get("consistent"))
                row = st.columns([0.45, 0.18, 0.2, 0.17])
                row[0].markdown(f"`{source_file}`")
                row[1].markdown(str(item.get("chunk_count", 0)))
                row[2].markdown("Consistent" if consistent else "Needs review")
                if row[3].button("Delete", key=f"kb_delete_{source_file}", use_container_width=True):
                    result = delete_kb_file(source_file, collection_name)
                    if result.get("status") == "success":
                        st.success(f"Deleted {source_file}")
                        st.rerun()
                    else:
                        st.error(result.get("message", "Delete failed"))

        st.divider()
        danger_cols = st.columns([0.22, 0.78])
        if danger_cols[0].button("Delete all", use_container_width=True, disabled=not files):
            st.session_state.kb_confirm_delete_all = True
        if st.session_state.kb_confirm_delete_all and files:
            danger_cols[1].warning("Click again to remove every indexed file from the current collection.")
            if danger_cols[1].button("Confirm delete all", type="primary"):
                for item in files:
                    delete_kb_file(item.get("source_file", ""), collection_name)
                st.session_state.kb_confirm_delete_all = False
                st.rerun()

    with st.expander("Knowledge-base notes", expanded=False):
        st.text_input("Future collection name", value="future_collection", disabled=True)
        st.caption("Knowledge Q&A can target one collection or `__all__`; Agent retrieval still defaults to cross-collection search.")


def render_history_workspace() -> None:
    if not st.session_state.show_home:
        render_chat_workspace()
        return

    st.markdown('<p class="page-title">History</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Open a previous session from the sidebar or start a new one.</p>', unsafe_allow_html=True)
    sessions = fetch_sessions()
    if not sessions:
        st.info("No saved sessions yet.")
        return

    with st.container(border=True):
        for session in sessions:
            session_id = session.get("session_id", "")
            cols = st.columns([0.55, 0.25, 0.2])
            cols[0].markdown(f"**{summarize(session.get('title', ''), 56)}**")
            cols[1].caption(session.get("updated_at", ""))
            if cols[2].button("Open", key=f"history_open_{session_id}", use_container_width=True):
                load_session(session_id)
                st.rerun()


def handle_chat_input() -> None:
    if st.session_state.nav_tab == "kb":
        return

    placeholder = "直接输入你的问题或任务，我会自动选择问答、知识库、工具执行或澄清。"

    prompt = st.chat_input(placeholder)
    if not prompt:
        return

    if st.session_state.show_home:
        st.session_state.show_home = False
        if st.session_state.nav_tab == "history":
            st.session_state.nav_tab = "assistant"
        if not st.session_state.chat_mode:
            st.session_state.chat_mode = "conversation"

    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        assistant_message = stream_backend_answer(prompt)
    st.session_state.messages.append(assistant_message)
    if assistant_message.get("pending_action"):
        st.rerun()


def main() -> None:
    init_state()
    render_sidebar()

    if st.session_state.nav_tab == "kb":
        render_kb_workspace()
    elif st.session_state.nav_tab == "history":
        render_history_workspace()
    else:
        render_chat_workspace()

    handle_chat_input()


if __name__ == "__main__":
    main()
