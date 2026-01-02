import os
import io
import re
import json
import time
import csv
import hashlib
import tempfile
from pathlib import Path
from http import HTTPStatus
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from PIL import Image
import dashscope


# ===================== Key（部署版：Secrets 优先，其次环境变量） =====================
API_KEY = st.secrets.get("DASHSCOPE_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", "")
MODEL_NAME = "qwen-vl-max"
dashscope.api_key = API_KEY


# ===================== 页面配置 =====================
st.set_page_config(page_title="中药多图对比", layout="wide")
st.title("⚖️ 蜜炙甘草 AI 多标准品 × 多样品 对比鉴定")
st.caption(f"Streamlit version: {st.__version__}")


# ===================== 目录（系统临时目录更稳）=====================
BASE_TEMP = Path(tempfile.gettempdir()) / "herb_check_app"
REF_DIR = BASE_TEMP / "ref"
SAMPLE_DIR = BASE_TEMP / "sample"

def _mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

_mkdir(REF_DIR)
_mkdir(SAMPLE_DIR)

# iPhone/iPad 常见 HEIC/HEIF；需要 pillow-heif 才能读
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "heic", "heif"}

def try_register_heif() -> bool:
    try:
        from pillow_heif import register_heif_opener  # type: ignore
        register_heif_opener()
        return True
    except Exception:
        return False

HEIF_OK = try_register_heif()


# ===================== 工具函数 =====================
def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def basename(p: str) -> str:
    return Path(p).name

def is_api_key_valid(key: str) -> bool:
    return bool(key) and key.startswith("sk-") and len(key) > 10

def safe_unlink(path: str):
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass

def rerun_now():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass
        # ===================== Prompt（强制 JSON 输出，便于汇总） =====================
def build_prompt_json_only() -> str:
    return """
你是一位拥有30年经验的中药炮制与质量鉴别专家，专长蜜炙甘草。
请以第一张图片【标准样品】为唯一标杆，严格鉴定第二张图片【待测样品】质量。

请【只输出严格 JSON】（不要 Markdown、不要多余文字、不要代码块），字段必须完整，结构如下：

{
  "conclusion": "合格|基本合格|不合格",
  "similarity": 0-100 的整数,
  "comparison": {
    "color": {"result": "...", "reason": "..."},
    "texture": {"result": "...", "notes": "..."},
    "oiliness": {"result": "...", "notes": "..."},
    "defects": [
      {"type":"焦斑|生心|霉变|虫蛀|杂质|无", "severity":"无|轻|中|重", "location":"..."}
    ]
  },
  "differences": ["差异要点1","差异要点2","...（最多6条）"],
  "suggestions": ["建议1","建议2","..."]
}

规则：
- similarity 必须是 0-100 整数
- defects 至少给 1 条；若未见缺陷，用 {"type":"无","severity":"无","location":"无"} 表示
- differences 最多 6 条
""".strip()


# ===================== JSON 提取/容错 =====================
def extract_json_obj(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    chunk = m.group(0)
    try:
        obj = json.loads(chunk)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ===================== 图片保存 + 缩放压缩（移动端加速关键） =====================
def save_and_optimize(uploaded_file, folder: Path, max_side: int = 1024, quality: int = 85) -> Optional[str]:
    if uploaded_file is None:
        return None
    try:
        _mkdir(folder)
        data = uploaded_file.getvalue()
        raw_name = getattr(uploaded_file, "name", "") or "upload"
        ext = Path(raw_name).suffix.lower().lstrip(".")

        if not ext:
            mime = getattr(uploaded_file, "type", "") or ""
            if "png" in mime:
                ext = "png"
            elif "webp" in mime:
                ext = "webp"
            else:
                ext = "jpg"

        if ext not in ALLOWED_EXT:
            st.error(f"不支持的格式：.{ext}（建议 JPG/PNG；HEIC 需 pillow-heif）")
            return None

        if ext in {"heic", "heif"} and not HEIF_OK:
            st.error("检测到 HEIC/HEIF，但环境缺少 pillow-heif，无法解码。请改传 JPG/PNG，或安装 pillow-heif。")
            return None

        img = Image.open(io.BytesIO(data)).convert("RGB")

        w, h = img.size
        if max(w, h) > max_side:
            scale = max(w, h) / float(max_side)
            img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)

        digest = sha1_bytes(data)[:12]
        safe_stem = Path(raw_name).stem[:50].replace(" ", "_")
        out_name = f"{safe_stem}_{digest}_ms{max_side}_q{quality}.jpg"
        out_path = folder / out_name

        if not out_path.exists():
            img.save(out_path, format="JPEG", quality=int(quality), optimize=True)

        return str(out_path.resolve())

    except Exception as e:
        st.exception(e)
        return None


# ===================== 模型调用（缓存 + 轻量重试） =====================
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def cached_compare_json(ref_path: str, sample_path: str, model_name: str, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"image": ref_path},
                {"text": "【标准参考样品】"},
                {"image": sample_path},
                {"text": "【待测样品】"},
                {"text": prompt},
            ],
        }
    ]

    last_err = None
    for attempt in range(2):
        try:
            resp = dashscope.MultiModalConversation.call(
                model=model_name,
                messages=messages,
            )
            if resp.status_code == HTTPStatus.OK:
                content = resp.output.choices[0].message.content
                if isinstance(content, list):
                    texts = [x.get("text") for x in content if isinstance(x, dict) and x.get("text")]
                    return "\n\n".join(texts) if texts else str(content)
                return str(content)

            code = getattr(resp, "code", None)
            msg = getattr(resp, "message", None)
            last_err = f"识别出错：HTTP={resp.status_code} code={code} message={msg}"
        except Exception as e:
            last_err = f"错误: {e}"

        time.sleep(0.6 * (attempt + 1))

    return last_err or "未知错误"


def analyze_pair(ref_path: str, sample_path: str) -> Dict:
    prompt = build_prompt_json_only()
    raw = cached_compare_json(ref_path, sample_path, MODEL_NAME, prompt)
    obj = extract_json_obj(raw)

    if not obj:
        obj = {
            "conclusion": "—",
            "similarity": "—",
            "comparison": {
                "color": {"result": "—", "reason": "—"},
                "texture": {"result": "—", "notes": "—"},
                "oiliness": {"result": "—", "notes": "—"},
                "defects": [{"type": "解析失败", "severity": "—", "location": "请查看原始输出"}],
            },
            "differences": ["模型输出未能解析为 JSON（建议检查提示词/模型返回格式）"],
            "suggestions": ["部署请确认 Key/额度；可降低并发并重试"],
            "_raw": raw,
        }
    return obj


# ===================== 报告渲染 =====================
def render_report_md(obj: Dict) -> str:
    conclusion = obj.get("conclusion", "—")
    similarity = obj.get("similarity", "—")
    comp = obj.get("comparison", {}) or {}
    color = comp.get("color", {}) or {}
    texture = comp.get("texture", {}) or {}
    oil = comp.get("oiliness", {}) or {}
    defects = comp.get("defects", []) or []
    diffs = obj.get("differences", []) or []
    sugg = obj.get("suggestions", []) or []

    def fmt_list(items):
        if not items:
            return "- 无"
        if isinstance(items, list):
            return "\n".join([f"- {x}" for x in items if str(x).strip()][:10]) or "- 无"
        return f"- {items}"

    defect_lines = []
    if isinstance(defects, list):
        for d in defects[:10]:
            if isinstance(d, dict):
                defect_lines.append(
                    f"- {d.get('type','—')}（{d.get('severity','—')}）位置/说明：{d.get('location','—')}"
                )
            else:
                defect_lines.append(f"- {str(d)}")
    else:
        defect_lines = [f"- {str(defects)}"]

    md = f"""
### 《对比鉴定报告》
**总评结论：** {conclusion}  
**相似度评分：** {similarity}

#### 1) 四项对比
- **色泽对比：** {color.get("result","—")}  
  - 原因/解释：{color.get("reason","—")}
- **切面纹理：** {texture.get("result","—")}  
  - 备注：{texture.get("notes","—")}
- **油润度：** {oil.get("result","—")}  
  - 备注：{oil.get("notes","—")}
- **缺陷筛查：**
{chr(10).join(defect_lines) if defect_lines else "- 无"}

#### 2) 差异要点（按影响排序）
{fmt_list(diffs)}

#### 3) 操作建议
{fmt_list(sugg)}
""".strip()
    return md


# ===================== 质检汇总/风险等级 =====================
def _to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def compute_risk(similarity: Optional[int], defect_types: List[str]) -> str:
    defect_set = set([d for d in defect_types if d and d != "无"])
    if ("霉变" in defect_set) or ("虫蛀" in defect_set) or ("生心" in defect_set):
        return "高"
    if similarity is not None and similarity < 70:
        return "高"
    if ("焦斑" in defect_set) or ("杂质" in defect_set):
        return "中"
    if similarity is not None and similarity < 85:
        return "中"
    return "低"

def make_summary_row(ref_path: str, sample_path: str, report: Dict) -> Dict:
    comp = report.get("comparison", {}) or {}

    color_result = ((comp.get("color") or {}) if isinstance(comp.get("color"), dict) else {}).get("result", "—")
    texture_result = ((comp.get("texture") or {}) if isinstance(comp.get("texture"), dict) else {}).get("result", "—")
    oiliness_result = ((comp.get("oiliness") or {}) if isinstance(comp.get("oiliness"), dict) else {}).get("result", "—")

    defects = comp.get("defects", []) or []
    defect_types = []
    defect_detail_lines = []

    if isinstance(defects, list):
        for d in defects[:10]:
            if isinstance(d, dict):
                t = d.get("type", "—")
                sev = d.get("severity", "—")
                loc = d.get("location", "—")
                if t and t != "无" and t not in defect_types:
                    defect_types.append(t)
                defect_detail_lines.append(f"{t}({sev})@{loc}")
            else:
                defect_detail_lines.append(str(d))
    else:
        defect_detail_lines.append(str(defects))

    defects_text = "、".join(defect_types) if defect_types else "无"
    defects_detail = "；".join(defect_detail_lines) if defect_detail_lines else "无"

    diffs = report.get("differences", []) or []
    diff_top3 = "；".join([str(x) for x in diffs[:3]]) if isinstance(diffs, list) and diffs else (str(diffs) if diffs else "无")

    sugg = report.get("suggestions", []) or []
    suggestions_summary = "；".join([str(x) for x in sugg[:2]]) if isinstance(sugg, list) and sugg else (str(sugg) if sugg else "无")

    similarity_int = _to_int(report.get("similarity", None), None)
    risk = compute_risk(similarity_int, defect_types)

    return {
        "risk": risk,
        "ref": basename(ref_path),
        "sample": basename(sample_path),
        "conclusion": report.get("conclusion", "—"),
        "similarity": similarity_int if similarity_int is not None else report.get("similarity", "—"),
        "color_result": color_result,
        "texture_result": texture_result,
        "oiliness_result": oiliness_result,
        "defects": defects_text,
        "defects_detail": defects_detail,
        "diff_top3": diff_top3,
        "suggestions_summary": suggestions_summary,
    }
# ===================== Session State =====================
def init_list_state(key: str):
    if key not in st.session_state:
        st.session_state[key] = []

def add_path_to_list(path: Optional[str], list_key: str):
    if path and path not in st.session_state[list_key]:
        st.session_state[list_key].append(path)

def remove_path_from_list(list_key: str, path: str, delete_file: bool = False):
    st.session_state[list_key] = [p for p in st.session_state[list_key] if p != path]
    if delete_file:
        safe_unlink(path)

def clear_list(list_key: str, delete_files: bool = False):
    if delete_files:
        for p in st.session_state[list_key]:
            safe_unlink(p)
    st.session_state[list_key] = []

init_list_state("ref_list")
init_list_state("sample_list")
init_list_state("last_results")  # [{ref_path, sample_path, report}]


# ===================== 参数区（移动端默认更快） =====================
with st.expander("⚙️ 速度/质量参数（手机/iPad建议默认）", expanded=False):
    max_side = st.slider("图片最大边长（越小越快）", 512, 1536, 1024, 128)
    quality = st.slider("JPEG质量（越小越快）", 50, 95, 85, 5)
    concurrency = st.slider("并发数（移动端建议 2~4）", 1, 8, 3, 1)
    show_thumbs = st.checkbox("显示缩略图预览（可能稍慢）", value=False)
    realtime_reports = st.checkbox("实时显示每组报告（可能稍慢）", value=False)
    delete_files_on_remove = st.checkbox("删除列表项时同时删除磁盘文件（省空间）", value=False)


def render_list_with_delete(list_key: str, title: str):
    st.write(title)
    if not st.session_state[list_key]:
        st.info("（空）")
        return

    for i, p in enumerate(st.session_state[list_key], 1):
        c1, c2, c3 = st.columns([0.08, 0.72, 0.20])
        with c1:
            st.write(f"{i}.")
        with c2:
            st.write(basename(p))
        with c3:
            if st.button("✖ 删除", key=f"del_{list_key}_{i}", use_container_width=True):
                remove_path_from_list(list_key, p, delete_file=delete_files_on_remove)
                rerun_now()

        if show_thumbs:
            st.image(p, use_container_width=True)


# ===================== UI（三列） =====================
col1, col2, col3 = st.columns([1, 1, 1.2], gap="large")

with col1:
    st.header("1. 标准图（可追加多张）")
    ref_one = st.file_uploader("每次选择 1 张标准图并加入列表", type=list(ALLOWED_EXT), key="ref_one")
    if ref_one:
        path = save_and_optimize(ref_one, REF_DIR, max_side=max_side, quality=quality)
        add_path_to_list(path, "ref_list")
        st.success(f"已加入：{getattr(ref_one,'name','标准图')}") if path else st.error("标准图处理失败")

    a, b = st.columns(2)
    with a:
        st.button("🧹 清空标准图", on_click=clear_list, args=("ref_list", delete_files_on_remove), use_container_width=True)
    with b:
        st.write(f"已添加：**{len(st.session_state.ref_list)}** 张")

    st.divider()
    render_list_with_delete("ref_list", "标准图列表：")

with col2:
    st.header("2. 待测图（可追加多张）")

    cam = st.camera_input("拍照（每次拍 1 张加入列表）", key="cam_one")
    if cam:
        path = save_and_optimize(cam, SAMPLE_DIR, max_side=max_side, quality=quality)
        add_path_to_list(path, "sample_list")
        st.success("已加入：拍照图片") if path else st.error("拍照图片处理失败")

    sample_one = st.file_uploader("或每次选择 1 张待测图并加入列表", type=list(ALLOWED_EXT), key="sample_one")
    if sample_one:
        path = save_and_optimize(sample_one, SAMPLE_DIR, max_side=max_side, quality=quality)
        add_path_to_list(path, "sample_list")
        st.success(f"已加入：{getattr(sample_one,'name','待测图')}") if path else st.error("待测图处理失败")

    a, b = st.columns(2)
    with a:
        st.button("🧹 清空待测图", on_click=clear_list, args=("sample_list", delete_files_on_remove), use_container_width=True)
    with b:
        st.write(f"已添加：**{len(st.session_state.sample_list)}** 张")

    st.divider()
    render_list_with_delete("sample_list", "待测图列表：")

with col3:
    st.header("3. 对比结果")

    ref_abs: List[str] = st.session_state.ref_list
    sample_abs: List[str] = st.session_state.sample_list
    total_pairs = len(ref_abs) * len(sample_abs)
    st.write(f"将生成对比：**{len(ref_abs)} 标准 × {len(sample_abs)} 样品 = {total_pairs} 组**")

    if is_api_key_valid(API_KEY):
        st.success("API Key 已读取 ✅（Secrets/环境变量）")
    else:
        st.warning('当前 API Key 不可用（为空或仍是占位符）。请在 Streamlit Secrets 配置：DASHSCOPE_API_KEY="sk-..."')

    cA, cB = st.columns([0.65, 0.35])
    with cA:
        with st.form("run_form"):
            run = st.form_submit_button(
                "🔍 开始对比",
                type="primary",
                use_container_width=True,
                disabled=(not ref_abs or not sample_abs or not is_api_key_valid(API_KEY)),
            )
    with cB:
        if st.button("🧾 清空本次结果", use_container_width=True):
            st.session_state["last_results"] = []
            rerun_now()

    if run:
        st.session_state["last_results"] = []
        progress = st.progress(0)
        status = st.status("准备开始…", expanded=True)

        tasks = [(r, s) for r in ref_abs for s in sample_abs]
        done = 0
        status.update(label=f"分析中：0/{total_pairs}", state="running")

        live_container = st.container() if realtime_reports else None

        with ThreadPoolExecutor(max_workers=int(concurrency)) as ex:
            future_map = {ex.submit(analyze_pair, r, s): (r, s) for r, s in tasks}

            for fut in as_completed(future_map):
                r, s = future_map[fut]
                done += 1
                progress.progress(done / total_pairs)
                status.update(label=f"分析中：{done}/{total_pairs}", state="running")

                try:
                    report = fut.result()
                except Exception as e:
                    report = {
                        "conclusion": "—",
                        "similarity": "—",
                        "comparison": {
                            "color": {"result": "—", "reason": "—"},
                            "texture": {"result": "—", "notes": "—"},
                            "oiliness": {"result": "—", "notes": "—"},
                            "defects": [{"type": "错误", "severity": "—", "location": str(e)}],
                        },
                        "differences": ["运行异常"],
                        "suggestions": ["请降低并发或检查网络/Key/额度"],
                    }

                st.session_state["last_results"].append({
                    "ref_path": r,
                    "sample_path": s,
                    "report": report
                })

                if realtime_reports and live_container is not None:
                    with live_container:
                        st.subheader(f"#{done}/{total_pairs}  标准：{basename(r)}  vs  样品：{basename(s)}")
                        st.markdown(render_report_md(report))
                        st.divider()

        status.update(label="全部完成 ✅", state="complete", expanded=False)

    # ===================== 质检汇总表 + 下载 =====================
    if st.session_state["last_results"]:
        st.subheader("📊 质检汇总表（可下载 CSV）")

        rows = [make_summary_row(x["ref_path"], x["sample_path"], x["report"]) for x in st.session_state["last_results"]]

        risk_order = {"高": 0, "中": 1, "低": 2}
        def sort_key(r):
            sim = r.get("similarity")
            simv = sim if isinstance(sim, int) else 9999
            return (risk_order.get(r.get("risk", "低"), 9), simv)

        rows_sorted = sorted(rows, key=sort_key)
        st.dataframe(rows_sorted, use_container_width=True)

        csv_buf = io.StringIO()
        fieldnames = [
            "risk",
            "ref", "sample",
            "conclusion", "similarity",
            "color_result", "texture_result", "oiliness_result",
            "defects", "defects_detail",
            "diff_top3",
            "suggestions_summary",
        ]
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(r)

        st.download_button(
            "⬇️ 下载质检汇总 CSV",
            data=csv_buf.getvalue().encode("utf-8-sig"),
            file_name="herb_qc_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("📦 下载每组完整 JSON（可选）", expanded=False):
            all_json = [
                {"ref": basename(x["ref_path"]), "sample": basename(x["sample_path"]), "report": x["report"]}
                for x in st.session_state["last_results"]
            ]
            st.download_button(
                "⬇️ 下载完整 JSON",
                data=json.dumps(all_json, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="herb_qc_full.json",
                mime="application/json",
                use_container_width=True,
            )