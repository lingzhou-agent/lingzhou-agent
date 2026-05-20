"""tools/file.py — 文件读写和编辑工具。"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from core.paths import project_root, data_dir
from typing import Any

from tools.registry import ToolManifest, ToolParam, ToolResult, ToolContext, tool

_log = logging.getLogger("lingzhou.tools.file")

# 核心目录/文件检测 — 基于路径模式，无需维护静态列表
_CORE_DIRS = frozenset({"core", "memory", "provider"})
_CORE_EXTRA = frozenset({"tools/registry.py"})


def _is_core_file(path: Path) -> bool:
    """检测是否为核心 .py 文件（按路径模式，新增文件自动覆盖）。"""
    if path.suffix != ".py":
        return False
    try:
        rel = path.relative_to(_repo_root())
    except ValueError:
        return False
    if rel.parts[0] in _CORE_DIRS:
        return True
    if str(rel) in _CORE_EXTRA:
        return True
    return False

# 文件大小限制
MAX_READ_CHARS = 100_000
MAX_WRITE_CHARS = 200_000

# 写入允许的根目录 — workspace 沙箱
# 空集合 = 允许所有路径；设为具体路径则只允许写入这些目录内的文件。
# 系统级操作（如 /etc/systemd）请通过 shell.run 执行。
_WRITE_ALLOW_ROOTS: frozenset[str] = frozenset({
    str(project_root()),
    str(data_dir()),
})


def _path_guard(path: Path) -> tuple[bool, str]:
    """路径安全守卫。
    
    返回 (is_safe, warning)。
    - 路径穿越检测（../etc/passwd 等）
    - 可选 workspace 沙箱
    """
    try:
        resolved = path.resolve()
    except Exception:
        return False, f"无法解析路径: {path}"
    
    # 路径穿越检测：确保解析后的路径仍在预期范围内
    if _WRITE_ALLOW_ROOTS:
        for root in _WRITE_ALLOW_ROOTS:
            try:
                resolved.relative_to(root)
                break
            except ValueError:
                continue
        else:
            return False, (
                f"路径 {path} 不在允许的工作区内。"
                f"只能写入 {', '.join(_WRITE_ALLOW_ROOTS)} 内的文件。"
            )
    
    return True, ""



import re


def _fuzzy_find(content: str, old_text: str) -> int:
    """模糊匹配链：依次尝试宽松策略找到 old_text 在 content 中的位置。
    
    策略：
    1. 行去空格匹配 — 每行 strip 后比较
    2. 空白归一化 — 多空格/Tab 坍塌为单空格
    3. 换行归一化 — 字面 \\n 转为实际换行
    
    返回匹配位置，找不到返回 -1。
    """
    # 策略1: 行去空格匹配
    old_lines = [l.strip() for l in old_text.split("\n")]
    content_lines = content.split("\n")
    for i in range(len(content_lines) - len(old_lines) + 1):
        match = True
        for j, old_l in enumerate(old_lines):
            cl = content_lines[i + j].strip()
            if not cl.startswith(old_l):
                match = False
                break
        if match:
            return sum(len(l) + 1 for l in content_lines[:i])
    
    # 策略2: 空白归一化（忽略所有空白差异）
    def _strip_spaces(s):
        return re.sub(r"\s+", "", s)
    old_nosp = _strip_spaces(old_text)
    content_nosp = _strip_spaces(content)
    idx = content_nosp.find(old_nosp)
    if idx != -1 and len(old_nosp) >= 6:
        # 反推原始位置
        _pos = 0
        for _i in range(len(content)):
            if not content[_i].isspace():
                if _pos == idx:
                    return _i
                _pos += 1
        return content.find(old_text.split("\n")[0].strip())
    
    # 策略3: 换行归一化
    old_unescaped = old_text.replace("\\n", "\n").replace("\\t", "\t")
    idx = content.find(old_unescaped)
    if idx != -1:
        return idx
    
    return -1


def _safety_guard(path: Path, operation: str) -> tuple[str, str | None]:
    """代码修改安全守卫：备份 + 语法验证。
    返回 (warning, error)。error 非空表示应阻断操作。
    """
    warnings: list[str] = []
    
    # 1. 自动备份（Python 文件）
    if path.suffix == '.py' and path.exists():
        backup = path.with_suffix(path.suffix + '.lingzhou-backup')
        try:
            backup.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
        except Exception:
            pass  # 备份失败不阻断
    
    # 2. 核心文件警告
    if _is_core_file(path):
        try:
            rel = str(path.relative_to(_repo_root()))
        except ValueError:
            rel = str(path)
        warnings.append(
            f"⚠️ 正在修改核心文件 {rel}。请修改后立即用 shell.run 验证系统可正常导入/启动。"
        )
    
    return "\n".join(warnings) if warnings else "", None


def _verify_python_syntax(path: Path, content: str) -> str | None:
    """验证 Python 文件语法。返回 None 表示通过，否则返回错误信息。"""
    if not path.suffix == '.py':
        return None
    try:
        import ast
        ast.parse(content)
        return None
    except SyntaxError as e:
        return f"语法错误: {e}"
    except Exception as e:
        return f"验证异常: {e}"


def _pycompile_check(path: Path) -> str | None:
    """对磁盘上已写入的 .py 文件执行 py_compile.compile()，返回 None 表示通过，否则返回错误信息。"""
    if path.suffix != ".py":
        return None
    try:
        import py_compile
        py_compile.compile(str(path), doraise=True)
        return None
    except py_compile.PyCompileError as e:
        return f"py_compile 校验失败: {e}"
    except Exception as e:
        return f"py_compile 异常: {e}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _log_preview_text(text: str, limit: int = 80) -> str:
    first_line = (text or "").splitlines()[0] if text else ""
    cleaned = " ".join(first_line.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _tail_after_anchor(path: Path, anchor: str) -> Path | None:
    parts = path.parts
    if anchor not in parts:
        return None
    idx = len(parts) - 1 - parts[::-1].index(anchor)
    tail = parts[idx + 1 :]
    if not tail:
        return None
    return Path(*tail)


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        key = str(p)
        if key and key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _read_rel_candidates(path: Path) -> list[Path]:
    rels: list[Path] = []

    if not path.is_absolute():
        rels.append(path)

    for anchor in ("workspace", "lingzhou", ".lingzhou"):
        rel = _tail_after_anchor(path, anchor)
        if rel is not None:
            rels.append(rel)
            if rel.parts and rel.parts[0] == "workspace" and len(rel.parts) > 1:
                rels.append(Path(*rel.parts[1:]))

    if path.name and path.name not in ("", "."):
        rels.append(Path(path.name))

    return _unique_paths(rels)


def _candidate_paths(path: Path, bases: list[Path]) -> list[Path]:
    rels = _read_rel_candidates(path)
    candidates: list[Path] = []
    seen: set[str] = set()

    for rel in rels:
        for base in bases:
            candidate = base / rel
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)

            if rel.parts and rel.parts[0] != "workspace":
                candidate = base / "workspace" / rel
                key = str(candidate)
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)

    return candidates


def resolve_read_path(path: Path, ctx: ToolContext | None = None) -> Path:
    if path.exists():
        return path

    repo_root = _repo_root()
    workspace = _workspace_dir(ctx) if ctx is not None else None
    if workspace is not None:
        for candidate in _candidate_paths(path, _unique_paths([workspace, workspace.parent, repo_root])):
            if candidate.exists():
                return candidate

    cwd = Path.cwd()
    home = Path.home()
    bases: list[Path] = [cwd, *cwd.parents, home, home / ".lingzhou", repo_root]

    for candidate in _candidate_paths(path, _unique_paths(bases)):
        if candidate.exists():
            return candidate

    return path


def _workspace_dir(ctx: ToolContext) -> Path | None:
    raw = getattr(getattr(ctx.config, "loop", None), "workspace_dir", "")
    if not raw:
        return None
    try:
        return Path(str(raw)).expanduser()
    except (TypeError, ValueError):
        return None


def workspace_candidate_path(path: Path, ctx: ToolContext) -> Path | None:
    workspace = _workspace_dir(ctx)
    if workspace is None:
        return None

    roots = _unique_paths([workspace, workspace.parent, _repo_root()])
    for candidate in _candidate_paths(path, roots):
        if candidate.exists() or candidate.parent.exists():
            return candidate
    return None


def _resolve_mutation_path(path: Path, ctx: ToolContext) -> Path:
    if path.exists():
        return path

    resolved = resolve_read_path(path, ctx)
    if resolved.exists():
        return resolved

    workspace_candidate = workspace_candidate_path(path, ctx)
    if workspace_candidate is not None:
        return workspace_candidate

    return path


@tool(ToolManifest(
    name="file.list",
    description="列出目录内容。支持 shallow list，用于替代 shell.run 的 ls/find 场景。",
    prefer_tier="reader",
    progress_category="info",
    capabilities=("ask_evidence", "plan_bootstrap_exempt", "plan_alignment_exempt", "completion_info_only"),
    params=[
        ToolParam("path", "string", "目录路径", required=True),
        ToolParam("limit", "number", "最多返回多少项，默认 200", required=False),
        ToolParam("include_hidden", "boolean", "是否包含隐藏文件，默认 false", required=False),
    ],
))
async def file_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = resolve_read_path(Path(params.get("path") or "").expanduser(), ctx)
    limit = int(params.get("limit") or 200)
    include_hidden = bool(params.get("include_hidden", False))

    if not path.exists():
        return ToolResult(summary=f"路径不存在: {path}", error="FileNotFound")
    if not path.is_dir():
        return ToolResult(summary=f"不是目录: {path}", error="NotADirectory")

    try:
        entries = []
        for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
            if not include_hidden and child.name.startswith('.'):
                continue
            suffix = "/" if child.is_dir() else ""
            entries.append(child.name + suffix)
        clipped = entries[:max(0, limit)]
        remaining = max(0, len(entries) - len(clipped))
        body = "\n".join(clipped)
        if remaining:
            body += f"\n... (+{remaining} more)"
        payload = {"path": str(path), "count": len(entries), "returned": len(clipped)}
        return ToolResult(
            summary=body or "（空目录）",
            evidence=json.dumps(payload, ensure_ascii=False),
            resource_key=str(path),
            fingerprint=f"list:{hashlib.md5((body or '（空目录）').encode()).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            metadata=payload,
        )
    except Exception as e:
        _log.exception("列出目录失败: %s", path)
        return ToolResult(summary=f"列出失败: {path}", error=type(e).__name__)


@tool(ToolManifest(
    name="file.read",
    description=(
        "读取文件内容。支持三种方式：\n"
        "1. 不传参数 → 读全文\n"
        "2. offset + limit → 从第 offset 行开始读 limit 行（推荐，直觉友好）\n"
        "3. start + end → 按字符下标区间读（精确控制）\n"
        "⚠️ 修改文件前先用 offset/limit 读完整函数（≥20行），避免碎片化。"
    ),
    prefer_tier="reader",
    progress_category="info",
    capabilities=("ask_evidence", "plan_bootstrap_exempt", "plan_alignment_exempt", "completion_info_only"),
    params=[
        ToolParam("path", "string", "文件路径", required=True),
        ToolParam("offset", "number", "起始行号（1-indexed），配合 limit 使用", required=False),
        ToolParam("limit", "number", "读取行数", required=False),
        ToolParam("start", "number", "起始字符下标（含），默认 0", required=False),
        ToolParam("end", "number", "结束字符下标（不含），默认到末尾", required=False),
        ToolParam("max_chars", "number", "最大字符数", required=False),
    ],
))
async def file_read(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    raw_path = str(params.get("path") or "").strip()
    if not raw_path:
        return ToolResult(summary="path 不能为空", error="EmptyPath", skipped=True)

    path = resolve_read_path(Path(raw_path).expanduser(), ctx)
    max_chars_raw = params.get("max_chars")
    max_chars: int | None = int(max_chars_raw) if max_chars_raw is not None else None
    has_range = ("start" in params) or ("end" in params)
    has_line_range = ("offset" in params) or ("limit" in params)

    if not path.exists():
        return ToolResult(summary=f"文件不存在: {path}", error="FileNotFound")
    if not path.is_file():
        return ToolResult(summary=f"不是文件: {path}", error="NotAFile", skipped=True)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        total = len(text)

        start = 0
        end = total

        if has_line_range:
            lines = text.split("\n")
            line_offset = max(0, int(params.get("offset", 1)) - 1)  # 1-indexed → 0-indexed
            line_limit = int(params.get("limit", 50)) if "limit" in params else 50
            selected = lines[line_offset:line_offset + line_limit]
            text = "\n".join(selected)
            if line_offset > 0 or line_offset + line_limit < len(lines):
                text = f"[行 {line_offset + 1}-{min(line_offset + line_limit, len(lines))} / 共 {len(lines)} 行]\n{text}"
        elif has_range:
            start = int(params.get("start") or 0)
            end_raw = params.get("end")
            end = int(end_raw) if end_raw is not None else total
            text = text[start:end]

        if max_chars is not None:
            text = text[:max(0, max_chars)]

        return ToolResult(
            summary=text,
            resource_key=str(path),
            fingerprint=f"read:{hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            metadata={
                "path": str(path),
                "chars": len(text),
                "has_range": has_range,
                "start": start,
                "end": end,
                "max_chars": max_chars,
                "log_summary": (
                    f"file.read path={path} chars={len(text)}"
                    + (f" range=[{start}:{end}]" if has_range else "")
                    + (f" preview={_log_preview_text(text)!r}" if text else " preview=''" )
                ),
            },
        )
    except Exception as e:
        _log.exception("读取文件失败: %s", path)
        return ToolResult(summary=f"读取失败: {path}", error=type(e).__name__)


@tool(ToolManifest(
    name="file.write",
    description="写入文件内容。如果文件已存在则覆盖全部内容。创建新文件时自动创建父目录。",
    progress_category="mutation",
    capabilities=("completion_mutation",),
    params=[
        ToolParam("path", "string", "文件路径", required=True),
        ToolParam("content", "string", "要写入的内容", required=True),
    ],
))
async def file_write(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = _resolve_mutation_path(Path(params.get("path") or "").expanduser(), ctx)
    content = params.get("content")

    if content is None:
        return ToolResult(summary="写入内容为空", error="EmptyContent", skipped=True)

    # 路径守卫：沙箱 + 穿越检测
    ok, err = _path_guard(path)
    if not ok:
        return ToolResult(summary=err, error="PathBlocked", skipped=True)

    text = str(content)
    if len(text) > MAX_WRITE_CHARS:
        return ToolResult(
            summary=f"内容超出大小限制: {len(text)} > {MAX_WRITE_CHARS} 字符",
            error="ContentTooLarge", skipped=True
        )

    try:
        # 安全守卫：备份原文件 + 核心文件警告
        guard_warning, _ = _safety_guard(path, "写入")
        
        # 目录保护：路径是目录而非文件时给出明确提示
        if path.is_dir():
            return ToolResult(
                summary=f"无法写入：{path} 是一个目录，不是文件。请指定具体文件路径（如 {path}/README.md）。",
                error="IsDirectory",
                skipped=True,
            )
        
        path.parent.mkdir(parents=True, exist_ok=True)
        # 写前语法验证（.py 文件）：语法错误直接阻断，不写入损坏文件
        if path.suffix == ".py":
            syntax_error = _verify_python_syntax(path, text)
            if syntax_error:
                return ToolResult(
                    summary=f"拒绝写入: {path}\n{syntax_error}\n备份（若有）: {path.with_suffix('.py.lingzhou-backup')}",
                    error="PythonSyntaxError",
                    skipped=True,
                )
        # 原子写入：先写临时文件，成功后再 rename
        tmp_path = path.with_suffix(path.suffix + ".lingzhou-tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)  # 原子 rename
        summary = f"写入成功: {path} ({len(text)} 字符)"
        if guard_warning:
            summary += f"\n{guard_warning}"
        compile_err = _pycompile_check(path)
        if compile_err:
            summary += f"\n⚠️ {compile_err}"
        syntax_ok = compile_err is None
        return ToolResult(
            summary=summary,
            resource_key=str(path),
            fingerprint=f"write:{hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            state_delta={"file": "written", "chars": len(text), "syntax_ok": syntax_ok},
            metadata={"path": str(path), "chars": len(text), "syntax_ok": syntax_ok},
        )
    except Exception as e:
        _log.exception("写入文件失败: %s", path)
        return ToolResult(summary=f"写入失败: {path}", error=type(e).__name__)


@tool(ToolManifest(
    name="file.edit",
    description=(
        "对文件进行精确文本替换。支持单处或多处替换（edit 列表）。"
        "每个 edit 包含 oldText（原文本）和 newText（新文本），oldText 必须在文件中唯一匹配。"
        "这是修改文件的首选工具——相比全量覆盖的 file.write，edit 只改需要改的部分，安全且节省 token。"
    ),
    progress_category="mutation",
    capabilities=("completion_mutation",),
        params=[
        ToolParam("path", "string", "文件路径", required=True),
        ToolParam("edits", "object",
                  "替换操作列表，每项包含 oldText（要替换的原文）和 newText（替换后的内容）。"
                  "例: [{\"oldText\": \"foo\", \"newText\": \"bar\"}, {\"oldText\": \"baz\", \"newText\": \"qux\"}]",
                  required=True),
    ],
))
async def file_edit(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = _resolve_mutation_path(Path(params.get("path") or "").expanduser(), ctx)
    edits_raw = params.get("edits")

    if not path.exists():
        return ToolResult(summary=f"文件不存在: {path}（edit 只能修改已存在的文件，新文件请用 file.write）", error="FileNotFound")

    if not edits_raw:
        return ToolResult(summary="edits 参数为空，请提供至少一个 {oldText, newText} 替换操作", error="EmptyEdits", skipped=True)

    # 支持 list 或 JSON 字符串
    if isinstance(edits_raw, str):
        try:
            edits = json.loads(edits_raw)
        except json.JSONDecodeError:
            return ToolResult(summary="edits 不是合法的 JSON 数组", error="InvalidJSON")
    elif isinstance(edits_raw, list):
        edits = edits_raw
    else:
        return ToolResult(summary="edits 必须是数组或 JSON 字符串", error="InvalidType", skipped=True)

    try:
        # 目录保护
        if path.is_dir():
            return ToolResult(
                summary=f"无法编辑：{path} 是一个目录，不是文件。请指定具体文件路径。",
                error="IsDirectory",
                skipped=True,
            )

        # 路径守卫
        ok, err = _path_guard(path)
        if not ok:
            return ToolResult(summary=err, error="PathBlocked", skipped=True)
        
        # 安全守卫：备份原文件 + 核心文件警告
        guard_warning, _ = _safety_guard(path, "编辑")
        original = path.read_text(encoding="utf-8", errors="replace")
        content = original
        matched: list[dict[str, Any]] = []
        applied = []

        for i, edit in enumerate(edits):
            old_text = edit.get("oldText", "") if isinstance(edit, dict) else ""
            new_text = edit.get("newText", "") if isinstance(edit, dict) else ""

            if not old_text:
                return ToolResult(summary=f"edits[{i}]: oldText 不能为空", error="EmptyOldText", skipped=True)

            # 所有 edit 都基于原始文件匹配，而不是增量匹配修改后的内容。
            first_idx = original.find(old_text)
            if first_idx == -1:
                # 模糊匹配链：依次尝试宽松策略
                fuzzy_idx = _fuzzy_find(original, old_text)
                if fuzzy_idx != -1:
                    first_idx = fuzzy_idx
                    _log.info("[file.edit] 使用模糊匹配找到 oldText")
                else:
                    # 查找 partial match 帮助 LLM 定位
                    first_line = old_text.split("\n")[0][:60]
                    partial_idx = original.find(first_line) if len(first_line) > 10 else -1
                    context = ""
                    if partial_idx != -1:
                        ctx_start = max(0, partial_idx - 150)
                        ctx_end = min(len(original), partial_idx + len(first_line) + 300)
                        context = f"\n实际内容（该位置附近）:\n{original[ctx_start:ctx_end]}"
                    else:
                        # 完全没找到，返回完整文件内容，让 LLM 自己定位并重建 oldText
                        context = f"\n完整文件内容:\n{original}"
                    return ToolResult(
                        summary=f"edits[{i}]: oldText 在文件中未找到。{context}\n请用 file.read 确认完整的当前内容后重试。",
                        error="OldTextNotFound",
                        skipped=True,
                        metadata={"old_preview": old_text[:80], "partial_match": partial_idx != -1},
                    )

            second_idx = original.find(old_text, first_idx + 1)
            if second_idx != -1:
                return ToolResult(
                    summary=(
                        f"edits[{i}]: oldText 在文件中出现 {original.count(old_text)} 次，不够唯一。"
                        f"请扩大 oldText 范围使其唯一，或拆分为多次 edit 调用。"
                    ),
                    error="NonUniqueOldText",
                    skipped=True,
                )

            matched.append({
                "index": i,
                "start": first_idx,
                "end": first_idx + len(old_text),
                "new_text": new_text,
                "old_preview": old_text[:60] + ("..." if len(old_text) > 60 else ""),
                "new_preview": new_text[:60] + ("..." if len(new_text) > 60 else ""),
            })

        matched.sort(key=lambda item: int(item["start"]))
        for prev, cur in zip(matched, matched[1:]):
            if int(prev["end"]) > int(cur["start"]):
                return ToolResult(
                    summary=(
                        f"edits[{prev['index']}] 与 edits[{cur['index']}] 在原文件中发生重叠。"
                        "请合并成一次 edit，或拆成互不重叠的目标区域。"
                    ),
                    error="OverlappingEdits",
                    skipped=True,
                )

        for item in reversed(matched):
            start = int(item["start"])
            end = int(item["end"])
            content = content[:start] + str(item["new_text"]) + content[end:]
            applied.append({
                "index": item["index"],
                "old_preview": item["old_preview"],
                "new_preview": item["new_preview"],
            })

        changes_made = len(matched)
        if content == original:
            return ToolResult(summary=f"编辑未产生变化: {path}", error="NoChange", skipped=True)

        # 写前语法验证（.py 文件）：语法错误直接阻断，不写入损坏文件
        if path.suffix == ".py":
            syntax_error = _verify_python_syntax(path, content)
            if syntax_error:
                return ToolResult(
                    summary=f"拒绝编辑: {path}\n{syntax_error}\n备份（若有）: {path.with_suffix('.py.lingzhou-backup')}",
                    error="PythonSyntaxError",
                    skipped=True,
                )
        # 原子写入
        tmp_path = path.with_suffix(path.suffix + ".lingzhou-tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
        applied.reverse()
        applied_summary = "\n".join(
            f"  [{a['index']}] {a['old_preview']} → {a['new_preview']}"
            for a in applied
        )
        summary = f"编辑成功: {path}（{changes_made} 处替换）\n{applied_summary}"
        if guard_warning:
            summary += f"\n{guard_warning}"
        compile_err = _pycompile_check(path)
        if compile_err:
            summary += f"\n⚠️ {compile_err}"
        syntax_ok = compile_err is None
        payload = {"path": str(path), "changes": changes_made, "applied": applied}
        return ToolResult(
            summary=summary,
            evidence=json.dumps(payload, ensure_ascii=False),
            resource_key=str(path),
            fingerprint=f"edit:{hashlib.md5(content.encode('utf-8', errors='replace')).hexdigest()[:12]}",
            artifact_paths=[str(path)],
            state_delta={"file": "edited", "changes": changes_made, "syntax_ok": syntax_ok},
            metadata={**payload, "syntax_ok": syntax_ok},
        )
    except Exception as e:
        _log.exception("编辑文件失败: %s", path)
        return ToolResult(summary=f"编辑失败: {path}", error=type(e).__name__)


@tool(ToolManifest(
    name="file.delete",
    description=(
        "删除 workspace 中的指定文件。"
        "⚠️ 不可逆操作，请确认路径正确。"
        "主要用途：初始化完成后删除 BOOTSTRAP.md，或清理临时文件。"
    ),
    progress_category="mutation",
    capabilities=("completion_mutation",),
    params=[
        ToolParam("path", "string", "要删除的文件路径（相对于 workspace 或绝对路径）", required=True),
    ],
))
async def file_delete(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = _resolve_mutation_path(Path(params.get("path") or "").expanduser(), ctx)

    if not path.exists():
        return ToolResult(summary=f"文件不存在: {path}", error="FileNotFound", skipped=True)

    if path.is_dir():
        return ToolResult(summary=f"目标是目录而非文件: {path}（file.delete 仅删除文件）", error="IsDirectory", skipped=True)

    ok, err = _path_guard(path)
    if not ok:
        return ToolResult(summary=err, error="PathBlocked", skipped=True)

    try:
        path.unlink()
        _log.info("file.delete: 已删除 %s", path)
        return ToolResult(
            summary=f"已删除: {path}",
            state_delta={"file": "deleted", "path": str(path)},
            resource_key=str(path),
        )
    except Exception as e:
        _log.exception("删除文件失败: %s", path)
        return ToolResult(summary=f"删除失败: {path}", error=type(e).__name__)
