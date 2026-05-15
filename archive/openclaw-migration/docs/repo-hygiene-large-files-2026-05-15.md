# 仓库卫生审计：大文件来源与清理建议

**日期：** 2026-05-15

---

## 1. 结论摘要

当前 `HEAD` 的源码树里，已经**没有**这些大文件：
- `_migration_out/`
- `*.sqlite`
- `*.db`
- `*.wal`
- `*.shm`
- `*.trajectory.jsonl`

但 Git 历史里仍然保留了大量大对象，因此 push 仍可能非常大。

---

## 2. 已确认的事实

### 2.1 当前源码树是干净的

检查结果：
- `git ls-files` 中没有 `_migration_out`、`*.sqlite`、`*.wal`、`*.trajectory.jsonl`
- 当前 HEAD 最大文件主要是源码和测试，不是数据库/轨迹文件

### 2.2 历史里确实存在大文件对象

历史对象显示 `_migration_out/openclaw_to_lingzhou/...` 下曾进入 Git：
- `agents/main/sessions/*.trajectory.jsonl`
- `agents/main/sessions/*.jsonl.bak-*`
- `memory/main.sqlite`
- `memory/main.sqlite.tmp-*`
- `*.wal`
- 各类迁移临时文件 / bak / deleted / reset 文件

这些对象解释了为什么 push 会达到数百 MiB。

### 2.3 明确的历史来源：一次“完整迁移产物”被提交进仓库

从 Git 历史看：
- `315b014` 这个提交曾把 `_migration_out/...` 整体加入仓库
- 后续 `a47de01` 又把它们删掉

这说明问题不是“当前代码又在持续生成并跟踪”，而是：

> **曾经有一次迁移/导入过程，把运行时产物整体带进了源码仓库。**

---

## 3. 程序层面是否有“workspace 边界滞后”？

### 3.1 有，且已开始修正

最初迁移脚本：`archive/openclaw-migration/scripts/migrate_openclaw_full_memory.py`

旧路径：
```python
DST_WORKSPACE = DST_BASE / 'workspace'
IMPORT_DIR = DST_WORKSPACE / 'memory-import'
ARCHIVE_ROOT = IMPORT_DIR / 'openclaw-source'
```

旧行为：
- `archive_sources()` 会把 OpenClaw 来源文件复制到 `workspace/memory-import/openclaw-source`
- `backup_runtime_files()` 会把当前 workspace 文件备份到 `workspace/memory-import/backups`

现已改为：
```python
DST_IMPORTS = DST_BASE / 'imports' / 'openclaw'
IMPORT_DIR = DST_IMPORTS
ARCHIVE_ROOT = IMPORT_DIR / 'source-archive'
```

并且当前运行时目录也已迁移到：
- `~/.lingzhou/imports/openclaw/`
- 不再占用 `~/.lingzhou/workspace/`

### 3.2 这意味着什么

这说明当前程序设计中仍然存在一个边界问题：

> **迁移产物被放在了 workspace 下。**

虽然当前脚本复制的是“原始来源文件”和“workspace 备份”，不是完整 `.lingzhou/state` 或 agent runtime DB，
但从架构边界上说，这仍然不够干净。

### 3.3 为什么说它是“轻度滞后”

因为当前脚本复制的内容主要是：
- 顶层 Markdown（AGENTS/SOUL/USER/TOOLS/HEARTBEAT/MEMORY）
- OpenClaw workspace 的 `memory/` 目录

它**没有**像历史里那次污染那样，去复制整个：
- agents/main/sessions/
- sqlite runtime DB
- WAL 文件
- 大型 trajectory 轨迹

所以：
- **历史污染是重度问题**（整包 runtime 产物进 Git）
- **当前脚本是轻度边界滞后**（import/archive 放错层）

---

## 4. 正确的目录边界

### 应放在 `~/.lingzhou/workspace/` 的
- BOOTSTRAP / IDENTITY / SOUL / USER / TOOLS / HEARTBEAT / MEMORY
- skills/
- 人类可读、可编辑、帮助认知启动和协作的文档

### 不应放在 `~/.lingzhou/workspace/` 的
- 迁移导入原始档案
- 运行时数据库
- 会话轨迹
- 进程状态
- 大日志
- 缓存
- SQLite/WAL/tmp/bak/deleted/reset 文件

### 更合适的位置
- `~/.lingzhou/state/` → 运行时真相（DB / process state）
- `~/.lingzhou/logs/` → 日志
- `~/.lingzhou/cache/` → 缓存
- `~/.lingzhou/imports/` 或 `~/.lingzhou/migration/` → 迁移导入产物

---

## 5. 根因判断

### 根因 A：历史上曾把“完整迁移产物”当成源码工件提交
这是 Git 历史大对象的主要来源。

### 根因 B：迁移脚本曾把 import/archive 放在 workspace 下
这不是造成数百 MiB 历史对象的直接主因，但说明当时：

> **程序对“workspace = 认知窗口，不是运行时后场”的边界定义还不够彻底。**

这一点现已开始修正。
---

## 6. 清理建议

### 第一层：已完成（防继续污染）
- `.gitignore` 已加入：
  - `_migration_out/`
  - `.lingzhou/`
  - `*.sqlite*`
  - `*.db*`
  - `*.wal`
  - `*.shm`
  - `*.tmp-*`
  - `*.trajectory.jsonl`

### 第二层：程序边界修正（已开始）
- 已把 `workspace/memory-import/` 迁移到：
  - `~/.lingzhou/imports/openclaw/`
- 已将运行时 workspace 中的迁移计划/映射文档一并移出
- 当前 workspace 中不再保留迁移目录
- 后续可选：是否保留一份轻量导入摘要索引，由人工决定

### 第三层：Git 历史清理（待做）
- 使用 `git filter-repo` / BFG 清理历史中的：
  - `_migration_out/`
  - `*.sqlite*`
  - `*.wal`
  - `*.trajectory.jsonl`
- 然后 force push 一次干净历史

---

## 7. 最终判断

1. **当前源码树是干净的**
2. **大文件问题主要来自历史污染，不是当前 HEAD**
3. **当前程序的主要边界滞后已开始修正：迁移归档已移出 workspace**
4. **下一步应继续清理剩余运行时兼容残留，再做 Git 历史清理**
