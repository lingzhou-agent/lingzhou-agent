# OpenClaw 到 Lingzhou 迁移方案

## 1. 核心结构差异
- **OpenClaw**: (待确认源格式) 传统扁平文件或单一向量存储。
- **Lingzhou**: `nodes/*.json` (灵魂层/灾难恢复) + `semantic.db` (搜索索引层/SQLite FTS5 + 可选向量列)。

## 2. 迁移原则
- **JSON 优先**: 所有迁移数据必须先格式化为 `MemoryNode` 兼容的 JSON，写入 `lingzhou/memory/nodes/` 目录。
- **索引延迟重建**: 写入完成后，通过 `SemanticMemory.rebuild_index()` 或重启系统自动同步至 DB，严禁直接操作 SQLite。
- **字段映射**: 重点处理 `embedding` (JSON float array)、`tags` (JSON array)、`activation`/`valence` (Ebbinghaus 衰减计算)。

## 3. 执行路径
1. 导出 OpenClaw 原始记忆/技能数据。
2. 编写转换脚本：清洗字段 -> 映射至 MemoryNode 结构 -> 生成 `{id}.json`。
3. 将 JSON 批量放入 `lingzhou/memory/nodes/`。
4. 触发 `rebuild_index()` 验证 FTS5 检索与向量混合检索。
5. 运行 `quality_checker.evaluate_retrieval_quality` 进行召回率验收。