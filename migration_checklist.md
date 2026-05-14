# OpenClaw → Lingzhou 灵魂迁移执行清单

## 映射与策略概览
| 源文件 (OpenClaw) | 目标路径 (Lingzhou) | 内容边界 | 适配策略 | 风险等级 |
|---|---|---|---|---|
| workspace/SOUL.md | workspace/SOUL.md | 核心价值观(EMA)、使命 | 合并：保留 Lingzhou EMA 格式，注入 OpenClaw 价值观权重，更新使命描述 | 低 |
| workspace/IDENTITY.md | workspace/IDENTITY.md | 自我认知、持续主体声明 | 覆盖：采用 Lingzhou 持续状态/记忆架构，保留“非工具”核心定位 | 低 |
| workspace/BOOTSTRAP.md | workspace/BOOTSTRAP.md | 心智原则、判断基线 | 融合：将 6 条原则映射至 Lingzhou 认知循环，替换运行时特定术语 | 中 |
| workspace/HEARTBEAT.md | workspace/HEARTBEAT.md | 自检周期、触发条件 | 重构：按 Lingzhou tick/interval 机制重写检查项，移除 OpenClaw 专有调度 | 低 |
| workspace/TOOLS.md | workspace/TOOLS.md | 工具限制、最佳实践 | 追加：合并已知限制，补充 Lingzhou 专属工具链经验 | 低 |
| workspace/USER.md | workspace/USER.md | 用户偏好、互动模式 | 合并：保留现有偏好，增量追加新观察，不覆盖历史 | 低 |
| workspace/DREAMS.md | workspace/DREAMS.md | 长期志向 | 初始化：清空旧占位，按 Lingzhou consolidate 机制重建 | 无 |

## 执行步骤（仅读取/验证，暂不写入运行态）
1. 逐文件 file.read 核对目标侧当前内容，标记需保留/删除/重写的段落
2. 针对 BOOTSTRAP 与 HEARTBEAT，确认 Lingzhou 运行时心跳/自检接口兼容性
3. 生成适配后的草案文件至 _migration_drafts/ 目录供人工复核
4. 用户确认后，按清单顺序执行热加载/替换（优先低风险的 USER/TOOLS，最后 SOUL/IDENTITY）

## 约束与回滚
- 所有写入操作前必须备份当前文件至 _migration_backup/
- 若 state/runtime.db 结构不兼容，暂停迁移并先升级 schema
- 任何单文件迁移失败不影响其他模块，支持独立回滚