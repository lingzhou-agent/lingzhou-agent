# 灵舟 (lingzhou) 文档索引

## 核心文档
- [设计蓝图](docs/ROADMAP-2026.5.15.md) — 架构路线图
- [自驱力设计](docs/self-drive-design.md) — 理论基础与实现
- [判断层](docs/judgment-layer.md) — 判断与路由

## 运行
```bash
# 系统服务
systemctl start|stop|restart|status lingzhou

# 命令行
lingzhou gateway logs tail -f     # 实时日志
lingzhou gateway logs stats        # 统计概览
lingzhou gateway logs errors       # 错误/警告
lingzhou gateway logs wechat       # 微信消息
lingzhou gateway plugin list       # 插件列表
lingzhou gateway restart           # 重启
```

## 工具目录 (46 tools)

### 文件 (4)
`file.read` `file.write` `file.edit` `file.list`
- 原子写入、workspace沙箱、路径穿越检测、大小限制(100k/200k)

### Shell & Process (6)
`shell.run` `shell.capabilities` `process.list` `process.poll` `process.log` `process.write` `process.kill`

### Memory (6)  
`memory.search` `memory.add_wm` `memory.add_semantic` `memory.get_fact` `memory.set_fact` `memory.snapshot`

### Task (9)
`task.add` `task.advance` `task.complete` `task.fail` `task.list` `task.resume` `task.update` `task.wait` `task.plan`

### Web (2)
`web.fetch` `web.search`

### Browser (5)
`browser.navigate` `browser.snapshot` `browser.click` `browser.type` `browser.scroll`

### Media (3)
`image.analyze` `image.generate` `tts.speak`

### Schedule (4)
`schedule.add` `schedule.ack` `schedule.cancel` `schedule.list`

### Meta (5)
`skill.list` `skill.search` `reflect.structural` `failure.dismiss` `exec`

## 设计原则
- **LLM 感知优先**: 所有信号以叙事注入 WM，不机械阻塞
- **可配不硬编码**: 阈值、窗宽、尝试次数均可通过 lingzhou.json 调整
- **自驱非指令**: 好奇心以"内心感知"形式呈现，LLM 自主选择是否响应
