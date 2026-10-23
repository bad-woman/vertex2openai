---
title: Vertex2OpenAI Express Adapter
emoji: 🔄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Vertex2OpenAI Express Adapter

Vertex2OpenAI 是一个 **OpenAI API 兼容代理**。它对外提供 OpenAI 风格的 `/v1/chat/completions` 与 `/v1/models` 接口，对内支持两条上游通道调用 **Google Agent Platform（原 Vertex AI）的 Gemini 模型**：

- **Express API Key（标准模式）**：用官方 `google-genai` SDK + 多账号 `Express API Key` 轮询调用（每个 Key 自动探测并绑定各自归属的 GCP Project ID）。
- **Cookie 直连反代（Studio 模式）**：用 Google Cloud 控制台 Cookie + Project ID 直连控制台私有 `batchGraphql` 接口（无需浏览器）。

> 管理控制台 UI 名为 **agentplatform2api**；仓库/镜像名仍沿用 `Vertex2OpenAI`。

---

## 核心功能特性

### 1. 双上游通道 + 跨通道主备自动兜底
- **多 Key 轮换重试**：单次请求遇到 `429 / 500 / 502 / 503 / 504` 或网络断连抖动时，每次退避重试会自动切换到下一个健康的 Express Key（并同步切换该 Key 对应的 Project ID），绝不死磕同一个 Key。
- **跨通道互备（Key ↔ Cookie）**：控制台「重试与退避」内置「允许跨通道兜底重试」开关（默认开启）。主通道选 Express 时，所有 Key 试完一轮可自动切去调 Cookie 兜底；主通道选 Cookie 时，若 Cookie 限流或过期失效，自动无缝切到 Express Key 池接力完成回复。
- **坏凭证自动隔离与红色告警**：当某个 Express Key 或 Cookie 报 `401 / 403 / 项目未开计费` 时，当次请求自动隔离跳过该坏凭证，并在「运行日志」中打印红色高亮告警（精确到第 `#几` 个 Key 与项目 ID），提醒及时删除或更换。

### 2. 管理控制台（日间马卡龙 / 夜间护眼双主题，单文件免构建）
- **仅密码登录**：打开根路径 `/`，输入密码（即 `API_KEY`）即可进入控制台。
- **即添即探**：添加或修改 Express Key 移开输入框即自动热保存，并后台自动探测该 Key 所属的 GCP Project ID。
- **严格分区的模型参数面板**：
  - **① 按模型参数**：思考强度与原生思考控制、生图分辨率与比例、采样默认值与采样策略、预填充兼容模式与思维链守卫、system/预填充注入，全部支持选定模型后点「保存为该模型专属（★）」。
  - **② 全局基础设施设置**：标准模式 location 区域钉定、输入图片压缩、重试与跨通道兜底、假流式与弱网续传缓冲、调试开关，全局唯一且与 ① 区互不覆盖。
- **实时监控**：实时彩色日志推流（支持点击单条复制）、服务健康度百分比圆环、双通道 Token 消耗统计。

### 3. Gemini 3.x / 4.x 与生图深度适配
- 支持流式（SSE）、假流式保活（防 `gemini-4-argon` 等长思考模型 60s 超时断连）与非流式。
- **弱网可靠续传与首包前取消止损**：内置带序号的 `ResumableBuffer`，支持客户端断线凭 `Last-Event-ID` 或 `?resume_from=` 无损续传；首包产出前若前端手动点击停止，后台 1.5 秒内立即终止上游调用与重试。
- **原生函数调用（Function Calling）**：两条通道均支持工具声明、并行调用、流式分片参数聚合（`streamFunctionCallArguments` / `partialArgs`）及 `thought_signature` 思考签名无损回传。
- **预填充兼容与思维链守卫**：针对 Gemini 3.x / 4.x 拒绝以 `assistant` 轮次结尾的限制，内置 `smart` / `keep_turn` / `minimal` 兼容模式、流式重叠去重与未闭合标签思维链守卫（`prefill_cot_guard`），完美适配 SillyTavern 等酒馆预设。
- **精准 500 错误分类**：将提示词安全拦截（`PROHIBITED_CONTENT` / `SAFETY`）与空候选/思考截断从 500 错误中剥离（不盲目重试），而对真正的上游 `500 INTERNAL` 与网络断连按用户设置次数退避轮询。

---

## 快速开始（Docker Compose）

编辑 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  openai-to-gemini:
    image: ghcr.io/bad-woman/vertex2openai:latest
    container_name: vertex2openai
    ports:
      - "8050:7860"
    volumes:
      - ./data:/app/data
    environment:
      - STATE_DIR=/app/data
      - API_KEY=your_strong_password_and_api_key
      - VERTEX_EXPRESS_API_KEY=your_key_1,your_key_2
    restart: unless-stopped
```

启动服务：

```bash
docker compose up -d
```

浏览器打开 `http://localhost:8050`，输入 `API_KEY` 即可进入控制台。

---

## 环境变量说明

| 变量 | 必填 | 默认值 | 说明 |
|---|---:|---|---|
| `API_KEY` | 是 | `123456` | 保护本代理的 Bearer Token，同时也是**控制台登录密码**。 |
| `VERTEX_EXPRESS_API_KEY` | 否 | 空 | 初始导入的 Express API Key（多个用逗号分隔，仅首次启动导入一次，后续以控制台保存列表为准）。 |
| `STATE_DIR` | 否 | `.` | `web_state.json` 存放目录，Docker 部署务必挂载持久化卷。 |
| `PROXY_URL` | 否 | 空 | 上游请求使用的 HTTP / HTTPS / SOCKS 代理地址。 |
| `SSL_CERT_FILE` | 否 | 空 | 自定义 SSL 证书路径。 |
| `MODELS_CONFIG_URL` | 否 | 仓库 `vertexModels.json` | 自定义远程模型列表 JSON 地址。 |
| `GOOGLE_COOKIE` | 否 | 空 | Cookie 直连模式的初始 Google Cookie（可在控制台热更新）。 |
| `GOOGLE_PROJECT_ID` | 否 | 空 | Cookie 直连模式的初始 GCP Project ID（可在控制台热更新）。 |

---

## 模型列表配置（`vertexModels.json`）

当前支持的模型清单（系统会自动为所有支持搜索的模型附加 `-search` 联网搜索别名）：

```json
{
  "models": [
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-image",
    "gemini-nano-banana-2.1",
    "gemini-3-pro-image",
    "gemini-3-flash-preview"
  ]
}
```

### 模型能力自动适配规则（`app/model_capabilities.py`）
- **采样参数自动剥离**：自 `gemini-3.6-flash`、`gemini-3.5-flash-lite` 及 `gemini-4-*` 起，官方已废弃 `temperature` / `top_p` / `top_k`，代理会自动剥离以防 400 报错；惩罚参数（`presence_penalty` / `frequency_penalty`）全系默认剥离。
- **思考档位（`thinking_level`）**：
  - `3.6-flash` / `3.5-flash` / `3.5-flash-lite` / `3.1-flash-lite`：支持 `minimal / low / medium / high`。
  - `3.7-flash` / `3.8-flash` / `3.1-pro-preview` / `4-argon`：支持 `low / medium / high`（最低合法档为 `low`）。
  - 跑酒馆角色扮演预设时，推荐在控制台将「原生思考控制」设为 **「关闭原生思考」**，自动忽略前端传来的 `reasoning_effort=xhigh`，将思考压到该模型最低合法档并剥离原生思考块。
- **生图模型**：`gemini-3-pro-image`、`gemini-3.1-flash-image`、`gemini-nano-banana-2.1` 均支持全部 15 种官方宽高比（`1:1, 3:2, 2:3, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 9:21, 1:4, 4:1, 1:8, 8:1`）及最高 `4K` 分辨率，生图请求自动启用假流式保活输出。

---

## 调用示例

### 1. 查询可用模型
```bash
curl http://localhost:8050/v1/models \
  -H "Authorization: Bearer your_api_key"
```

### 2. 流式或非流式对话（支持 `-search` 后缀开启 Google 搜索）
```bash
curl http://localhost:8050/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.8-flash-search",
    "messages": [
      {"role": "user", "content": "总结今天科技圈的重要新闻。"}
    ],
    "stream": true
  }'
```

### 3. 生图请求（可通过扩展字段指定分辨率与比例）
```bash
curl http://localhost:8050/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-nano-banana-2.1",
    "messages": [
      {"role": "user", "content": "生成一只在霓虹雨夜屋顶上的赛博朋克黑猫"}
    ],
    "image_size": "2K",
    "aspect_ratio": "16:9"
  }'
```

---

## 本地开发与语法自检

```bash
# 编译检查全部 Python 源码语法
python -m compileall app

# 本地启动服务
cd app
uvicorn main:app --host 0.0.0.0 --port 7860
```