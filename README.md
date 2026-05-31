---
title: Vertex2OpenAI Express Adapter
emoji: 🔄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Vertex2OpenAI Express Adapter

Vertex2OpenAI 是一个 **OpenAI API 兼容代理**。它对外提供 OpenAI 风格的 `/v1/chat/completions` 和 `/v1/models` 接口，对内只调用 **Google Agent Platform / Vertex AI Express Mode 的 Gemini API**。

> 当前版本已经重构为 **Express Mode 专用模式**：不再包含 Pay / Service Account / 标准 Vertex AI 项目模式，也不再维护 `-openai`、`-openaisearch` 这类上游 OpenAI-compatible endpoint 路径。

## 为什么改为 Express Mode 专用

Express Mode 官方调用方式使用 API Key，并且不需要在请求里拼接 `project` 或 `location`。本项目现在统一通过 `google-genai` SDK 初始化：

```python
client = genai.Client(vertexai=True, api_key=VERTEX_EXPRESS_API_KEY)
```

这样可以避免旧实现中为了兼容不同路径而手动发现 project、拼接 `locations/global` 或 `/endpoints/openapi` 所带来的复杂度。

## 功能特性

- **OpenAI 兼容接口**
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- **Express Mode 专用鉴权**
  - 通过 `VERTEX_EXPRESS_API_KEY` 调用 Gemini Express Mode。
  - 支持配置多个 Express API Key，用英文逗号分隔。
  - 支持随机选择或轮询选择 API Key。
- **Gemini 原有能力保留**
  - 普通文本对话。
  - 流式和非流式响应。
  - OpenAI tools / function calling 到 Gemini function calling 的转换。
  - Google Search 增强模型别名：`模型名-search`。
  - Gemini 2.5 / 3.x 推理配置适配。
  - 生图模型配置，包括图片输入压缩、比例解析、`image_config`、图片输出 Markdown data URL 转换。
  - 429 / 503 / 502 / quota / resource exhausted 自动退避重试。
- **中文运行日志**
  - 密钥选择、模型配置、上游调用、重试、错误、Token 统计等信息均使用中文说明。

## 已移除内容

以下内容已在本版本移除：

- `[PAY]` 模型前缀和 Pay / Service Account 调用路径。
- `GOOGLE_CREDENTIALS_JSON`、`CREDENTIALS_DIR` 等 Service Account 配置。
- 自动发现 Project ID 的 hack 逻辑。
- `-openai`、`-openaisearch` 模型后缀路径。
- Google 上游 OpenAI-compatible endpoint wrapper。

如果客户端仍请求 `[PAY]...` 或 `...-openai` / `...-openaisearch`，服务会返回明确的 400 错误，提示改用 Express Mode 普通模型名或 `-search` 模型名。

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|---|---:|---|---|
| `API_KEY` | 是 | `123456` | 保护本代理服务的 API Key。客户端请求本服务时使用 `Authorization: Bearer <API_KEY>`。 |
| `VERTEX_EXPRESS_API_KEY` | 是 | 空 | Gemini Express Mode API Key。多个 Key 用英文逗号分隔。 |
| `ROUNDROBIN` | 否 | `false` | `true` 表示多个 Express Key 按顺序轮询；`false` 表示随机选择。 |
| `FAKE_STREAMING` | 否 | `false` | `true` 时先用非流式请求上游，再向客户端模拟流式输出；图片模型会自动启用假流式保护。 |
| `FAKE_STREAMING_INTERVAL` | 否 | `1.0` | 假流式等待期间发送 keep-alive chunk 的间隔秒数。 |
| `MODELS_CONFIG_URL` | 否 | 空 | 可选远程模型列表地址；为空时直接使用本地 `vertexModels.json`，远程加载失败也会回退到本地配置。 |
| `SAFETY_SCORE` | 否 | `false` | 是否把 Gemini safety ratings 附加到输出中。 |
| `PROXY_URL` | 否 | 空 | 上游 HTTP/HTTPS/SOCKS 代理。 |
| `SSL_CERT_FILE` | 否 | 空 | 自定义证书路径。 |
| `HUGGINGFACE` | 否 | `false` | Hugging Face Spaces 特殊鉴权模式开关。 |
| `HUGGINGFACE_API_KEY` | 否 | 空 | Hugging Face Spaces 模式下返回的内部 API Key。 |

## 本地 Docker 运行

编辑 `docker-compose.yml`，至少设置：

```yaml
environment:
  - API_KEY=your_adapter_api_key
  - VERTEX_EXPRESS_API_KEY=your_vertex_express_api_key
```

启动：

```bash
docker compose up -d
```

默认会把宿主机 `8050` 映射到容器内 `7860`：

```text
http://localhost:8050
```

## Hugging Face Spaces 部署

1. 创建 Docker SDK Space。
2. 上传仓库文件。
3. 在 Space Secrets 中添加：
   - `API_KEY`：保护代理服务的密钥。
   - `VERTEX_EXPRESS_API_KEY`：Gemini Express Mode API Key。
   - 可选：`ROUNDROBIN`、`FAKE_STREAMING`、`PROXY_URL` 等。
4. 等待 Space 自动构建并运行。

## 调用示例

### 查询模型

```bash
curl http://localhost:8050/v1/models \
  -H "Authorization: Bearer your_adapter_api_key"
```

### 非流式对话

```bash
curl http://localhost:8050/v1/chat/completions \
  -H "Authorization: Bearer your_adapter_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "用一句话介绍 Gemini Express Mode。"}
    ],
    "stream": false
  }'
```

### 流式对话

```bash
curl http://localhost:8050/v1/chat/completions \
  -H "Authorization: Bearer your_adapter_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "user", "content": "写一首短诗。"}
    ],
    "stream": true
  }'
```

### Google Search 增强

把模型名改成 `-search` 后缀即可：

```json
{
  "model": "gemini-2.5-flash-search",
  "messages": [
    {"role": "user", "content": "今天有哪些 Gemini API 相关更新？"}
  ]
}
```

## 模型列表配置

本地模型列表位于 `vertexModels.json`：

```json
{
  "models": [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash"
  ]
}
```

`/v1/models` 会在普通模型之外，为非图片 Gemini 模型自动添加 `-search` 别名。不会再添加 `[PAY]`、`[EXPRESS]`、`-openai` 或 `-openaisearch`。

## 关于 429 / quota / resource exhausted

429 通常表示 Express Mode 免费额度、速率限制、共享资源或瞬时容量不足。项目会自动退避重试，但重试不能突破上游配额。建议：

- 降低客户端并发。
- 控制最大输出长度。
- 开启 billing 或升级到更适合生产吞吐的模式。
- 配置多个合法 Express API Key 并启用随机或轮询选择。
- 不要在 Express Mode 中随机拼接 location；Express Mode 官方路径不需要 project/location。

## 开发检查

常用检查命令：

```bash
python -m compileall app
```

如需本地启动：

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 7860
```
