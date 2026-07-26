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

- **Express API Key（标准模式）**：用官方 `google-genai` SDK + `VERTEX_EXPRESS_API_KEY` 调用。
- **Cookie 直连反代模式**：用 Google Cloud 控制台 Cookie + Project ID 直连控制台私有 `batchGraphql` 接口（无需浏览器）。

> 管理控制台 UI 名为 **agentplatform2api**；仓库/镜像名仍沿用 `Vertex2OpenAI`。
>
> 📌 **说明**：Google 已将 Vertex AI 更名为 **Agent Platform**，文档表述已同步（历史标识符如环境变量 `VERTEX_EXPRESS_API_KEY`、SDK 参数 `vertexai=True`、文件名 `vertexModels.json` 为兼容性保留）。**无头浏览器（Playwright）模式已移除**，由更轻量的 Cookie 直连模式完全替代。

## 功能特性

- **双上游通道，一键切换**
  - Express API Key：多 Key 随机或轮询调用官方 SDK。
  - Cookie 直连反代：Cookie + SAPISIDHASH 直连 `batchGraphql`，走网页端配额（含最新预览模型），真流式、防 60s 超时。**注意：走的是私有接口，见下方风险提示。**
- **OpenAI 兼容接口**：`GET /v1/models`、`POST /v1/chat/completions`。
- **管理控制台（浅色风格，单文件，免构建）**
  - **仅密码登录**：打开根路径 `/`，输入密码（即 `API_KEY`）即可，无需账号。
  - 标准模式 / Cookie 直连模式在线一键切换。
  - 在线热更新并保存 Google Cookie 与 Project ID；智能解析 `Cookie-Editor` 导出的 JSON / Header String；自动从整条控制台 URL 提取 Project ID。
  - **模型参数面板**：按所选模型显示其支持能力，并在线调整思考强度、生图分辨率与比例、采样默认值、输入图压缩、重试、假流式/轮询/安全分显示、预填充兼容模式（详见下文）。
  - **实时监控**：运行日志推流、健康度图表（成功/错误/拥堵重试）、Token 消耗统计（两条通道均计入）。
- **Gemini 能力与适配**
  - 文本对话、流式（SSE）与非流式。
  - OpenAI tools / function calling ↔ Gemini function calling 适配（含 Gemini 3.x 多轮所需的 thought signature 编解码）。**注意：函数调用仅在 Express 通道支持；Cookie 直连通道不下发函数声明。**
  - **安全分类对齐**：两条通道下发同一套安全设置（`HARM_CATEGORY_HATE_SPEECH`、`DANGEROUS_CONTENT`、`SEXUALLY_EXPLICIT`、`HARASSMENT`、`JAILBREAK`），阈值最宽松，避免通道间行为不一致。
    > 📌 **更正**：早期版本曾把"有思考、正文为空"归因于 Cookie 通道缺少 `HARM_CATEGORY_JAILBREAK`。这个解释是错的——按官方文档，**越狱分类器默认就是关闭的**，要打开必须显式把该分类的阈值设成具体的拦截值；不下发它不会启用任何过滤，下发 `OFF` 也只是空操作。该现象的真实成因见下方"3.6-flash 只返回思考"一节（前端恒发 `reasoning_effort=xhigh` 导致原生思考在 HIGH 档跑飞/被截断）。
  - **上游错误如实透传**：模型在当前项目/区域不可用（404）、权限不足（403）、参数非法（400）等，会以对应 HTTP 状态码 + OpenAI 错误格式返回，而非笼统的 500。
  - Google Search 增强：**文本模型**在模型名后加 `-search` 后缀按需开启。
  - 自动保留 Gemini 思考过程（Thinking），以 `reasoning_content` 返回。
  - 生图模型：输入图压缩、按模型的比例白名单校验、4K 等分辨率、图片输出转 Markdown data URL；生图强制"假流式"整块输出，避免超大 base64 卡死前端。
  - **按模型能力自动裁剪参数**：不同模型支持的参数不同，代理会自动移除目标模型不支持的参数以避免 400（详见"控制台与模型参数"）。
  - **自动退避重试**：两条通道均内置 429/拥堵自动退避重试，**次数与间隔可在控制台调整**（默认约 10 次）。流式模式下，连接建立即发送 SSE 心跳、且退避等待期间持续发送心跳，避免前端因长时间无数据（如 3.1-pro 频繁 429）而**超时断开**。
  - **预填充智能兼容**：自动处理"以 model 轮次结尾"被新模型拒绝（400）的问题（详见下文）。
  - **断连即停**：客户端断开后立即停止上游调用与重试。
- **中文运行日志**：密钥轮询、上游调用、重试退避、权限报错、Token 统计等均为中文实时说明。

## 快速开始（本地 Docker）

编辑 `docker-compose.yml`，设置初始环境变量：

```yaml
environment:
  - API_KEY=your_adapter_api_key
  - VERTEX_EXPRESS_API_KEY=your_vertex_express_api_key
```

启动：

```bash
docker compose up -d
```

默认将宿主机 `8050` 映射到容器 `7860`。浏览器打开控制台并用密码（`API_KEY`）登录：

```text
http://localhost:8050
```

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|---|---:|---|---|
| `API_KEY` | 是 | `123456` | 保护本代理的 Key，同时也是**控制台登录密码**。客户端请求用 `Authorization: Bearer <API_KEY>`。 |
| `VERTEX_EXPRESS_API_KEY` | 否 | 空 | Gemini Express Mode API Key，多个用英文逗号分隔。标准模式使用。 |
| `ROUNDROBIN` | 否 | `false` | 多 Express Key 轮询(`true`)或随机(`false`)。可在控制台热改。 |
| `FAKE_STREAMING` | 否 | `false` | 先非流式取回再模拟流式；生图模型始终强制启用。可在控制台热改。 |
| `FAKE_STREAMING_INTERVAL` | 否 | `1.0` | 假流式等待期间 keep-alive 间隔秒数。可在控制台热改。 |
| `MODELS_CONFIG_URL` | 否 | 仓库 `vertexModels.json` | 远程模型列表地址；改远程文件即可刷新，无需重部署。 |
| `SAFETY_SCORE` | 否 | `false` | 是否把 Gemini safety ratings 附加到输出。可在控制台热改。 |
| `PROXY_URL` | 否 | 空 | 上游 HTTP/HTTPS/SOCKS 代理。 |
| `SSL_CERT_FILE` | 否 | 空 | 自定义证书路径。 |
| `GOOGLE_COOKIE` | 否 | 空 | Cookie 直连模式的 Google Cookie（初始值，后续可在控制台更新）。 |
| `GOOGLE_PROJECT_ID` | 否 | 空 | Cookie 直连模式的 Project ID（初始值，后续可在控制台更新）。 |
| `EXPERIMENT_FLAGS` | 否 | 空 | 可选：batchGraphql 的 `experimentFlagsBinary`，一般无需设置。 |
| `STATE_DIR` | 否 | `.` | `web_state.json` 的存放目录。用 Docker 时请指向挂载卷，否则重建容器会丢失全部设置与 Cookie。 |
| `ALLOW_DEFAULT_KEY` | 否 | 空 | 仅用于在公开托管环境（如 HF Space）临时放行默认 `API_KEY`，正常部署不要设置。 |

> 提示：`ROUNDROBIN`、`FAKE_STREAMING(_INTERVAL)`、`SAFETY_SCORE` 等环境变量仅作为**初始值**，运行时以控制台设置为准。

---

## 行为变更说明（整改后）

按 `REFACTOR_PLAN.md` 完成整改后，以下行为与旧版本不同，升级时请注意：

| 项目 | 旧行为 | 新行为 |
|---|---|---|
| **思考档位兜底** | 非法档位一律抬到 `high`（Pro 上选 minimal 反而变 high） | 就近**向下**夹取：Pro 选 minimal → `low` |
| **`retry_max` 语义** | Express 通道当作总次数，设 0 时一次请求都不发 | 统一为「重试次数」，总请求数 = `retry_max + 1`，设 0 仍请求一次；取值钳到 0–50 |
| **并行函数调用（流式）** | 只发第一个，`index` 恒为 0 | 全部下发，`index` 跨 chunk 稳定递增 |
| **思考签名** | base64 拼进 `tool_call_id`（上千字符，易被前端截断） | 短 id（≤40 字符）+ 进程内旁路缓存；旧格式仍可解析；取不回时降级为官方 `skip_thought_signature_validator` 哨兵 |
| **Cookie 通道 + 函数调用** | 静默把 `role=tool` 折成 model，发出错乱历史 | 入口返回 400，提示切换到标准模式 |
| **Cookie 通道输入图** | 不压缩、不支持 http(s) 图片、不解析正文内联图 | 与标准通道一致（压缩开关对两条通道都生效） |
| **`stop` 字段** | 只接受数组，传字符串 422 | 字符串/数组都接受 |
| **`logprobs` 字段** | 按 Gemini 语义当整数 | 兼容 OpenAI 的 `logprobs: bool` + `top_logprobs: int` |
| **Express 流式 usage** | 从不下发，客户端显示 0 | 支持 `stream_options.include_usage` |
| **控制台 Cookie 回显** | 明文返回完整 Cookie | 仅返回掩码；输入框留空＝保持原 Cookie 不变 |
| **登录** | 无限速、会话 token 为确定值 | 失败 3 次后指数退避；随机会话 token，可单独失效 |
| **状态文件** | 每次读设置都同步读盘、非原子写 | 内存优先 + 原子写 + 权限 0600 + 支持 `STATE_DIR` |
| **文本保真** | 所有消息的多空格/缩进被压平 | 仅在确实抽走内联图片时才压平 |
| **2.5 Flash-Lite 思考预算** | 下限按 0 处理 | 下限 512（`0` 仍表示关闭） |
| **生图比例 `9:21`** | 在白名单里（无官方出处） | 已移除，落到"由模型决定" |

新增两个控制台开关：**思考签名内嵌 tool_call_id**（默认关，多副本部署才需开）与**生图下发 system 指令**（默认关，开启前请真机验证）。

新增 `scripts/check_models.py`：打印各模型的能力判定，用于对着官方文档逐列核对。

---

## 控制台与模型参数

控制台"模型参数"面板可在线调整全局默认值，并按所选模型显示其支持情况。可调项：思考强度（3.x 档位 / 2.5 预算）、生图分辨率与默认比例、采样默认值（temperature/top_p/max_tokens）、输入图压缩（开关/边长/质量）、重试次数与退避、假流式/轮询/安全分显示、预填充兼容模式与"预填充时压制原生思考"开关、续写指令模板、Cookie 调试日志。

### 参数优先级

**单次请求 > 该模型专属 > 控制台全局默认 > 内置默认。**

- 有请求级写法的参数：请求体字段优先。标准字段 `temperature`/`top_p`/`max_tokens` 等，扩展字段 `reasoning_effort`、`thinking_budget`、`image_size`、`aspect_ratio`/`ar`。客户端不传时依次回退到"该模型专属 → 控制台全局 → 模型默认"。
- 两个例外：
  1. **"模型不支持"优先级最高**（最后一步裁剪）：目标模型不支持的参数，无论来自请求、专属还是全局都会被移除（避免 400）。
  2. **全局项无请求级/专属写法**：图压缩、重试、假流式、轮询、安全分显示、预填充模式与压制开关、续写模板、Cookie 调试仅由控制台全局决定。

### 按模型单独保存参数（per-model overrides）

模型参数面板顶部选择模型后，可为**当前所选模型**单独保存专属值——支持覆盖的三类：**思考（档位/预算）、生图（分辨率/比例）、采样默认（temperature/top_p/max_tokens）**。

- **保存为该模型专属**：只把上述三类字段存成该模型的专属配置；下拉框中该模型名后会显示 `★`，并出现"已有专属参数"徽章。
- **清除该模型专属**：删除该模型专属配置，回退到全局默认。
- 底部"保存设置"按钮保存**全局默认 + 基础设施项**；当所选模型已有专属配置时，它**不会**用面板里显示的专属值覆盖全局（避免误操作），只保存基础设施项。
- 基础设施项（图压缩/重试/假流式/预填充模式与压制/安全分/Cookie 调试）不支持按模型覆盖，始终全局唯一。

### 按模型能力自动裁剪（`app/model_capabilities.py`，依据官方文档）

- **采样参数弃用**：自 **Gemini 3.6 Flash / 3.5 Flash-Lite 起（及所有更新/未来模型）**，`temperature`/`top_p`/`top_k` 已废弃（现被忽略、未来返回 400），代理会**自动移除**；更早的 3.x（如 3.0–3.5 非 lite）仍可用，但官方建议保持默认。
- **`candidate_count`**：所有 Gemini 3.x 不支持，自动移除。
- **思考**：Gemini 3.x 用 `thinking_level`（`minimal`/`low`/`medium`/`high`，不可完全关闭；各模型默认不同，如 3.6-flash=medium、pro=high、flash-lite=minimal）；Gemini 2.5 用 `thinking_budget`（`-1` 动态；2.5-flash 可设 `0` 关闭，2.5-pro 最低 128）。
  - **原生思考控制（`native_thinking_mode`）**——控制台"思考强度"卡片的下拉，支持"保存为该模型专属"：
    - **跟随请求（默认）**：用前端发来的 `reasoning_effort`。⚠️ SillyTavern 等前端常在**每次请求都发 `reasoning_effort`（如 `xhigh`）**，会覆盖你在控制台设的档位。
    - **关闭原生思考（角色扮演推荐）**：忽略前端 effort，把档位压到该模型最低（3.x=`minimal`、2.5-flash 预算 `0`、2.5-pro `128`），并**不返回思考**。
    - **强制用上方档位**：忽略前端 effort，用你在卡片里选的档位（返回思考）。
  - 🎭 **酒馆预设“卡原生思维链”一键配置**：许多预设把思维链写在 **system 提示**里（不是预填充），并恒发 `reasoning_effort=xhigh`。把"原生思考控制"选 **“关闭原生思考”** 即可（可用"保存为该模型专属"只对 3.6-flash 生效），让预设自己的思维链接管。
  - ⚠️ **重要（Studio/Cookie 通道实测）**：`batchGraphql` 私有接口**会忽略 `includeThoughts=false`**——即使设了也照样回传思考。因此 Cookie 通道在"关闭原生思考"时会**在响应侧主动剥离思考块**，并**把档位压到 `minimal`**（这才是真正减少原生思考、避免重预设在思考阶段被截断/无正文的关键）。标准（Express）通道由 SDK 原生支持不返回思考。
  - 🩺 **3.6-flash 在 Studio 只返回思考、无正文（`FINISH_REASON_UNSPECIFIED`）怎么办**：真机定案——这是**原生思考在 HIGH 档跑飞/被截断**（SillyTavern 恒发 `reasoning_effort=xhigh` 覆盖了控制台档位），**不是**安全策略/`HARM_CATEGORY_JAILBREAK` 的问题（已用含/不含 jailbreak 的多组对照验证）。**解决：把"原生思考控制"设为"关闭原生思考"**（对 3.6-flash 用"保存为该模型专属"），即忽略前端 `xhigh`、压到 `minimal` 并剥离原生思考——真机验证可稳定输出正文与预设自带的思维链。仅设"思考档位=minimal"无效，因为会被前端 `xhigh` 覆盖。
- **生图**：剥离全部采样参数；`response_modalities=["TEXT","IMAGE"]`；**两个生图模型比例白名单不同**（pro-image 10 种；flash-image 15 种，含 `1:4/4:1/1:8/8:1/9:21`），控制台按所选模型过滤，后端也会校验，选到不支持的比例会**自动回退为"由模型决定"（不报错）**。
- **预填充（重要，专为 SillyTavern 等酒馆预设优化）**：Gemini 3.x 拒绝以 `model` 轮次结尾的请求（返回 `Requests ending with a model turn are not supported.`）。代理内置"预填充智能兼容"（`smart`/`minimal`/`off`，默认 `smart`，控制台可切）：
  - **按模型能力自动选策略**：2.5 及更早模型允许以 model 轮次结尾 → **原生透传**，模型直接续写你的预填充，最忠实；3.x 拒绝 → 自动把末尾 assistant 预填充转成末尾 user 的"续写指令"（模板可在控制台自定义）。两种情况都会把预填充文本**拼回输出开头**，并对模型复述的重叠部分**自动去重**。
  - **预填充时压制原生思考（"卡思维链"，默认开启）**：酒馆预设通常自带思维链，靠预填充卡掉模型原生思考、让预设的思维链接管。开启后，检测到预填充即把思考压到该模型最低并**不回传思考**：3.x 压到 `minimal`（`pro` 无 minimal 则 `low`，官方规定 3.x 无法完全关闭思考）；2.5-flash 预算设 `0` **完全关闭**、2.5-pro 降到最低 `128`。**单次请求显式传 `reasoning_effort` / `thinking_budget` 时不压制**（请求优先）。可在控制台关闭此开关恢复模型原生思考。
  - **与模型名无关，新模型自动生效**。
- **新增/未来模型**：按家族/版本模式自动归类；未知/未来型号按"最新代"前向安全处理（自动移除已废弃采样参数、走预填充兼容）。

---

## Cookie 直连模式配置指引（支持手机与电脑）

> ⚠️ **使用前请先读这段风险提示**
>
> Cookie 直连模式的原理是：用硬编码的网页客户端 key 和固定的 `querySignature`，
> 冒充 Google Cloud 控制台前端去调用其**私有** `batchGraphql` 接口。因此：
>
> - **无兼容性承诺**：这是内部接口，Google 改一次 `querySignature` 或参数结构就会全线失效，且不会有任何公告。
> - **条款风险**：以自动化方式访问非公开接口，可能与 Google Cloud 的使用条款相冲突，存在账号被限制或处置的风险。请仅用自有账号、自担风险。
> - **凭证敏感度极高**：配置的 Cookie 含 `__Secure-1PSID` 等完整会话凭证，等价于该 Google 账号的完整访问权。请勿把本服务部署到公开可访问的地方，务必设置强 `API_KEY`。
>
> 如果你需要的是稳定、可长期依赖的方案，请使用标准模式（Express API Key）。

在控制台切换到 **Agent Platform Studio (Cookie 直连反代)**，需配置 **Cookie** 与 **Project ID**：

### 1. 获取完整 Google Cookie
关键会话凭证（如 `__Secure-1PSIDTS`、`__Secure-1PSID`）带 `HttpOnly`，无法用书签脚本提取，需按下述方式获取：
- **电脑端**：登录 [Google Cloud Console](https://console.cloud.google.com) → 按 **F12** → **Network** → 刷新页面 → 点任意成功请求 → 复制 **Request Headers** 里的 `Cookie:` 整段，粘贴到控制台。
- **手机端**：iOS(Safari) 或 Android(Kiwi) 安装 `Cookie-Editor` → 登录控制台 → 插件 **Export** 为 **Header String** 或 **JSON** → 整段粘贴，系统自动解析。

### 2. 获取 Project ID
- 从控制台顶部项目选择器复制，或直接把含 `?project=xxx` 的整条 URL 粘贴到输入框，系统自动提取。

> ⚠️ **关于 Cookie 有效期**：通常较为持久——只要不退出登录、不修改密码、Google 未主动失效会话，一般可维持**数周甚至更久**（实测可用一个月以上），**并非只有 1~2 小时**。仅当接口确实报 `Permission Denied` / `predict denied` 时，重新获取并到控制台保存激活即可。

---

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
      {"role": "user", "content": "用一句话介绍 Gemini。"}
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

### Google Search 搜索增强（文本模型）

在模型名后加 `-search` 后缀：

```json
{
  "model": "gemini-2.5-flash-search",
  "messages": [
    {"role": "user", "content": "今天有哪些 Gemini API 相关更新？"}
  ]
}
```

### 单次请求覆盖参数（任意前端可用）

通过请求体额外字段按需覆盖控制台默认：

```json
{
  "model": "gemini-3-pro-image",
  "messages": [{"role": "user", "content": "生成一只赛博朋克猫"}],
  "image_size": "2K",
  "aspect_ratio": "16:9"
}
```

（文本模型可用 `reasoning_effort`：`low`/`medium`/`high`，或 2.5 用 `thinking_budget` 整数。）

> **注**：`temperature`/`top_p`/`max_tokens` 等标准字段、以及上述扩展字段，其优先级恒高于"该模型专属"与"控制台全局"设置。若要为某模型设持久默认值又不想每次请求都带，请用控制台的"保存为该模型专属"。

---

## 模型列表配置

默认模型列表在远程 `MODELS_CONFIG_URL` 或本地 `vertexModels.json`：

```json
{
  "models": [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-image",
    "gemini-3-pro-image",
    "gemini-2.5-pro",
    "gemini-2.5-flash"
  ]
}
```

> ⏰ **停用时间线（核对于 2026-07-26，请自行复核官方 deprecations 页）**
> - `gemini-2.5-pro` / `gemini-2.5-flash` / `gemini-2.5-flash-lite`：**2026-10-16 停用**。届时 `thinking_budget` 分支与"2.5 原生预填充透传"路径将没有活模型，可整体删除。
> - `gemini-3-flash-preview`：官方推荐替代为 `gemini-3.5-flash`，但**该模型目前仍可正常调用**（2026-07-26 用 Express Key 实机验证通过），因此**保留在默认清单中**；待官方正式停用后再移除。
> - `gemini-3.1-flash-image` / `gemini-3-pro-image`：GA 版本（不带 `-preview`），对应的 `-preview` 版本已于 2026-06-25 停用。

`/v1/models` 会自动为**非生图**的 Gemini 模型生成带 `-search` 后缀的别名。新增模型只需把 ID 加进此列表即可（能力自动归类，无需改代码）。

---

## 关于 429 报错与并发控制

429（Resource Exhausted）常因上游限额不足或请求频率过高。项目已内置退避重试，另建议：
- 控制客户端并发频率。
- 适当减小最大输出 Token。
- 配置多个有效 Express Key 并开启轮询。
- 及时更新失效或权限受限的 Google Cookie。

---

## 后续升级与扩展

- **新增模型**：把模型 ID 加入 `vertexModels.json`（或远程 `MODELS_CONFIG_URL`）。`model_capabilities.py` 按**家族/版本模式**自动归类（思考方式、采样裁剪、生图比例/分辨率、预填充），**未知/未来型号按"最新代"前向安全处理**。基本即插即用。
- **迁移到 Interactions API**：代码按上游通道解耦（`app/upstreams/` 下各类实现 `BaseUpstream`；能力判定、消息转换、参数构建均可复用），新增一个 `InteractionsUpstream` 并在路由层接入即可。

  现状（核对于 2026-07-26）：
  - **Gemini Developer API 侧**：Interactions API 已于 2026-06 **GA**，官方推荐新项目使用；`generateContent` 被标为 legacy 但继续完整支持。
  - **Agent Platform 侧**：Interactions API 仍标注为 **experimental**。
  - **Express 模式**：REST 面只有 `countTokens` / `generateContent` / `streamGenerateContent` —— **这才是本项目暂不迁移的直接原因**（本项目走的正是 Express 通道）。

---

## 本地开发与检查

```bash
# 语法检查
python -m compileall app

# 本地启动
cd app
uvicorn main:app --host 0.0.0.0 --port 7860
```
