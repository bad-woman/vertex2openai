## 2026-11-19 新增 `gemini-4-neon`、模型分类配置驱动与双通道 Function Calling / 思考签名深度修复

### 1. 新增 `gemini-4-neon` & `vertexModels.json` 分类能力驱动（`vertexModels.json`、`app/model_loader.py`、`app/model_capabilities.py`、`app/main.py`、`README.md`）
- **`gemini-4-neon` 完整适配**：纳入四档思考（`low / medium / high / xhigh`，原生默认 `high`）及一元非流式 / 假流式 `64K`（`65536`）输出上限，与 `gemini-4-hydrogen` 同级；
- **向后兼容的分类配置架构**：`vertexModels.json` 新增 `categories` 分组与分类级 `capabilities` 声明，同时保留顶层扁平 `models` 列表确保向下兼容；后续新增同类 Gemini 4 机型仅需在对应分类加入模型名即可自动继承 `xhigh` / 一元上限，无需改动 Python 代码；
- **控制台分组下拉与能力徽章**：`#model-sel` 按分类渲染 `<optgroup>` 分组并新增 `一元上限 64K / 32K` 能力徽章；同步细化生图模型官方宽高比白名单（`gemini-3-pro-image` 10 项、`gemini-3.1-flash-image` 14 项、`gemini-nano-banana-2.1` 15 项）。

### 2. Function Calling、思考签名与结构化输出协议修复（Express SDK & Cookie 双通道）
- **自定义工具与联网搜索冲突防护（`app/synthetic_wrapper.py`、`app/api_helpers.py`、`app/upstreams/express_sdk.py`、`app/upstreams/cookie_proxy.py`）**：
  - 新增 `is_builtin_search_tool()`，带有效 `parameters.properties` 的客户端自定义 `search / browse / web_search` 函数不再被误劫持为内置 Google Search；
  - Express `-search` 后缀请求若已携带 `function_declarations`，自动跳过注入 `google_search`，避免触发 Vertex AI 400（不支持自定义函数与内置搜索同请求混用）；
  - 抗截断沙箱在 `tool_choice="none"` 且对话历史含工具轮次时保留工具声明并仅允许 `compose_reply`，避免历史校验失败。
- **流式并行同名工具调用与 `willContinue=false` 收尾帧（`app/message_processing.py`、`app/upstreams/cookie_proxy.py`）**：
  - 升级 `FunctionCallStreamAssembler.feed()`，支持无 `id` 场景下并行调用同一函数（如同时查多个城市天气）的精准拆包与防误合，同时兼容重复 `name` 收尾帧不产生幽灵空调用；
  - Cookie `_extract_from_results` 识别显式 `{"willContinue": false}` 收尾帧，同时保留空 `functionCall: {}` 上的 `thoughtSignature` 透传。
- **工具返回 Pydantic 解包与多模态工具结果（`app/message_processing.py`）**：
  - 新增 `_extract_tool_response_payload()`，正确解包分段 `ContentPartText / ContentPartImage` 列表，彻底修复对象 `repr()` 泄漏；Express 通道支持将工具返回图片封装为官方 `FunctionResponsePart(inline_data=FunctionResponseBlob(...))`（含安全回退），Cookie 通道保守保持纯文本占位以兼容 `batchGraphql` 协议。
- **思考签名拓扑、孤儿工具自愈与结构化输出（`app/api_helpers.py`、`app/message_processing.py`、`app/upstreams/cookie_proxy.py`、`app/schema_validation.py`）**：
  - 禁止将后续文本/思考增量合并到已带 `thought_signature` 的 Part 上；Cookie `signature_only` 块独立追加不再覆盖前序签名；`sanitize_orphan_tool_turns` 降级孤儿调用时同步清空陈旧 `extra_content / reasoning_content` 并保留重复 `tool_call_id` 响应；
  - Cookie `_cookie_ui_schema` 升级为严格 OpenAPI `Schema` 白名单（支持本地 `$ref` 展开与环保护、可空 `anyOf/oneOf` 折叠、`const` 转 `STRING` `enum`、剥离非法 JSON Schema 键、省略无参 `OBJECT` 的 `parameters`、避免下发未验证的 `propertyOrdering`）；修复 dict 型无名 `tool_choice` 误强制 `ANY`；
  - Express `response_format={"type": "json_schema"}` 优先使用官方 `response_json_schema`（递归剥离 `$schema`）；Cookie 非生图文本模型补齐 `responseMimeType` + `responseSchema`（仅在显式传 `response_format` 时生效）；
  - 双通道全链路（含 `apply_prefill_compat` 预填充兼容）支持 OpenAI `role="developer"` 映射至 `system_instruction`（并修复 Cookie 未知角色误映 `model`）；补齐全量安全拦截 `finish_reason`（`PROHIBITED_CONTENT / RECITATION / BLOCKLIST / SPII / IMAGE_SAFETY` → `content_filter`）与 `MALFORMED_*` → `length` 映射；`schema_validation` 兼容大写 OpenAPI 类型名。

### 3. 续传轮询与假流式用量修复（`app/routes/chat_api.py`、`app/api_helpers.py`）
- **非流式续传轮询返回完整 `payload`（`GET /v1/resumable/{stream_id}`）**：针对 `buf.kind == "json"` 在完成后直接返回 `payload` 与 `status_code`，且仅在实际交付非错误有效 `payload` 后才执行 `mark_delivered()`，解决轮询拿到空 `events: []` 却误标已交付导致同 ID 重试重复调用上游的问题；
- **Express 假流式支持 `stream_options: {"include_usage": true}`**：`_chunk_openai_response_dict_for_sse` 新增 `include_usage` 参数并在调用处透传 `wants_usage(exp_req_obj)` 与守恒校核后的 `usage`，开启时在 `[DONE]` 前下发 `choices: []` 的 `usage` 帧，关闭时保持原帧序列完全不变。

### 4. 抗截断架构重构：单字段 `compose_reply` + 融合低误杀 `SyntheticTailGuard` + 0 字超限自愈（`app/synthetic_wrapper.py`、`app/message_processing.py`、`app/api_helpers.py`、`app/upstreams/cookie_proxy.py`、`app/main.py`）
- **源头减负（砍掉冗余 `done` 字段与焦虑元指令）**：
  - `compose_reply` 精简为**单必填字段 `content`**（Express 在 SDK 支持时采用 `parameters_json_schema + additionalProperties: False`，Cookie 回退标准 `parameters`），彻底消除长文末尾从 `content` 切换到 `,done:true}` 时的闭合悬崖；
  - 控制提示词精简为极简中性措辞，去除易诱发模型自检复读的 `closing the content string` / `done to true` 字眼。
- **融合低误杀尾部保护引擎 `SyntheticTailGuard`（Express & Cookie 双通道通用）**：
  - **根治提前误截断**：流中彻底移除激进的闭合标签（`</div>`）与闭合括号（`]}`）切断规则，多余闭合残片仅在流末（`flush`）且全部括号均未配对时清理；流中通用重复熔断阈值提升至 **`≥12 次 且 ≥600 字`**（1600 字历史回看 + 128 字暂扣尾窗，豁免代码与排版分隔符）；
  - **精准狙击 4 类实锤跑飞**：
    1. 未闭合行内反引号超 `100` 字或跨行时**回溯取消保护**（防止乱码里的孤立 `` ` `` 致盲后续检测），并修复单行三反引号 ```` ```x``` ```` 误判；
    2. 新增 48 字高密度货币/符号乱码检测（`find_symbol_salad`）；
    3. 新增英文自检句尾粘连孤立闭合括号（`_META_TAIL`，支持 `.!}?)` / `.!»}]` / `.!»?>}]` / `.!》?>}]` 并校验 `}` 孤立未配对防误伤代码）与同义词轮换型恐慌循环检测，首句即可 0 泄漏秒断；
    4. 新增句末孤立 `}` + `400` 字多语种收尾念叨观察窗（`_stray_close` + `_is_chatter`），念叨即从 `}` 切除，正常正文则原样放行；
  - **第二次重复调用智能去重 & 截断 JSON 恢复**：`FunctionCallStreamAssembler` 新增流式二次调用前 `40` 字探测（`SAME_DRAFT_PROBE`），自动抑制重复前缀并仅放行新增尾巴；非流式截断字符串 `args` 支持有界内容恢复；中间 `$.content` 分片漏带 `willContinue` 不再提前按 EOF 裁剪或封口；Cookie `_extract_from_results` 严格按 `data == "functionCall"` 区分空收尾帧。
- **消除假 `max_tokens` 误报 & 0 字思考超限自动重写**：
  - 主动切除跑飞尾巴后（含流式、假流式、非流式及 Cookie 通道），统一恢复 `finish_reason="stop"`，避免前端误报达上限截断；撤销上游静默断流伪造 `length`；
  - 控制台调用记录与终端日志透传并明确区分上游真实原因（`MAX_TOKENS` vs `MALFORMED_FUNCTION_CALL`）；
  - 新增 **0 字正文即耗尽 `MAX_TOKENS`（思考占满上限）自动重写 1 次**：Express 真流式、假流式、非流式下自动追加精简思考提示重写，降低空回复报错率。

---

## 2026-11-18 流式抗截断（`compose_reply`）`MALFORMED_FUNCTION_CALL` 源头治本与末尾状态栏闭合标签保护

### 1. 保留双必填门栓 + 移除过严 `parameters_json_schema` 扩展（`app/synthetic_wrapper.py`）
- 保持 `required: ["content", "done"]` 与 `property_ordering: ["content", "done"]` 双阶段收敛门栓，防止长文末尾状态栏未写完提前闭合；
- Express 通道 `compose_reply` 移除易在长文嵌套引号/标签下校验脱轨的 `parameters_json_schema` + `additionalProperties: False`，统一采用 Vertex 原生标准 `parameters`（`types.Schema`）声明，大幅压低 `MALFORMED_FUNCTION_CALL` 触发率；
- 在传输控制指令中显式增加硬性约束：所有已打开的 XML/HTML 标签（`</...>`）、代码围栏与末尾状态栏面板必须在关闭 `content` 字符串及设置 `done=true` 之前完整配对闭合。

### 2. 未出正文前的 `MALFORMED_FUNCTION_CALL` 零等待无感剥壳自愈 & 思考后空正文拦截（`app/upstreams/express_sdk.py`、`app/api_helpers.py`）
- `prepare_express_context` 新增 `disable_sandbox=True` 纯净无沙箱上下文构建（干净剥离 `compose_reply` 声明及传输控制指令，保留预填充与客户端真实工具）；
- 严格区分“已产出思考块（`reasoning_content`）”与“已产出可见正文/真实工具（`has_yielded_visible_body`）”，修复高思考档位下先吐出思考块、正文 0 字即崩工具调用却被漏判为 200 成功空回的 Bug；
- 真流式下若尚未产出任何可见正文就遇到 `MALFORMED_FUNCTION_CALL / MALFORMED_RESPONSE / UNEXPECTED_TOOL_CALL`，不断开客户端 SSE 连接、0 等待就地剥离 `compose_reply` 回退原生流式输出正文，并自动过滤重连流中的重复思考块；假流式与非流式同步支持自动剥壳降级重试；补齐 `MALFORMED_*` → `finish_reason="length"` 映射。

### 3. 修复 `$.done` 过早封口、中断尾窗冲刷与多行状态栏周期豁免（`app/message_processing.py`、`app/api_helpers.py`、`app/synthetic_wrapper.py`）
- `FunctionCallStreamAssembler` 仅在 `not fc_will_continue` 或实锤命中溢出裁剪（`trimmed=True`）时才置 `_synth_eof_sealed = True`，确保 `$.done` 提前到达后尾随分片补发的闭合标签（如 `</affinity>`）无损保留；
- Express 与 Studio 流式在中途/尾跳异常中断退出前，强制冲刷 `flush_synthetic_tail` 滑动尾窗（96～640 字），杜绝已接收在本地内存中的状态栏收尾标签被静默丢弃；
- `analyze_synthetic_tail` 对含合法配对闭合标签（`paired_close_tag`）的片段予以周期裁剪豁免，并在存在未闭合 XML/HTML 容器标签时放宽流中普通重复阈值（`min_reps=18, min_total_len=280`）。

### 4. 修复 Gemini 4 流式与压制思考下 Token 统计被清零或仅剩单个数（`0` / `8`）的 Bug（`app/api_helpers.py`、`app/message_processing.py`）
- **流式跨帧单调 `max()` 保留**：防止上游在流式末尾附带的仅含 `{"traffic_type": "ON_DEMAND"}` 空壳包将前序帧已下发的真实 Token 统计覆盖回 `0`；
- **`total_token_count` 总量守恒校核（`_reconcile_usage_counts`）**：兼容跨 SDK 版本 `model_extra` 驼峰字段读取；当开启压制思考导致 `thoughts_token_count` 仅返回 `0` 或 `8` 标记壳、或流式参数分片下 `candidates_token_count` 仅统计收尾壳时，自动通过 `total_token_count - prompt_token_count` 守恒差额精准还原真实的隐藏思考与正文 Token 数。

---

## 2026-11-16 新增控制台「调用记录」全链路追踪面板 & 修复原生思考 Token 统计漏计

### 1. 控制台「🖥️ 运行日志」新增二级切换「📋 调用记录」明细面板（`app/logger.py`、`app/api_helpers.py`、`app/upstreams/cookie_proxy.py`、`app/main.py`）
- **环形内存队列（`CallRecorder`）**：线程安全保留最近 200 条请求级调用摘要（自动淘汰最旧记录，零磁盘 I/O 开销），并通过 `/api/stats` 的 `records_version` 版本号驱动前端按需增量拉取（无新请求时零重绘、不打断已展开详情行）。
- **精准区分标准模式与 Cookie 模式、拆分「入 / 思 / 正」三项 Token 与 `tok/s` 测速**：
  - **Express 标准模式**：精确提取 Google 官方 `usage_metadata` 的 `prompt_token_count`（入）、`thoughts_token_count`（原生思考）与 `candidates_token_count`（正文输出），同时修复原概览页与日志漏计 `thoughts_token_count` 的问题；
  - **Studio (Cookie) 模式**：针对私有 `batchGraphql` 不回传 `usageMetadata` 的特性，诚实标注 `Cookie 无回传` 并附带实际正文字数；
  - **剥离重试耗时的纯速度算法**：真流式下基于上游首个模型 Token 到达时刻精确记录**真实首字响应时间（TTFT）**与**首字到流结束的纯吐字耗时**，计算真实流式 `tok/s`；假流式与非流式（原生一元整包返回）展示整包耗时与等效均速 `≈ xx tok/s`。
- **精简凭证与状态徽章 + 点击展开重试/错误轨迹**：
  - 支持 `全部 / 成功 / 失败` 三档筛选；自动高亮 `⚠️ 达上限截断`（`finish_reason=length`）；
  - 成功时仅简洁展示最终命中的单一凭证（如 `Key #3` 或 `🍪 Cookie`），多凭证耗尽失败时汇总为 `试 N 凭证均败（止于 Key #X）`，点击任意行即可展开完整退避轨迹与错误诊断。

---

## 2026-11-14 根治长回复流式 O(N²) 帧膨胀变慢、弱网续传缓冲深度加固与原教旨抗截断升级

### 0. 根治酒馆长回复流式「越写越慢」SSE 帧膨胀（降低 30~100 倍带宽与内存占用）
- **问题根因**：Express 真流式 `convert_chunk_to_openai` 原先在**每一个普通增量 token 帧**的 `delta.extra_content.google.ordinary_parts` 中都重复携带从第 1 个字累积至今的全部正文列表，且 `json.dumps` 未启用 `ensure_ascii=False`（每个汉字转义为 `\uXXXX` 占 6 字节），导致 1.2 万字回复的 SSE 响应体平方级膨胀至 **9~15 MB**，并同步撑爆 `ResumableStore` 内存缓冲。
- **收尾单发拓扑 + 相邻分片合并 + 原生 UTF-8 序列化**：
  - Express 与 Studio 双通道真流式普通吐字帧/思考帧**彻底剥离 `extra_content`**，仅在收尾帧（`finish_reason` 发出时，含尾窗冲刷保底帧）或携带真实 `tool_calls` 帧时发送唯一一次 `extra_content`；假流式亦统一移至收尾帧发送；
  - `ToolCallIndexer.record_ordinary_part` 与 Studio `_append_merged_ordinary_meta` 自动将相邻同类型（`text` / `thought`）增量合并为单个紧凑元素，避免上千碎字典膨胀；全链路 SSE 启用 `ensure_ascii=False`。
- **弱网可靠续传缓冲（`app/resumable.py`、`app/routes/chat_api.py`）四处并发与内存漏洞修复**：
  1. **修复同 `stream_id` 重试竞态误删新后台泵任务**：`register_pump` 的 `done_callback` 增加任务身份核验（`self._pumps.get(sid) is done_task`），`_cancel_if_abandoned` 与 `CancelledError` 清理增加缓冲区实例身份核验（`is buf`），彻底杜绝旧取消回调误杀新发起的重试流；
  2. **修复半截断流残次品锁死 30 分钟无法重Roll的 Bug**：新增 `saw_midstream_error` 标记（补齐 `[Studio API 错误]` 识别）——若流在中途断网报错，带 `Last-Event-ID` / `resume_from` 的断线续传仍可拉取剩余帧，而酒馆用户直接重Roll（`start=0`）时自动丢弃残次缓冲并重新请求上游；
  3. **已签收缓冲快速回收与 64MB 全局内存软顶**：客户端已完整收到 `[DONE]`（`fully_delivered=True`）的缓冲在 60 秒后即由清理器释放，不再傻留 30 分钟；`_evict_locked` 优先淘汰已签收缓冲并增加 64MB 总字节上限保护。

### 1. 纯正假流式 / 非流式回归 + 有序双必填字段（`content` + `done`）协议级防未闭合
- **保留原教旨上游一元调用**：
  - **Express 假流式与非流式** 100% 调用 Google 原生一元 `client.aio.models.generate_content`（严禁混入 `stream_function_call_arguments`），由上游一次性完整生成返回，避免流式逐块审查导致的半路敏感词截断；仅在**真流式**下启用 `generate_content_stream` + `stream_function_call_arguments`；
  - **Studio (Cookie) 假流式与非流式** 保持整包 `client.post` 一次性取回且不传 `streamFunctionCallArguments`。
- **有序双必填字段（`content` + `done`）带外收敛机制（`build_emit_tool_declaration`）**：
  - 将 `compose_reply` 从孤立单字段升级为静态固定的 `{"content": string, "done": boolean}` + `propertyOrdering: ["content", "done"]`（100% 保持 Vertex 隐式前缀缓存命中）；
  - **治本防未闭合与防尾部幻觉**：Google 受限解码状态机（FSM）强制模型在写完 `content` 后必须闭合双引号并输出 `, "done": true}`，为模型注意力提供确定性的结构出口台阶，彻底消除单字段下在非流式/假流式/流式中把外层括号幻觉进 `content` 或卡在字符串内未闭合的问题；
  - 真流式下网关一旦观测到 `$.done` 路径信号即确定 `content` 已协议级闭合，立即封口执行尾锚定检查，无需猜任何酒馆预设标签。

### 2. Gemini 4 模型级非流式 / 假流式 `max_output_tokens` 精准限幅与官方 SDK 规范对齐
- **精准区分 `4-hydrogen` (64K) 与 `4-argon` / `4-carbon` (32K)（`clamp_max_output_tokens`）**：
  - 在非流式与假流式（一元 `generateContent`）下，`gemini-4-hydrogen` 上限为 `65536`（64K），而 `gemini-4-argon`、`gemini-4-carbon` 等其他 Gemini 4 模型自动将超出 `32768` 的 `max_output_tokens` 安全夹紧至 `32768`（32K），防止酒馆前端滑块拉满 64K 触发上游 400 超限报错或撞硬顶未闭合。
- **官方 `streamFunctionCallArguments` 支持矩阵对齐**：真流式下对 Gemini 3+ / 4+ 文本模型（含 `3.5-flash-lite`）开启分片，对官方明确不支持的 `gemini-3.1-flash-lite` 与生图模型自动禁用，并支持客户端工具并存与协议报错自动剥离重试。
- **边界修复**：修复 `is_gemini_response_valid` 预填充括号续写误判与空 `compose_reply` 放行 Bug；补齐短回复尾窗冲刷时的 `thought_signature` 保留（`pop_synth_signature`）；修复 Studio 顶层 GraphQL `errors` 解析及假流式/非流式降级时漏剥离 `compose_reply` 控制指令的问题。

### 3. 移除破坏前缀缓存的随机注释注入 & 两档快速轮换退避策略
- **彻底移除 `_build_retry_gen_config` 随机 HTML 注释加盐**：保持所有重试请求提示词 100% 纯净无扰，同时完整保留 `genai.Client` 的 `_CLIENT_CACHE` 连接池复用以节省容器内存、以及 `express_location=global` 默认路由。
- **适配快速模式公共池高频 429 的两档退避（`_compute_smart_retry_wait`）**：不设跨请求长冷却禁闭、不强绑单 Key，保持清爽轮询；单次请求内切换到本轮尚未试过的另一把凭证时仅等 `0.3 ~ 0.6s` 立即换号敲门，所有可用凭证试满一整轮（或仅单凭证）时再按控制台 `retry_backoff_seconds`（带 `0.8 ~ 1.2` 随机错峰抖动）退避进入下一轮。

---

## 2026-11-12 抗截断词法状态机重构、思考签名全链路修复与 Gemini 4 系列规则精简

### 0. 新增 `gemini-4-hydrogen`、`XHIGH` 第四思考档位与 64K 全模式输出适配
- `vertexModels.json` 新增 `gemini-4-hydrogen`（自动支持 `-search` 后缀），`requirements.txt` 升级至官方要求的 `google-genai>=2.30.0` 并移除冗余依赖 `curl_cffi`。
- **完整支持 Gemini 4 官方 `XHIGH` 档位（含工具调用场景）与 64K 全模式输出上限**：`gemini-4-*` 全系开放 `low / medium / high / xhigh` 四档（Express 与 Cookie 均统一按官方大写枚举 `"XHIGH"` 下发），并彻底修复原代码导致 `XHIGH` + 工具调用触发 `MALFORMED_RESPONSE` 的三处协议缺陷：
  1. **修复历史工具消息 Part 顺序颠倒**：原代码在客户端剥离 `extra_content` 时把 `tool_parts`（`function_call`）排在了 `ordinary_parts`（`text`）之前，导致历史每一轮模型回复变成非法的 `[function_call, text]` 倒装结构，诱发 `XHIGH` 深思考模仿该倒装结构而被上游判为 `MALFORMED_RESPONSE`；现已严格修正为 `[ordinary_parts, tool_parts]` 因果顺序；
  2. **客户端工具无损切换至 `parameters_json_schema`**：Express 通道不再将客户端 JSON Schema（如 Zod 的 `const` / `additionalProperties` / `anyOf`）塞入旧版 OpenAPI `parameters` 导致约束被静默丢弃，改用官方 `parameters_json_schema` 无损透传；
  3. **禁止向 `FunctionCall` / `FunctionResponse` 注入伪造的 `call_xxx` ID**，保持历史 `FunctionCall` Part 与模型原始输出严格一致。
- **修复 Express 真流式静默吞掉空回的 Bug（立即报错、不盲目重试）**：当未开启沙箱时上游返回 `MALFORMED_RESPONSE` 或空候选无正文，网关不再将其误判为正常完成并吐出空回复，而是立即返回明确报错（不执行无意义的重复重试）。
- **根治工具思考签名错位漏接与长楼层刷屏（保留全历史哨兵保护防 `MALFORMED_RESPONSE`）**：
  - 实测确认 Gemini 4 在历史旧楼层存在大量无签名 `functionCall` 时会因内部状态机失步必现 `finish_reason=MALFORMED_RESPONSE`（仅吐 14 tokens 即中断）：现已完整恢复对全历史无签名工具调用（每步首个调用）自动补全 `skip_thought_signature_validator` 免校验哨兵的保护；
  - 修复同轮先输出思考/正文再发起工具调用时未将本轮签名绑定给工具、以及收尾块漏接签名的 Bug；新增 `NO_THOUGHT` 状态并将 `SignatureStore` 缓存扩容至 72 小时 / 20000 条，同时将多楼层哨兵提示合并为单行日志汇总，既保证 100% 正常生成又不刷屏。

### 1. 零前瞻确定性词法状态机（`LexicalState`）与代码/前端美化免误伤保护
- **问题根因**：旧版 `sanitize_synthetic_tail` 采用无状态全文正则（如匹配 `]}` / `]})` / `}])}` 及 `Done!` 后空白）并在流式进行期以结构字符重复 `>= 2` 次即断流，导致正常 Python 嵌套括号（如 `foo({"a": [1]})`、多层缩进 `)\n)`）、Markdown 表格（`| --- | --- |`）、连续代码块与酒馆前端美化嵌套闭合标签（`</div></div>`）在流式或非流式下被误判腰斩；若简单放宽正则又会漏放真实的末尾闭合残片。
- **确定性流式词法追踪**：
  - 实时追踪 CommonMark 多行代码围栏（```` ``` ```` / `~~~`，含多反引号嵌套与单行 ```` ```高亮``` ```` 识别）与行内反引号作用域，代码块内括号组合 100% 豁免；
  - 维护代码块外弹性括号配对栈 `()[]{}`、同段直双引号配对及带属性引号感知的 HTML 标签开闭计数器，精准区分「正文合法配对闭合」与「栈归零后凭空多出的外层工具悬空闭合残片」；
  - 支持从预填充（`prefill_text`）无缝继承词法栈状态，完美兼容酒馆预填充美化外框与「继续生成（Continue）」续写半截代码/HTML。

### 2. 流式动态隔离窗口（Adaptive Quarantine）与流中/流末分级修剪
- **流式进行期（In-Flight）**：常规保持 96 字符低延迟滑窗；遇可疑重复或半截工具自检前缀时动态扩容至 640 字符隔离观察，正常宽表格、多层 `</div>` 写完后自动无损放行，仅对实锤工具自检泄漏与真病态死循环熔断止损。
- **流结束期（At-EOF）**：严格锚定全文最末尾 `\Z`，仅切除 100% 未配对的悬空工具闭合尾巴（含未换行闭合围栏粘连 ` ```")} ` 特判），并修复 `_unwrap_nested_json_if_needed` 误剥 ```` ```json ```` 代码块与 Studio 非流式漏冲刷尾窗的问题。
- **思维链守卫精准化（`detect_unclosed_cot_tag`）**：过滤代码块内 `<` `>` 运算符干扰，仅对真正的思考/推理语义标签下发 CoT 闭合指令，不再误把酒馆 `<div>`、`<style>`、`<content>` 等美化或正文标签当成思考块提前闭合。

### 3. 全链路思考签名（`thought_signature`）与工具拓扑深度排查修复
- **修复 Cookie 通道图片免校验哨兵未 Base64 编码导致的 400 Bug**：将 `cookie_proxy.py` 中明文 `"skip_thought_signature_validator"` 修正为标准 Protobuf Base64 编码字符串，解决缓存未命中时回传历史图片报 400 的问题。
- **保护模型生图原始字节与角色隔离**：历史 `model` 轮次中的图片若命中签名缓存，保持原图字节不做 JPEG 有损重压缩，确保加密签名 100% 校验通过；`user` 角色上传的参考图正常压缩且绝不误挂模型思考签名；补齐流式生图时签名落在尾随空文本块（`signature_only`）上的关联捕获。
- **过滤无签名的历史 `reasoning_content`**：当客户端（如开启保存思考的酒馆/NextChat/Cherry Studio）回传历史 `reasoning_content` 但丢弃了 `extra_content` 加密签名时，自动省略无签名的 `thought=True` 块，避免触发 400 签名缺失报错并节省历史 Token。
- **抗截断并存真实工具真签名继承 & 孤儿工具轮次自愈（`sanitize_orphan_tool_turns`）**：流式与非流式下同轮 `compose_reply` 与客户端真实工具并存时，自动将真签名继承给首个无签名真实工具；自动自愈因删楼层、手动停止生成或上下文滑窗截断产生的不配对 `tool_calls` / `tool` 消息，彻底杜绝会话永久卡死 400。
- **流式首包前重试与收尾帧加固**：修复 Studio 通道在压制思考期/尾窗缓冲期断流不自动重试的问题，以及 Express 流式尾窗冲刷后的 `finish_reason` 顺序保底。

---

## 2026-11-10 生图模型多轮修图 thought_signature 恢复与免校验保护、Gemini 4 系列适配

### 1. 生图模型多轮修图/连续追问思考签名全链路保护
- **问题根因**：Gemini 3.x 生图模型（如 `gemini-3.1-flash-image`）多轮对话中，Google 严格强制要求历史模型回复中的图片 (`inlineData`) 必须携带上轮的思考签名 (`thought_signature`)。普通 OpenAI 客户端（酒馆、NextChat 等）仅保存 markdown base64 图片，丢失了非标扩展字段，导致在第二轮追问/二次修图时触发上游 Google `HTTP 400: Image part is missing a thought_signature in content position 2, part position 1`。
- **ImageSignatureStore 图片哈希缓存**：
  - 新增线程安全的 `ImageSignatureStore`（SHA-256 图片字节哈希 ➔ `thought_signature` LRU 缓存）；
  - 上游生成图片时透明捕获并持久化思考签名；
  - 客户端下一轮将历史图片发回时，网关通过 SHA-256 毫秒级恢复原始真实的思考签名，无损保证 Gemini 连贯推理与上下文继承。
- **官方免校验哨兵兜底**：
  - 当缓存过期、服务重启或用户外部注入图片时，自动为模型角色的图片部分补充官方 `skip_thought_signature_validator` 免校验哨兵，彻底杜绝 400 阻断。
- **修正 `_requires_signature` 规则**：
  - 将 `gemini-3.1-flash-image`、`gemini-3-pro-image` 等生图模型正式纳入思考签名强校验序列，双通道（Express SDK 与 Cookie Proxy）全面受益。

### 2. 控制台与移动端交互体验优化
- **手机端导航栏自适应**：主导航栏采用 `repeat(4, minmax(0, 1fr))` 并在窄屏下隐藏 Emoji，确保「运行日志」等 4 个标签完整显示不越界。
- **参数控制台布局修复**：调整单列/双列流顺序为「思考 ➔ 预填充兼容 ➔ 采样 ➔ 生图」。
- **默认配置优化**：生图默认分辨率设为 1K，「多 Key 顺序轮询」默认开启，「Function Call 抗截断传输」样式统一。

### 3. Gemini 4 系列模型适配
- 支持 `gemini-4-` 正则动态匹配，本地自动包含 `gemini-4-carbon`，并支持本地与远程模型智能合并去重。

---

## 2026-10-23 多账号轮询容灾、Key ↔ Cookie 跨通道互备与 500 智能重试体系

### 1. 多 Express Key 轮换与 Project ID 动态感知
- 单次请求重试多 Key 轮替：当请求遇到 429 限流、500 内部错误或网络闪断时，退避重试自动按轮换顺序切换至下一个健康 Key，绝不死磕同一个账号。
- 项目绑定隔离：重试切换 Key 时，动态拼装各 Key 专属的 `projects/{key_pid}/locations/...` 资源路径，防止跨账号 Project ID 污染引发 403 权限拒绝。
- 坏凭证自动隔离与日志红色警示：遇 `401 API_KEY_INVALID` 或项目欠费（`requires billing`）时，自动在该请求周期内隔离坏 Key 并跳过，同时在运行日志打印红色高亮警示（精准定位 Key 序号 `#1/#2` 与项目 ID），提醒管理员及时删除或更换。

### 2. 跨通道主备兜底重试（Key ↔ Cookie 互备）
- 控制台「重试与退避」新增「允许跨通道兜底重试」开关：
  - 主通道为 Express 时，所有 Key 试完一轮后自动无缝切至 Cookie 通道兜底；
  - 主通道为 Cookie 时，遇到限流或 Cookie 会话过期，自动无缝切至 Express Key 列表接力完成回复；
  - 控制台「通道与凭证」页实现双凭证卡片常驻显示，无须切换主通道即可随时查看并维护备用 Cookie。

### 3. 精准 500 错误分类与退避重试优化
- 彻底区分「明文安全拦截（PROHIBITED_CONTENT / SAFETY）」与「服务繁忙 500 / 网络断连」：
  - 提示词拦截、静默空候选或长思考跑飞截断直接归入 400/502 并明确诊断提示，坚决不重试；
  - Google 上游真实的 `500 INTERNAL`、`502`、`503`、`504` 与本地网络 TransportError（`RemoteProtocolError` / `ConnectError`）严格按用户设置的最大重试次数执行退避重试。
- 修复非流式报错在指标监控中被双倍累加计数的 Bug。

### 4. 彻底清理已废弃代码与 Gemini 2.5 残留
- 彻底移除已于 2026-10-16 停用的 Gemini 2.5 系列模型及 `thinking_budget` 分支逻辑；
- 移除未使用的 `StreamingReasoningProcessor`、`extract_reasoning_by_tags`、旧单 Key 获取函数等近 180 行遗留死代码。

---

# vertex2openai 变更与更新日志（CHANGES.md）

基于原版 vertex2openai-main（2026-09-01），持续演进优化。

---

## 2026-10-18 架构级全面升级与十大核心问题修复

### 1. 弱网续传与前端手动取消止损（app/resumable.py、app/routes/chat_api.py）
- 智能断连感知（DetachedRequest）：区分「首包产出前的等待/退避重试期」与「已产出模型正文的生成期」。
  - 当上游尚未产出任何有效内容（处于首包等待或 429/503 退避重试）且前端手动点击停止/取消时，立即返回断连信号并经 1.5 秒防抖取消后台泵任务，彻底解决「前端手动取消后后台仍死磕重试直到次数耗尽」的问题。
  - 当上游已开始产出有效内容时，前端短暂断网不中断后台接收，完整保留弱网续传能力。
- 修复 Last-Event-ID 续传起点 Off-by-One Bug：按 W3C SSE 规范对 Last-Event-ID: k 从 k + 1 开始续传，消除断线重连后重复吐出最后一个 Token 的问题。
- 错误缓冲自动销毁：新增 is_real_output_frame 与 is_error_frame 校验，过滤仅含角色声明的空壳首帧；当流或非流式请求以报错结束且无有效模型输出时，自动从 ResumableStore 移除，防止 30 分钟内重试一直命中死报错缓存。
- 心跳帧隔离：pump_stream 自动过滤 : keep-alive 注释帧与 chatcmpl-keepalive 假流式保活帧，保持事件序号纯净；_evict_locked 与 sweep 淘汰缓冲时同步取消关联后台任务，杜绝孤儿协程泄漏。

### 2. 长思考模型（gemini-4-argon 等）防超时断连与双通道假流式保活（app/http_options.py、app/resumable.py、app/upstreams/cookie_proxy.py）
- SDK 超时扩容：在 app/http_options.py 中显式配置 Google GenAI SDK 的 HttpOptions(timeout=600_000)（10 分钟）与 httpx.Timeout(read=600.0)，解决 gemini-4-argon 在假流式（底层调用非流式 generate_content）长思考超过 60 秒被默认超时掐断的 Bug。
- 续传缓冲等待期主动心跳：sse_from_buffer 在等待慢思考模型输出期间，按配置自动向当前连接发送不占序号的保活帧（开启假流式时发送 chatcmpl-keepalive 数据帧，未开启时每 3 秒发送 : keep-alive 注释帧）。
- Cookie 通道假流式与剥离思考保活：
  - Cookie 通道新增文本模型假流式保活模式（studio_fake_stream），在后台汇总 batchGraphql 响应期间持续向前端发送心跳，并支持随时断连取消；
  - 真流式在开启「关闭原生思考 / 压制思考」（strip_thoughts=True）剥离漫长思考块期间，每 2 秒自动补发 SSE 心跳，防止前端因长思考静默期超时断连。

### 3. 错误诊断体系重构与 Cookie 失效 / 项目计费精准分流（app/upstreams/cookie_proxy.py、app/api_helpers.py）
- 修复 Cookie 失效被误报为项目计费问题的 Bug：
  - 移除旧版 _is_project_error 中因模型路径固定含 projects/xxx 而恒真的判断条件及过宽的 invalid argument 关键词；
  - 精准区分「GCP 项目未开计费（requires billing / service_disabled）」、「Cookie 会话过期（401 / unauthenticated）」与「aiplatform.endpoints.predict denied（明确提示 90% 以上为 Cookie 登录态失效，优先重取 Cookie）」。
- 完整错误详情提取（不再截断或丢弃 details）：
  - 修复 extract_upstream_error 正则遇到内嵌单引号提前截断的 Bug，新增对 APIError.details / fieldViolations 及 GraphQL extensions 的深层提取；
  - 将日志与响应错误长度上限放宽至 2000 字符，并将 traceback 接入控制台日志流；
  - 修复 Cookie 通道在假流式与非流式下遇到 GraphQL errors 时将错误信息拼入正文并误记为成功响应的 Bug。

### 4. 预填充兼容引擎升级与去重强化（app/message_processing.py、app/api_helpers.py、app/upstreams/cookie_proxy.py）
- 思维链守卫（cot_guard）位置优化：在 smart 模式下将守卫指令前置融合至续写引导语中（置于 prefill 之前），确保整条消息严格以用户的 prefill 原文断点收尾，消除因末尾变成规则说明而诱发模型重开标签的问题。
- 多模态 user 消息合并：支持最后一条 user 消息为分段数组（list）时的无损合并。
- 去重引擎升级（strip_prefill_overlap & PrefillDeduper）：
  - 支持短于 8 字符的短标签（如 <cot>、<think>）去重与前导空白/换行容错；
  - 自动识别并剥离模型在输出开头重复开启的未闭合标签，彻底解决 keep_turn 与 smart 模式下的双开标签问题；
  - 修复 PrefillDeduper 在首块仅收到换行符时提前退出的 Bug，并确保真流式下 prefill_text 严格排在原生 reasoning_content 思考块之后发出。

### 5. 流式工具调用参数分片聚合（streamFunctionCallArguments 加固）（app/message_processing.py、app/api_helpers.py、app/upstreams/cookie_proxy.py）
- 新增 FunctionCallStreamAssembler：完整支持 Gemini 3 / Gemini 4 的 partialArgs（RFC 9535 JSONPath）与 willContinue 分片参数流式拼接，完整保留首块 id 与 thought_signature，过滤流末尾空收尾标记块，防止分片函数调用丢失参数或分裂成多个空调用。

### 6. 生图模型搜索后缀与 Nano Banana 2.1 修复（app/routes/models_api.py、app/upstreams/cookie_proxy.py）
- 修复 Cookie 模式下调用 gemini-nano-banana-2.1 报 mediaResolution 枚举值非法的问题（修正为 MEDIA_RESOLUTION_HIGH）。
- 放开所有支持联网搜索的生图模型（gemini-3.1-flash-image、gemini-3-pro-image、gemini-nano-banana-2.1）在 /v1/models 中的 -search 别名生成。

### 7. 控制台交互修复、参数分区重构与马卡龙日夜双模视觉升级（app/main.py、app/config.py、app/express_key_probe.py）
- Express Key 即添即探：热添加新 Key 后直接点击「探测」或「探测全部项目」时，自动同步保存当前 Key 列表并探测归属项目，解决未先点保存就点探测导致新 Key 消失的逻辑错误。
- 模型参数分区严格隔离：将预填充兼容模式（prefill_mode）、预填充压制思考（prefill_suppress_thinking）、思维链守卫（prefill_cot_guard）统一归入「① 按模型参数」区并纳入 PER_MODEL_KEYS，与「② 全局基础设施设置」彻底分离、互不覆盖；修复 refreshModelSelStars 将「＊ 全局默认」覆盖为 __global__ 的显示 Bug。
- 马卡龙配色与柔和夜间护眼模式：
  - 全局升级为柔和马卡龙配色，并新增「🌙 夜间护眼 / ☀️ 日间马卡龙」一键切换（自动记忆、首屏防闪白）；
  - 修复服务健康度轮盘将「成功响应」显示为纯黑色的 Bug，新增实时成功率百分比徽章与动态比例 Token 进度条；
  - 运行日志支持点击任意单条日志直接复制该条内容。

---

## 历史迭代记录（2026-09 ~ 2026-10）

- 2026-10-04（签名轮换修复）：更新 app/cookie_auth.py 中的 STREAM_GENERATE_QUERY_SIGNATURE，修复 QUERY_SIGNATURE_NOT_FOUND 空回复问题。
- 2026-10-02（审计修复）：_resumable_stream 增加 CancelledError 分支清理占位缓冲；修复 resumable_poll 越界游标误标交付问题。
- 2026-09-30（并发竞态补丁）：ResumableStore.create_if_absent 持锁原子占位；轮询读完自动 mark_delivered。
- 第一轮 ~ 第三轮重做：Express Key 多账号管理与 Project ID 自动探测、弱网可靠交付缓冲、惩罚参数默认剥离。