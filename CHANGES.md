## 2026-11-12 Function Call 抗截断词法状态机重构（彻底修复 Python 代码与酒馆前端美化误截断 Bug）

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