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