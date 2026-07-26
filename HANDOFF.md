# 项目交接说明（HANDOFF）

> 这是为「换新对话继续改进」准备的交接文档。新对话里上传本项目 zip，并说明「先读 HANDOFF.md 了解现状，然后继续改进」即可。改完可删除本文件。

## 一、项目是什么
OpenAI 兼容代理（仓库名 `Vertex2OpenAI`，控制台 UI 名 `agentplatform2api`）。对外 `/v1/chat/completions`、`/v1/models`；对内两条上游调用 Google Agent Platform（原 Vertex AI）的 Gemini：
- **Express API Key**：官方 `google-genai` SDK（`vertexai=True`）+ `VERTEX_EXPRESS_API_KEY`。
- **Cookie 直连反代**：Cloud Console Cookie + SAPISIDHASH 直连私有 `batchGraphql`（无需浏览器）。

## 二、运行 / 测试
- **需 Python 3.11**（代码用了 `str | None` 联合语法；3.9 无法导入）。
- 依赖：`app/requirements.txt`。语法检查：`python -m compileall app`。
- 启动：`cd app && uvicorn main:app --host 0.0.0.0 --port 7860`；浏览器开 `/`，用密码（`API_KEY`，默认 `123456`）登录控制台。
- 无官方凭证时无法端到端联调；建议本地用真实 Express key / Cookie 复测。

## 三、关键模块
- `app/model_capabilities.py`：按**模型家族/版本**判定能力（思考 level/budget、采样裁剪、生图比例/分辨率白名单、`sampling_advice`、`_temp_deprecated()`）。**新增模型只改 `vertexModels.json`，自动归类；未知/未来型号按“最新代”前向安全处理。**
- `app/message_processing.py`：OpenAI↔Gemini 转换、输入图压缩、function call 的 `__thought__` 思考签名编解码、`apply_prefill_compat()`（预填充兼容）。
- `app/api_helpers.py`：`create_generation_config`（末尾按模型 `sanitize_sampling`）、`execute_gemini_call`（真流式/假流式/非流式、断连即停、prefill 拼接、重试计数）。
- `app/upstreams/express_sdk.py`、`app/upstreams/cookie_proxy.py`：两条上游，均接入能力矩阵 + 预填充 + 重试/错误计入大盘。生图强制假流式（整块 base64 单发）。
- `app/main.py`：FastAPI + 单文件控制台 HTML（浅色 Vercel 风）+ **仅密码登录（cookie 会话）** + `/api/settings`、`/api/capabilities`、`/api/cookie`、`/api/login|logout`、`/stream-logs`。
- `app/runtime_state.py`：持久化运行时 `settings`（`web_state.json`）。`app/config.py`：`DEFAULT_SETTINGS`。

## 四、已完成（阶段 1 + 2 + 3）
- **阶段1 bug 修复**：真流式中途重试去重、工具消息 JSON 优先级崩溃、空回复有效性检查、非流式诊断索引、流式 n>1、工具往返无签名兼容、`json_schema` 结构化输出、CORS、删死代码（`stream_engine`）。
- **删除无头浏览器 / Playwright**；术语 Vertex→Agent Platform；**断连即停**。
- **阶段2**：`model_capabilities` 能力矩阵；控制台重做（浅色、模型参数面板、仅密码登录、退出按钮）；参数化（思考 / 生图分辨率+比例 / 采样默认 / 图压缩 / 重试 / 假流式·轮询·安全分开关 / 预填充模式）；优先级 **单次请求 > 控制台 > 默认** + 按模型自动裁剪；Cookie 通道重试/错误计入大盘；数据大盘标题改 `agentplatform2api`。
- **阶段3（本轮，含真机联调，真实 Key+Cookie 全通道验证）**：
  1. **预填充"卡思维链"增强**（原问题1）：新增"预填充时压制原生思考"开关（默认开）——3.x 压到最低档且 `includeThoughts=false`，2.5-flash 预算 0 全关、2.5-pro 降 128；2.5 及更早改为**原生透传**（不转指令，模型直接续写）；续写指令模板控制台可编辑；预填充拼回输出开头时**自动去重**（`strip_prefill_overlap` + 流式 `PrefillDeduper`）。三通道（Express 流/非流、Cookie 流/非流/生图）全部接入。
  2. **Cookie 3.6-flash 无正文修复**（原问题2，真机定位）：根因是 Cookie 通道 `safetySettings` **缺 `HARM_CATEGORY_JAILBREAK`**（Express 通道一直有），越狱预设正文被拦、思考照常流出。已补该分类（真机验证正文恢复）。附带修复：`_extract_from_results` 过滤 `*_UNSPECIFIED` 枚举默认值（batchGraphql 每块都带，旧白名单会误吞/新全量透出会误判）；空响应/只有思考时**不再静默关流**，给出明确提示 + 自动落原始响应样本诊断日志（新增 `cookie_debug` 开关打印出站 `generationConfig`）。
  3. **Cookie token 统计**（原问题3）：私有接口不回传用量，已停止 Cookie 通道打 `💰` 统计行 → 大盘 token 只计标准通道；Cookie 成功数改由 `stats.add_success()` 单独计入（大盘标题标注"仅标准 Express 通道计入"）。
  4. **按模型单独保存参数**（原问题4）：`settings.model_overrides = { 模型ID: {PER_MODEL_KEYS} }`；`app_state.get_effective_settings(model)` 合并（专属>全局）；三通道构建改用它；控制台加"保存为该模型专属/清除专属"+ `★` 标示 + 徽章 + 全局保存防误覆盖。可覆盖：思考、生图、采样默认（基础设施项仍全局）。
  5. **健壮性**：路由层统一异常兜底，上游 404/403/400 如实透传（原本 2.5-pro 在本项目区域不可用会变笼统 500）。
  6. **测试入库**：`tests/`（103 条，pytest）——能力矩阵/预填充/去重/Cookie 加固/统计/错误提取/按模型覆盖，含 monkeypatch 的流式与非流式端到端。`.gitignore` 增补 `web_state.json`。

## 四·补、第二轮反馈修复（真机复测后，重要）
用户用真实酒馆预设（`Izumi 0629.json`，SillyTavern）复测反馈 5 项，均已处理：
1. **3.6-flash Studio 仍只返回思考、无正文 + 思考压制不生效（原 issue1+5，同一根因）**：真机分析预设发现两点——`assistant_prefill` 为空（**思维链写在 system 提示里，不用预填充**），且 SillyTavern **每次请求恒发 `reasoning_effort: xhigh`**。因此：(a) 基于预填充触发的"压制思考"根本不触发；(b) 前端 effort 覆盖了控制台档位（日志中 `thinkingLevel:HIGH, includeThoughts:true` 即来源于此）。**修复**：新增 `thinking_force_console`（忽略前端 reasoning_effort/thinking_budget，强制用控制台/专属设置）与 `hide_thoughts`（强制 includeThoughts=false）两个控制台开关；`_effort` 增加归一化（xhigh/max→high、min→minimal、auto→默认）；预填充压制路径也改为忽略前端 effort。真机验证：force_console+minimal 后原生思考从 2339 字降到 817，正文正常返回；再开 hide_thoughts 则 reasoning=0。**这直接对应用户"卡原生思维链"的真实诉求**（预设 CoT 在 system 提示、且前端恒发 effort 的场景）。
2. **2.5-pro（原 issue2）**：用户指出默认 `global` 区域可用。确认：Cookie 通道模型路径用 `locations/global`，2.5-pro 真机可用（返回正常）；上一轮 404 是 Express SDK 走了区域端点 `asia-southeast1`。非 bug，保留上一轮的"上游错误如实透传"改进。
3. **全局保存按钮易误触（原 issue3）**：保存前加 `confirm()` 弹窗并说明影响范围；按钮改名"保存全局设置"+ 旁注"不影响已设专属参数的模型"；后端本就隔离（per-model override 优先于全局，已加测试验证全局 temp 不覆盖专属 temp）。
4. **3.1-pro 频繁 429 长等待致前端中断（原 issue4）**：流式（Cookie 真流式 + Express 真流式）**连接建立即发 SSE 心跳**，且**退避等待期间持续发心跳**（`_sleep_with_heartbeat`，注释行 `: keep-alive`，客户端忽略、不注入内容），避免前端超时断开。
5. 见 1（issue5 与 issue1 同根因）。

## 四·补2、第三轮反馈修复（真机 batchGraphql 探针定位，最重要）
用户复测：Studio(Cookie) 的 **3.6-flash 仍只返回原生思考、无正文**（日志：14 个对象全是英文原生思考，末对象 parts 空 + finishReason UNSPECIFIED = 思考阶段被截断）；同内容标准模式正常。用户猜"Studio 不支持 includeThoughts"。**真机探针（e2e/probe_think.py，直连 batchGraphql）确证**：
- **`thinkingLevel` 有效**：MINIMAL → 原生思考 0 字、正文正常 STOP；HIGH → 1000+ 字思考。
- **`includeThoughts=false` 被 batchGraphql 完全忽略**：HIGH+includeFalse 仍回传 1273 字思考。**这就是 3.6-flash 无正文的根因**——SillyTavern 恒发 `reasoning_effort=xhigh` → 档位 HIGH，用户只开了"隐藏思考"（被忽略），于是 3.6-flash 在 450KB 重预设下用 HIGH 狂想、思考阶段就被 batchGraphql 截断，正文没机会产出。标准模式经 SDK 正确应用 includeThoughts=false，故正常。
- SDK 序列化核对（types.py:5542 ThinkingConfig，别名 to_camel）：Express 走的线格式与 batchGraphql 手写的**完全一致**（`{"thinkingLevel":"MINIMAL","includeThoughts":false}`），差异纯在后端行为。3.5+ 用 thinking_level 而非 budget（types.py:13473 明确）。

**修复**：
1. 引入统一的 **`native_thinking_mode`**（`request`/`off`/`console`）取代上一版的 `thinking_force_console` + `hide_thoughts` 两个布尔（旧键在 `resolve_thinking` 里向后兼容映射：hide_thoughts→off、force_console→console；控制台 loadParams 也做了下拉回填）。`native_thinking_mode` 已加入 `PER_MODEL_KEYS`（可按模型专属）。
2. **`off`（关闭原生思考）** = 压到该模型最低档 + 忽略前端 effort + include_thoughts=False；**Cookie 通道额外在响应侧剥离思考块**（`strip_thoughts`，在 `chat_completions` 用 `resolve_thinking(...).include_thoughts` 计算，流式跳过 thought 事件、非流式不附 reasoning_content）——因为 batchGraphql 忽略 includeThoughts。压 minimal 同时避免了重预设截断。
3. 真机验证：3.6-flash Studio + xhigh，`request` 模式原生思考 1042/正文 909；切 `off` 后原生思考 **0**、正文 459 干净输出；按模型专属 off 只影响 3.6-flash，2.5-flash 不受影响。

## 四·补3、第四轮：3.6-flash Studio 无正文的“最终定案”（真机复现真实预设）
用户复测仍无正文并提出：可能是 `safetySettings` 里的 `HARM_CATEGORY_JAILBREAK`（Studio 不支持）；并要求研究 `FINISH_REASON_UNSPECIFIED`。用**真实预设（Izumi 0629.json）**按 SillyTavern 结构重建请求、真机复现，最终定案：
- **`FINISH_REASON_UNSPECIFIED` + 空 parts = 生成异常终止**（研究：多为安全拦截或生成被切）。本例真机复现为**原生思考在 HIGH 档跑飞/被截断**，不是安全拦截。
- **排除 jailbreak/safety**：对 3.6-flash 用 5 种安全组合（含/不含 JAILBREAK、OFF/BLOCK_NONE、无 safety）均正常出正文；决定性对照——同一“无正文/跑飞”场景下把 JAILBREAK 留着、仅切 `native_thinking_mode=off` 即恢复 6913 字正文。故 **JAILBREAK 不是元凶**，保留（对 NSFW 反而有益，OFF=更少拦截）。
- **真正根因**：SillyTavern 恒发 `reasoning_effort=xhigh`。当请求**不以预填充结尾**（预设的“卡思维链K”assistant 预填充未处于末尾，或 ST 把用户输入放最后）时，`prefill` 压制不触发，3.6-flash 原生思考在 HIGH 档于重预设下**跑飞→乱码或思考阶段被截断→无正文**（用户日志 14 个思考对象即此）。“即使 minimal 也不行”是因为**控制台档位被前端 xhigh 覆盖**——单设档位无效，必须忽略前端。
- **真机对照（某真实酒馆预设，以 user 结尾 + xhigh）**：`native_thinking_mode=request` → 原生思考约 2969 字、正文跑飞成乱码；`native_thinking_mode=off` → 原生思考 0、正文 5900~6900 字、开头即预设自带的思维链标签。**修复确认。**
- **结论/给用户**：Cookie(Studio) 下角色扮演，把“原生思考控制”设为 **关闭原生思考**（建议用“保存为该模型专属”对 3.6-flash 生效）。“无正文”诊断信息已增强，会直接提示这一解法。

## 4.5、本轮真机验证结论（真实凭证跑通，重要）
- **项目区域 = `asia-southeast1`**；实测该项目**可用** `gemini-2.5-flash`、`gemini-3.6-flash`（含生图 `gemini-3.1-flash-image`）；`gemini-2.5-pro` 返回 **404 不在该区域/无权限**（非代码 bug，是项目授权/区域问题）。换项目或区域可能不同。
- **batchGraphql 流式噪声**：每个流式块都带 `promptFeedback.blockReason = BLOCKED_REASON_UNSPECIFIED` 与 `candidate.finishReason = FINISH_REASON_UNSPECIFIED`（proto 枚举默认值，**非**真实拦截/结束），必须过滤 `*_UNSPECIFIED`，否则误判。
- **JAILBREAK 分类**：batchGraphql **接受** `HARM_CATEGORY_JAILBREAK`（`OFF` 与 `BLOCK_NONE` 均 200 且有正文），与 Express 通道一致。这是问题2的关键修复点。
- **以 model 轮次结尾**：3.6-flash 经 batchGraphql 真机返回 `Requests ending with a model turn are not supported.`（HTTP 200 内嵌 error），证实预填充兼容对 3.x 的必要性；2.5 可原生透传。
- **Cookie 生图字段名已证实**（消除下方第七节不确定性）：`generationConfig.responseModalities=["TEXT","IMAGE"]` + `generationConfig.imageConfig.{imageSize,aspectRatio}` 真机可用，512 分辨率图片正确返回并按"整块 base64 单发"输出。
- **Cookie 用量**：真机确认 batchGraphql **不回传** `usageMetadata`（大盘 token 恒 0），故仅标准通道统计 token。
- 诚实边界：用良性角色扮演提示词无法直接复现"纯空正文"（所有变体都有正文）；但"缺 JAILBREAK 分类"这一通道不对称是确证的，"思考流出而正文被拦"正是安全拦截特征。**建议用你的真实酒馆预设再复测确认**。

## 四·补4、第五轮：按 REFACTOR_PLAN.md 全量整改（本轮）

对着官方文档做了一次全量审查，产出 `REFACTOR_PLAN.md` 并按其 PR-1～PR-4 实施完毕。**下面两条推翻了此前的结论，别再重走：**

1. **`HARM_CATEGORY_JAILBREAK` 与"无正文"无因果关系（已定案）。** 官方文档明确：越狱分类器**默认关闭**，要打开必须显式把该分类阈值设为具体拦截值。因此"Cookie 通道缺该分类 → 正文被拦"这一解释在机制上不成立，下发 `OFF` 也只是空操作。README 里的错误因果已删除。第四轮"真正根因是 HIGH 档思考跑飞"的定论仍然有效。
2. **Interactions API 的说法需修正。** Gemini Developer API 侧它已于 2026-06 GA 且被推荐用于新项目；Agent Platform 侧仍是 experimental。本项目暂不迁移的**真正理由**是：Express 模式的 REST 面只有 `countTokens` / `generateContent` / `streamGenerateContent`。

本轮修复清单（详见 `REFACTOR_PLAN.md` 的 ID 对应）：

- **P0-1** 档位就近向下夹取（`_clamp_level`）——修掉 Pro 上"选 minimal 得到 high"的反向行为。
- **P0-2** `retry_max` 语义统一为「重试次数」，`retry_max=0` 不再导致 Express 通道一次请求都不发。
- **P0-3** 流式并行函数调用不再被 `break` 丢弃，`index` 由 `ToolCallIndexer` 跨 chunk 稳定分配。
- **P0-4** 思考签名改为短 id + `app/signature_store.py` 旁路缓存，旧 `__thought__` 格式兼容解析，取不回时降级为官方哨兵；并修掉同角色合并会把文本 part 插进连续 FC 之间的问题。
- **P1-1～P1-9 / P2-1～P2-4 / S-1～S-3** 见 README 的「行为变更说明」表。

**仍未做（留给下一轮）：**
- `print` → `logging` 的完整迁移（本轮只做了第一步：修好线程安全与无界队列，并把 token 统计从正则反解改为直接记账）。
- `FunctionResponse` 字段名复核（**D-4，唯一需要真机确认的遗留项**）：官方迁移清单要求带 `call_id` + `name`，代码传的是 `id=`。请跑
  `python -c "from google.genai import types; print(types.FunctionResponse.model_fields.keys())"`
  确认是别名还是新字段，然后在代码里加注释锁死结论。
- 生图 `system_instruction`（P1-9）与生图 `google_search` 两通道不一致：已加开关，但默认值保持旧行为，等真机验证后再定。
- 真机多轮 function calling 压测（本轮只补了单测，无凭证无法端到端验证）。

## 五、关键结论 / 已纠正（重要，别重走弯路）
- **采样参数**：据官方 `latest-model` 文档——**自 Gemini 3.6 Flash / 3.5 Flash-Lite 起及所有更新/未来模型，`temperature`/`top_p`/`top_k` 已废弃**（现忽略、未来 400，需移除）；更早 3.x（3.0–3.5 非 lite）仍可用但建议保持默认。**所有 3.x 不支持 `candidate_count`**。（`_temp_deprecated()` 实现）
- **思考**：3.x 用 `thinking_level`（不可完全关；默认 3.6-flash=medium、pro=high、flash-lite=minimal）；2.5 用 `thinking_budget`（-1 动态；flash 可 0 关闭，pro 最低 128）。**优先级默认 请求 effort > 控制台/专属 > 模型默认**。用 `native_thinking_mode` 统一控制：`off`=忽略前端+压最低+不回传思考（Cookie 通道响应侧剥离），`console`=忽略前端+用控制台档位，`request`=跟随前端。effort 归一化：xhigh/max→high、min/minimal→minimal、auto/空→默认。
- **⚠️ batchGraphql 忽略 `includeThoughts`（真机确证）**：Studio/Cookie 通道设 includeThoughts=false 无效，仍回传思考；唯一有效的减思考手段是 `thinkingLevel=MINIMAL`。故"关闭原生思考"在 Cookie 通道靠"压 minimal + 响应侧剥离思考块"实现。标准（Express/SDK）通道原生支持 includeThoughts=false。
- **预填充**：3.6+ 拒绝以 model 轮次结尾（400）。`smart`=把末尾 assistant 预填充转成末尾 user 的“续写指令”，模型返回后把预填充**拼回输出开头**（真还原）；`minimal`=仅追加占位 user 防报错；`off`=不处理。**与模型名无关，新模型自动生效。**
- **`HARM_CATEGORY_JAILBREAK`**：在 Vertex/Agent Platform 合法（SDK 枚举含），Express 模式不报错；仅 Gemini Developer API 会 400。保留即可。
  ⚠️ **但它与"有思考无正文"没有因果关系**——官方文档写明该分类器默认关闭，不下发不会启用过滤。详见第四·补4 节。
- **生图比例白名单**：pro-image 10 种；flash-image 14 种（多 `1:4/4:1/1:8/8:1`）。`9:21` 无官方出处，已移除。分辨率：pro `1K/2K/4K`、flash `512/1K/2K/4K`。选到不支持的比例会自动回退“由模型决定”，不报错。
- **Cookie 有效期**：通常数周甚至更久，仅在 `Permission Denied` 时才需重取（不是 1~2 小时）。
- **Interactions API 迁移**：Agent Platform（Vertex）当前**不对基础 Gemini 模型开放 Interactions**（仅少数 agent），且 Express（API-key）与 Interactions 端点不兼容 → 暂无法迁移。等开放后新增 `InteractionsUpstream`（实现 `BaseUpstream`）接入路由即可，能力/预填充/参数逻辑可复用。

## 六、未做 / 待议（可继续的方向）
- **~~按模型单独保存参数~~**：✅ 本轮已完成（见第四节 3.4）。
- **~~测试入库~~**：✅ 本轮已完成（`tests/` 103 条）。
- **生图 `google_search`**：Express 生图默认开、Cookie 生图不开（不一致）；用户暂不动。
- **Cookie 通道函数调用**：未支持（不下发 `functionDeclarations`）。
- **建议（未做）**：登录防爆破 + HTTPS 提醒、`include_thoughts` 独立开关（当前随”预填充压制思考”联动）、服务端并发限制（信号量）、控制台”刷新模型”按钮、统计持久化。
- **模型名后缀**（`-4k`/`-think-x`）：未接入（仅 `-search` 生效）；用户决定不加。

## 七、已知不确定
- **~~Cookie 生图字段名~~**：✅ 本轮真机证实可用（见第 4.5 节）。
- **纯空正文复现**：良性提示词未能直接复现"只有思考没正文"，问题2 按"通道缺 JAILBREAK 分类"这一确证不对称修复；建议用真实酒馆预设复测确认。
- **工具调用多轮 / thought signature**：本轮验证了单轮 function calling（Express），多轮 thought signature 编解码未做真机多轮压测。
- **2.5-pro**：本项目区域 404，未能验证其思考预算路径的真机行为（代码逻辑有单测覆盖）。

## 八、测试（已入库）
- **运行方式**：需 Python 3.11。`uv venv --python 3.11 .venv && uv pip install -p .venv/bin/python -r app/requirements.txt pytest`，然后 `.venv/bin/python -m pytest tests/ -q`。
- **覆盖**：`tests/` 共 **103 条**——`test_model_capabilities`（家族/裁剪/思考/生图白名单）、`test_prefill*`（兼容/去重/增强/原生透传/思考压制）、`test_cookie_parsing` + `test_cookie_hardening`（解析/同角色合并/finishReason 映射/UNSPECIFIED 过滤/无正文诊断，含 monkeypatch 流式与非流式端到端）、`test_stats`（Cookie 不计 token、成功数单独计）、`test_api_helpers`、`test_error_handling`（上游错误提取）、`test_model_overrides`（存储/合并/三通道生效/清除）。
- `conftest.py` 自动隔离 `web_state.json`（切临时 CWD）与 `app_state` 内存态，测试间无串扰。
- **真机联调脚本**未入库（放在仓库外 `e2e/`，含凭证），交付前已清除。
