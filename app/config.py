from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    API_KEY: str = "123456"
    VERTEX_EXPRESS_API_KEY: Optional[str] = None
    FAKE_STREAMING: bool = False
    FAKE_STREAMING_INTERVAL: float = 1.0
    MODELS_CONFIG_URL: str = ""
    ROUNDROBIN: bool = False
    SAFETY_SCORE: bool = False
    PROXY_URL: Optional[str] = None
    SSL_CERT_FILE: Optional[str] = None
    # 标准（Express）模式的上游 base_url 覆盖。留空 = 用 SDK 默认（全局端点）。
    # 仅在确有需要时使用；要钉住 location 请用控制台的
    # express_location（见 DEFAULT_SETTINGS）+「通道与凭证」里的 Project ID，不要动这个。
    VERTEX_BASE_URL: Optional[str] = None

    # Cookie direct mode settings (Recommended for cloud deployments like Render)
    GOOGLE_COOKIE: Optional[str] = None         # Google Cookie string
    GOOGLE_PROJECT_ID: Optional[str] = None     # Google Cloud Project ID
    EXPERIMENT_FLAGS: Optional[str] = None      # experimentFlagsBinary (optional; paste from a console request if needed)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


_settings = AppSettings()

API_KEY = _settings.API_KEY

raw_vertex_keys = _settings.VERTEX_EXPRESS_API_KEY
if raw_vertex_keys:
    VERTEX_EXPRESS_API_KEY_VAL = [key.strip() for key in raw_vertex_keys.split(",") if key.strip()]
else:
    VERTEX_EXPRESS_API_KEY_VAL = []

FAKE_STREAMING_ENABLED = _settings.FAKE_STREAMING
FAKE_STREAMING_INTERVAL_SECONDS = _settings.FAKE_STREAMING_INTERVAL
MODELS_CONFIG_URL = _settings.MODELS_CONFIG_URL
ROUNDROBIN = _settings.ROUNDROBIN
SAFETY_SCORE = _settings.SAFETY_SCORE
PROXY_URL = _settings.PROXY_URL
SSL_CERT_FILE = _settings.SSL_CERT_FILE
VERTEX_BASE_URL = _settings.VERTEX_BASE_URL

GOOGLE_COOKIE = _settings.GOOGLE_COOKIE
GOOGLE_PROJECT_ID = _settings.GOOGLE_PROJECT_ID
EXPERIMENT_FLAGS = _settings.EXPERIMENT_FLAGS

REASONING_TAG = "agent_platform_think_tag"
# 向后兼容别名（历史代码引用 VERTEX_REASONING_TAG）
VERTEX_REASONING_TAG = REASONING_TAG


# ============================================================
# 控制台可调的运行时默认值（可在大盘热更新，持久化到 web_state.json）
# 优先级：单次请求 > 控制台设置(这些值) > 代码内置兜底
# 环境变量仅作为“初始值”。
# ============================================================
DEFAULT_SETTINGS = {
    # 思考
    "thinking_g3_level": "",              # 空=按模型各自默认(3.6-flash=medium/pro=high/flash-lite=minimal)；也可强制 minimal|low|medium|high
    "thinking_g25_budget": -1,            # Gemini 2.5 默认思考预算: -1=动态, 0=关(仅flash), 或整数
    # 生图
    "image_size": "4K",                   # 默认分辨率: 512|1K|2K|4K（按模型白名单校验）
    "image_aspect_ratio": "",             # 默认宽高比, ""=自动
    # 采样默认（客户端未显式传时使用；None=不注入）
    "default_temperature": None,
    "default_top_p": None,
    "default_max_tokens": None,
    # 输入图片压缩
    "img_compress_enabled": True,
    "img_compress_max_dim": 1536,
    "img_compress_max_mb": 1.5,
    "img_compress_quality": 85,
    # 重试
    # 语义：retry_max = 失败后的**重试**次数，总请求次数 = retry_max + 1。
    # 两条通道统一走 api_helpers.get_retry_settings() 读取（会钳位到 0–50）。
    "retry_max": 10,
    "retry_backoff_seconds": 5,
    # 开关（初始值取环境变量）
    "fake_streaming": FAKE_STREAMING_ENABLED,
    "fake_streaming_interval": FAKE_STREAMING_INTERVAL_SECONDS,
    "roundrobin": ROUNDROBIN,
    "safety_score": SAFETY_SCORE,
    # 预填充兼容模式: smart|minimal|off
    # 默认 smart。两种模式的优劣**取决于预填充的结尾形态**，用某真实酒馆预设
    # （预填充为一段完整句子 + 一个未闭合的思维链开标签）实测 gemini-3.6-flash × 3：
    #   smart      重复开标签 0/3，思考语言正确 3/3   ← 真实预设的常见形态
    #   keep_turn  重复开标签 3/3，思考语言正确 2/3
    # 真实预设的预填充多以完整句子收尾（"…¡Allá voy!"），keep_turn 追加的 user
    # 推动语会让模型当成新一轮，把开标签又写一遍；而该重复**去重逻辑抓不到**
    # （预填充结尾与输出开头无重叠），最终输出里出现两个开标签，破坏前端正则。
    # 仅当预填充停在半截 token（如 "<thinking>\n1."）时 keep_turn 才更优——
    # 那种情况下 smart 会跑题且丢格式（合成用例实测 0/3）。
    "prefill_mode": "smart",
    # 预填充触发时压制原生思考（“卡思维链”核心开关）：
    # 3.x 压到最低档（minimal/低于则 low）并关闭思考回传；2.5-flash 预算设 0 全关、2.5-pro 降到最低 128。
    # 此路径会忽略前端 effort（预填充时优先）。
    "prefill_suppress_thinking": True,
    # 原生思考控制（酒馆预设“卡原生思维链”核心）：
    #   request = 跟随前端 reasoning_effort（默认）
    #   off     = 关闭原生思考：压到该模型最低档 + 忽略前端 effort + 不回传思考
    #             （Studio/batchGraphql 忽略 includeThoughts，故 Cookie 通道会在响应侧剥离思考块）
    #   console = 忽略前端 effort，强制用控制台/该模型专属档位
    "native_thinking_mode": "request",
    # —— 以下两个为上一版布尔开关，保留仅作向后兼容（新 UI 用 native_thinking_mode）——
    "thinking_force_console": False,
    "hide_thoughts": False,
    # smart 模式续写指令模板（留空=用内置默认；预填充文本会自动附在模板之后）
    "prefill_instruction": "",
    # 出站参数调试：打印两条通道实际发出的 generationConfig / thinkingConfig。
    # 实机验证思考档位、采样裁剪是否生效时必开。
    # （旧键名 cookie_debug 保留为别名，只作用于 Cookie 通道的额外诊断）
    "debug_outbound": False,
    "cookie_debug": False,
    # 思考签名内嵌开关（默认关）：
    #   关 = 生成短 tool_call_id，签名存进进程内旁路缓存（推荐，避免被前端截断）
    #   开 = 退回旧的 `{id}__thought__{base64}` 内嵌格式，供多进程/多副本部署使用
    # 生图请求是否下发 system_instruction（默认关，保持既有行为）。
    # 官方未禁止生图模型使用系统指令，但旧代码一直剥离；打开前请先真机验证目标模型。
    "image_system_instruction": False,
    # 轻量前端（RikkaHub 等）注入：留空 = 不启用，酒馆用户不受影响。
    # 这两项解决的是"前端本身没有预设系统"的场景，见 message_processing.apply_console_injection。
    "inject_system_instruction": "",
    "inject_prefill": "",
    # 生图模型是否也注入预填充。实测预填充对生图有很强的引导力
    # （同一句"画一只猫"：无预填充→彩色写实照片；预填充承诺"纯黑白钢笔线稿"→真的输出线稿），
    # 但角色扮演用的预填充落到生图请求上会让模型改吐文本，故默认关、按需开。
    "inject_prefill_for_image": False,
    # 采样参数处理：auto=按版本自动判定 / deprecated=强制剥离 / allowed=强制保留。
    # 给"新出的模型版本号更小但已废弃采样"这类情况留的手动出口，免于改代码。
    "sampling_policy": "auto",
    # 思维链守卫（默认开）：预填充停在未闭合的思维链开标签时（标签名由各人预设决定，
    # 代理只按"未闭合"这一形状识别，不预设任何具体标签名），
    # 在续写指令末尾追加一条硬性要求——先写完思维链再闭合标签、然后才写正文。
    # 背景：预填充只把话头停在开标签上，没有任何一句话要求模型"必须先完成思考"，
    # 实测模型经常跳过思考直接写正文，前端正则于是抓不到思维链（多数情况没有思维链）。
    "prefill_cot_guard": True,
    # ===== 标准（Express）模式的 location 钉定（实测有效，见 README“标准模式的 location”）=====
    # 背景：只发裸模型名时，请求走 express 端点格式
    #   https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent
    # location 由 Google 后端自行路由，可能落到该模型**不提供服务**的区域并 404
    #   （实测 gemini-2.5-pro 被路由到 asia-southeast1 → 404 not found）。
    # 改发带项目与区域的完整资源路径后同一模型 200 正常：
    #   projects/{project}/locations/{location}/publishers/google/models/{model}
    # express_location 留空 = 保持旧行为（裸模型名）；填 global（推荐）或某区域即启用钉定。
    # 项目 ID 直接取「通道与凭证」里填的那个（或环境变量 GOOGLE_PROJECT_ID）——
    # 一个人通常只有一个 Express 项目，没必要再单独配一份。
    # ⚠️ 项目必须是该 API Key 有权且已开启计费的项目，否则 403（实测换成别的项目会
    #    "requires billing to be enabled"）。
    # 默认 global：多数 Gemini 模型只在 global 提供（如 gemini-2.5-pro），
    # 让后端自选区域会偶发 404。留空 = 回到"后端自选"的旧行为。
    # 钉定失败（项目不匹配/该区域无此模型）会自动退回裸模型名重试一次，见
    # api_helpers.is_location_pin_failure —— 所以这个默认值不会把任何人变糟。
    "express_location": "global",
    # 弱网可靠传输（见 resumable.py）：服务端响应缓冲 + 序列号事件 + 断线重放/分片轮询。
    # 上游读取由后台泵任务驱动，与前端连接解绑；Google 的回复总能完整落缓冲。
    "resumable_enabled": True,            # 总开关；关掉则回到原有直通行为
    "resumable_ttl_seconds": 1800,        # 已完成的缓冲保留时长（秒），超时清掉
    "resumable_max_streams": 200,         # 缓冲总数上限，超限淘汰最旧的已完成缓冲
    # 按模型单独保存的参数覆盖：{ "模型ID": { 键: 值, ... } }
    # 仅覆盖“与模型相关”的参数（见 PER_MODEL_KEYS）；优先级 请求 > 模型专属 > 全局 > 内置。
    "model_overrides": {},
}

# 允许按模型单独保存（覆盖全局默认）的参数键。
# 其余为基础设施级（图压缩/重试/假流式/预填充/安全分/调试等），保持全局唯一。
PER_MODEL_KEYS = [
    "native_thinking_mode",
    "thinking_g3_level",
    "thinking_g25_budget",
    "image_size",
    "image_aspect_ratio",
    "default_temperature",
    "default_top_p",
    "default_max_tokens",
    # 注入项按模型区分很常见：只给跑角色扮演的模型开，问答模型保持干净。
    "inject_system_instruction",
    "inject_prefill",
    # 续写指令也必须能按模型分开：文本模型要"接着往下写"，生图模型要"直接输出图片"，
    # 只有一份全局模板时，为文本调好的那句会让生图吐字符画。
    "prefill_instruction",
    # 这两个本就是"针对生图模型"的开关，作用域却是全局，用户按模型保存会落空。
    "image_system_instruction",
    "inject_prefill_for_image",
    "sampling_policy",
    # 思维链守卫按模型分开很自然：只给跑角色扮演预设的模型开。
    "prefill_cot_guard",
]
