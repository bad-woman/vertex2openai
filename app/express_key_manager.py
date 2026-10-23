import random
from typing import List
import config as app_config
from runtime_state import app_state


class ExpressKeyManager:
    """管理 Agent Platform Express Mode API Key，支持轮询或随机生成单次请求的完整重试序列。

    多账号支持：每个 Key 是一条记录（Key + 它自己的 GCP Project ID），
    列表持久化在 web_state.json，控制台可随时增删改（热生效，无需重启）。
    """

    def __init__(self):
        self.round_robin_index: int = 0

    def get_records(self) -> List[dict]:
        """当前启用的 Key 记录（保留控制台列表中的原始序号 index，以便告警日志精准对应 #1/#2）。"""
        all_records = app_state.get_express_keys()
        return [
            dict(r, index=idx)
            for idx, r in enumerate(all_records)
            if r.get("enabled", True)
        ]

    @property
    def express_keys(self) -> List[str]:
        return [r["key"] for r in self.get_records()]

    def get_total_keys(self) -> int:
        return len(self.get_records())

    def get_ordered_express_records(self) -> List[dict]:
        """为单次请求生成完整的 Key 轮换序列（重试时按此序列依次切换不同 Key）。"""
        records = self.get_records()
        if not records:
            return []
        if app_state.get_setting("roundrobin", app_config.ROUNDROBIN):
            start = self.round_robin_index % len(records)
            self.round_robin_index = (start + 1) % len(records)
            return records[start:] + records[:start]
        shuffled = list(records)
        random.shuffle(shuffled)
        return shuffled

    def refresh_keys(self):
        self.round_robin_index = 0
        print(f"🔄 [密钥刷新] 当前共 {self.get_total_keys()} 个可用 Express API Key。")
