import random
from typing import List, Optional, Tuple
import config as app_config


class ExpressKeyManager:
    """管理 Vertex AI Express Mode API Key，支持随机或轮询选择。"""

    def __init__(self):
        self.express_keys: List[str] = app_config.VERTEX_EXPRESS_API_KEY_VAL
        self.round_robin_index: int = 0

    def get_total_keys(self) -> int:
        return len(self.express_keys)

    def get_random_express_key(self) -> Optional[Tuple[int, str]]:
        if not self.express_keys:
            print("❌ [密钥配置] 未配置 VERTEX_EXPRESS_API_KEY，无法调用 Gemini Express Mode。")
            return None

        indexed_keys = list(enumerate(self.express_keys))
        random.shuffle(indexed_keys)
        original_idx, key = indexed_keys[0]
        print(f"🔑 [密钥选择] 已随机选择第 {original_idx + 1} 个 Express API Key。")
        return original_idx, key

    def get_roundrobin_express_key(self) -> Optional[Tuple[int, str]]:
        if not self.express_keys:
            print("❌ [密钥配置] 未配置 VERTEX_EXPRESS_API_KEY，无法调用 Gemini Express Mode。")
            return None

        if self.round_robin_index >= len(self.express_keys):
            self.round_robin_index = 0

        key = self.express_keys[self.round_robin_index]
        original_idx = self.round_robin_index
        self.round_robin_index = (self.round_robin_index + 1) % len(self.express_keys)
        print(f"🔑 [密钥选择] 已按轮询策略选择第 {original_idx + 1} 个 Express API Key。")
        return original_idx, key

    def get_express_api_key(self) -> Optional[Tuple[int, str]]:
        if app_config.ROUNDROBIN:
            return self.get_roundrobin_express_key()
        return self.get_random_express_key()

    def get_all_keys_indexed(self) -> List[Tuple[int, str]]:
        return list(enumerate(self.express_keys))

    def refresh_keys(self):
        self.express_keys = app_config.VERTEX_EXPRESS_API_KEY_VAL
        self.round_robin_index = 0
        print(f"🔄 [密钥刷新] 已重新加载 {len(self.express_keys)} 个 Express API Key。")
