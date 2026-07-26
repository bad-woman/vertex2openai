import copy
import json
import os
import tempfile
import threading

import config as app_config

# S-3：允许把状态落到挂载卷，避免 docker compose 重建后设置与 Cookie 全部丢失。
STATE_DIR = os.environ.get("STATE_DIR", ".")
STATE_FILE = os.path.join(STATE_DIR, "web_state.json")


class AppState:
    """运行态管理器（内存优先 + 写时落盘）。

    P1-4 的改动要点：
      - 旧实现每个 getter 都调 `_load_state()` 同步读盘。全项目有 20+ 处
        get_settings/get_setting/get_effective_settings 调用，且全在 async 请求路径上
        （每压缩一张图片还要再读一次），高并发下会把事件循环串行化。
        现在只在启动和显式 reload() 时读盘。
      - 落盘改为「临时文件 + os.replace」，避免进程崩溃时把 web_state.json 写坏。
      - getter 返回深拷贝，防止调用方无意间改到 model_overrides 这层嵌套字典。
      - 文件权限 0600：里面存着完整的 Google 会话 Cookie。
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._state = {"use_web_proxy": False}
        self._load_from_disk()

    # ---------- 持久化 ----------

    def _load_from_disk(self) -> None:
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._state.update(data)
        except Exception as e:
            print(f"⚠️ [状态管理器] 无法读取持久化配置文件，已自动降级为内存模式: {e}")

    def _save(self) -> None:
        """原子写：先写同目录临时文件再 os.replace（同一文件系统内是原子操作）。"""
        try:
            target_dir = os.path.dirname(STATE_FILE) or "."
            os.makedirs(target_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".web_state-", suffix=".tmp", dir=target_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._state, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, STATE_FILE)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            try:
                os.chmod(STATE_FILE, 0o600)   # 里面有完整 Google Cookie
            except OSError:
                pass
        except Exception as e:
            print(f"⚠️ [状态管理器] 无法保存状态到磁盘: {e}")

    def reload(self) -> None:
        """显式从磁盘重载（外部改了文件时用）。"""
        with self._lock:
            self._load_from_disk()

    # ---------- 通道开关与凭证 ----------

    def enable_web_proxy(self, enabled: bool):
        with self._lock:
            self._state["use_web_proxy"] = bool(enabled)
            self._save()
            print(f"🔄 [状态管理器] 网页反代状态已更新：{enabled}")

    def is_web_proxy_enabled(self) -> bool:
        with self._lock:
            return bool(self._state.get("use_web_proxy", False))

    def set_google_cookie(self, cookie_str: str):
        with self._lock:
            self._state["google_cookie"] = cookie_str
            self._save()
            print("🔄 [状态管理器] 谷歌独立 Cookie 已保存到运行状态")

    def get_google_cookie(self) -> str:
        with self._lock:
            return self._state.get("google_cookie", "")

    def set_project_id(self, project_id: str):
        with self._lock:
            self._state["google_project_id"] = project_id
            self._save()
            print(f"🔄 [状态管理器] 项目 ID 已保存: {project_id}")

    def get_project_id(self) -> str:
        with self._lock:
            return self._state.get("google_project_id", "")

    # ---------- 控制台可调设置 ----------

    def get_settings(self) -> dict:
        """完整设置（内置默认 + 持久化覆盖），保证所有键都存在；返回深拷贝。"""
        with self._lock:
            merged = copy.deepcopy(app_config.DEFAULT_SETTINGS)
            stored = self._state.get("settings")
            if isinstance(stored, dict):
                for k, v in stored.items():
                    if k in merged:
                        merged[k] = copy.deepcopy(v)
            return merged

    def get_setting(self, key: str, default=None):
        with self._lock:
            stored = self._state.get("settings")
            if isinstance(stored, dict) and key in stored:
                return copy.deepcopy(stored[key])
            if key in app_config.DEFAULT_SETTINGS:
                return copy.deepcopy(app_config.DEFAULT_SETTINGS[key])
            return default

    def update_settings(self, patch: dict) -> dict:
        """合并更新设置，只接受已知键，返回更新后的完整设置。"""
        if not isinstance(patch, dict):
            return self.get_settings()
        with self._lock:
            current = self._state.get("settings")
            current = dict(current) if isinstance(current, dict) else {}
            accepted = 0
            for k, v in patch.items():
                if k in app_config.DEFAULT_SETTINGS and k != "model_overrides":
                    current[k] = v
                    accepted += 1
            self._state["settings"] = current
            self._save()
            print(f"🔧 [状态管理器] 已更新 {accepted} 项运行时设置。")
        return self.get_settings()

    # ---------- 按模型参数覆盖 ----------

    def get_model_overrides(self) -> dict:
        with self._lock:
            stored = self._state.get("settings")
            if isinstance(stored, dict) and isinstance(stored.get("model_overrides"), dict):
                return copy.deepcopy(stored["model_overrides"])
            return {}

    def set_model_override(self, model_name: str, patch: dict) -> dict:
        model_name = (model_name or "").strip()
        if not model_name or not isinstance(patch, dict):
            return {}
        clean = {k: v for k, v in patch.items() if k in app_config.PER_MODEL_KEYS}
        with self._lock:
            settings = self._state.get("settings")
            settings = dict(settings) if isinstance(settings, dict) else {}
            overrides = settings.get("model_overrides")
            overrides = dict(overrides) if isinstance(overrides, dict) else {}
            overrides[model_name] = clean
            settings["model_overrides"] = overrides
            self._state["settings"] = settings
            self._save()
            print(f"🔧 [状态管理器] 已保存模型 {model_name} 的专属参数（{len(clean)} 项）。")
            return clean

    def clear_model_override(self, model_name: str) -> bool:
        model_name = (model_name or "").strip()
        with self._lock:
            settings = self._state.get("settings")
            if not isinstance(settings, dict):
                return False
            overrides = settings.get("model_overrides")
            if not isinstance(overrides, dict) or model_name not in overrides:
                return False
            overrides.pop(model_name, None)
            settings["model_overrides"] = overrides
            self._state["settings"] = settings
            self._save()
            print(f"🔧 [状态管理器] 已清除模型 {model_name} 的专属参数。")
            return True

    def get_effective_settings(self, model_name: str) -> dict:
        """该模型生效的设置：全局默认叠加该模型专属覆盖（仅 PER_MODEL_KEYS）。"""
        base = self.get_settings()
        overrides = base.get("model_overrides") or {}
        ov = overrides.get((model_name or "").strip())
        if isinstance(ov, dict):
            for k in app_config.PER_MODEL_KEYS:
                if k in ov:
                    base[k] = ov[k]
        return base


# 单例模式导出
app_state = AppState()
