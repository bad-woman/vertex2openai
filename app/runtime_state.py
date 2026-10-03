import copy
import json
import os
import tempfile
import threading
import time

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

    # ---------- Express API Key（多账号，可热编辑） ----------
    # 记录形态：{"key":..., "project_id":..., "project_source":"auto|manual|", 
    #            "detected_at": 时间戳, "note":..., "enabled": True}
    # 一个 Key 对应一个谷歌账号/项目，所以 Project ID 必须**跟着 Key 走**，
    # 不能共用 Cookie 通道那份全局 Project ID（见 express_sdk.resolve_express_model_path）。

    @staticmethod
    def _normalize_express_record(item) -> dict:
        if isinstance(item, str):
            item = {"key": item}
        if not isinstance(item, dict):
            return {}
        key = str(item.get("key", "") or "").strip()
        if not key:
            return {}
        source = str(item.get("project_source", "") or "").strip().lower()
        if source not in ("auto", "manual"):
            source = ""
        return {
            "key": key,
            "project_id": str(item.get("project_id", "") or "").strip(),
            "project_source": source,
            "detected_at": item.get("detected_at") or 0,
            "note": str(item.get("note", "") or "").strip(),
            "enabled": bool(item.get("enabled", True)),
        }

    def get_express_keys(self) -> list:
        with self._lock:
            stored = self._state.get("express_keys")
            if not isinstance(stored, list):
                return []
            out = []
            for item in stored:
                rec = self._normalize_express_record(item)
                if rec:
                    out.append(rec)
            return out

    def set_express_keys(self, records: list) -> list:
        """整表覆盖保存（控制台的增删改都走这里）。

        同时打上 express_keys_initialized 标记：**用户保存空列表后，
        启动时不得再从环境变量把 Key 导回来**（否则删不掉）。
        """
        clean, seen = [], set()
        for item in (records or []):
            rec = self._normalize_express_record(item)
            if not rec or rec["key"] in seen:
                continue
            seen.add(rec["key"])
            clean.append(rec)
        with self._lock:
            self._state["express_keys"] = clean
            self._state["express_keys_initialized"] = True
            self._save()
            print(f"🔑 [状态管理器] 已保存 {len(clean)} 个 Express API Key（明文存于 web_state.json，权限 0600）。")
        return clean

    def update_express_key_project(self, key: str, project_id: str, source: str = "auto") -> bool:
        """写回某个 Key 的 Project ID（自动探测或人工覆盖）。"""
        key = (key or "").strip()
        if not key:
            return False
        with self._lock:
            stored = self._state.get("express_keys")
            if not isinstance(stored, list):
                return False
            changed = False
            new_list = []
            for item in stored:
                rec = self._normalize_express_record(item)
                if not rec:
                    continue
                if rec["key"] == key:
                    rec["project_id"] = (project_id or "").strip()
                    rec["project_source"] = source if source in ("auto", "manual") else ""
                    rec["detected_at"] = time.time()
                    changed = True
                new_list.append(rec)
            if changed:
                self._state["express_keys"] = new_list
                self._save()
            return changed

    def import_env_express_keys_once(self, env_keys: list) -> int:
        """把环境变量 VERTEX_EXPRESS_API_KEY 做**一次性**初始导入。

        只在从未保存过（没有 express_keys_initialized 标记）时执行，
        因此用户在控制台把 Key 全删光并保存后，重启也不会被环境变量重新灌回来。
        """
        with self._lock:
            if self._state.get("express_keys_initialized"):
                return 0
            existing = self._state.get("express_keys")
            if isinstance(existing, list) and existing:
                self._state["express_keys_initialized"] = True
                self._save()
                return 0
            clean, seen = [], set()
            for k in (env_keys or []):
                k = str(k or "").strip()
                if not k or k in seen:
                    continue
                seen.add(k)
                clean.append(self._normalize_express_record({"key": k}))
            self._state["express_keys"] = clean
            self._state["express_keys_initialized"] = True
            self._save()
            if clean:
                print(f"📥 [状态管理器] 已从环境变量 VERTEX_EXPRESS_API_KEY 一次性导入 {len(clean)} 个 Key，"
                      "此后以控制台保存的列表为准。")
            return len(clean)

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
