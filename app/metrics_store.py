"""
SQLite 长久持久化存储（纯 Python 标准库 sqlite3，零额外依赖）。

落盘路径：os.path.join(STATE_DIR, "metrics.db")（自动复用 Docker 挂载卷 /app/data）。
负责持久化：
  1. daily_model_stats — 按天 × 模型 × 通道聚合的请求量、成功率、重试、Token（含隐式前缀缓存命中）、TTFT 与 tok/s；
  2. key_hits          — 每把 Express Key（SHA-256 前 16 位指纹，绝不落盘明文 Key）与 Cookie 通道的累计命中次数；
  3. call_records      — 最近调用记录（滚动保留最新 500 条，重启不丢）。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

STATE_DIR = os.environ.get("STATE_DIR", ".")
DEFAULT_DB_PATH = os.path.join(STATE_DIR, "metrics.db")
MAX_CALL_RECORDS = 500
COOKIE_HIT_FP = "__cookie__"


def key_fingerprint(raw_key: str) -> str:
    """计算 Express API Key 的非敏感 16 位 SHA-256 指纹（用于命中计数持久化，删改调序不串号）。"""
    clean = (raw_key or "").strip()
    if not clean:
        return ""
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]


def _local_day_str(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts if ts is not None else time.time()))


class MetricsStore:
    """线程安全的轻量 SQLite 持久化引擎（WAL 模式，启动自动建表，损坏/只读时优雅降级不崩服务）。"""

    def __init__(self, db_path: Optional[str] = None):
        self._lock = threading.RLock()
        self._db_path = db_path or DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    @property
    def db_path(self) -> str:
        return self._db_path

    def reconfigure(self, db_path: str) -> None:
        """切换数据库路径（供测试或显式重载使用）。"""
        with self._lock:
            self.close()
            self._db_path = db_path
            self._init_db()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def _init_db(self) -> None:
        try:
            parent = os.path.dirname(self._db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(self._db_path, timeout=5.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS daily_model_stats (
                    day TEXT NOT NULL,
                    model TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    req_total INTEGER NOT NULL DEFAULT 0,
                    req_ok INTEGER NOT NULL DEFAULT 0,
                    req_err INTEGER NOT NULL DEFAULT 0,
                    retries INTEGER NOT NULL DEFAULT 0,
                    prompt_tk INTEGER NOT NULL DEFAULT 0,
                    cached_tk INTEGER NOT NULL DEFAULT 0,
                    thought_tk INTEGER NOT NULL DEFAULT 0,
                    output_tk INTEGER NOT NULL DEFAULT 0,
                    ttft_sum REAL NOT NULL DEFAULT 0.0,
                    ttft_cnt INTEGER NOT NULL DEFAULT 0,
                    tps_sum REAL NOT NULL DEFAULT 0.0,
                    tps_cnt INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (day, model, channel)
                );

                CREATE TABLE IF NOT EXISTS key_hits (
                    key_fp TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    last_hit_at REAL NOT NULL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS call_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_call_records_created ON call_records(created_at DESC);
                """
            )
            conn.commit()
            try:
                os.chmod(self._db_path, 0o600)
            except OSError:
                pass
            self._conn = conn
        except Exception as e:
            self._conn = None
            print(f"⚠️ [统计持久化] SQLite 初始化失败（降级为纯内存统计）：{e}")

    # ---------- Key / Cookie 命中次数 ----------

    def record_key_hit(self, channel: str, raw_key: str = "") -> None:
        ch = "cookie" if channel == "cookie" else "express"
        fp = COOKIE_HIT_FP if ch == "cookie" else key_fingerprint(raw_key)
        if not fp:
            return
        now = time.time()
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute(
                    """
                    INSERT INTO key_hits (key_fp, channel, hit_count, last_hit_at)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(key_fp) DO UPDATE SET
                        hit_count = hit_count + 1,
                        last_hit_at = excluded.last_hit_at
                    """,
                    (fp, ch, now),
                )
                self._conn.commit()
            except Exception as e:
                print(f"⚠️ [统计持久化] 写入凭证命中次数失败：{e}")

    def get_key_hits_map(self) -> Dict[str, int]:
        with self._lock:
            if self._conn is None:
                return {}
            try:
                rows = self._conn.execute("SELECT key_fp, hit_count FROM key_hits").fetchall()
                return {str(r["key_fp"]): int(r["hit_count"] or 0) for r in rows}
            except Exception:
                return {}

    def get_cookie_hits(self) -> int:
        return int(self.get_key_hits_map().get(COOKIE_HIT_FP, 0))

    # ---------- 单次调用落盘（明细 + 按日按模型聚合） ----------

    def record_call(
        self,
        entry: Dict[str, Any],
        *,
        channel: str = "express",
        created_at: Optional[float] = None,
    ) -> Optional[int]:
        """写入一条调用记录并同步更新 daily_model_stats，返回持久化记录 ID。"""
        ts = float(created_at if created_at is not None else time.time())
        day = _local_day_str(ts)
        ch = "cookie" if channel == "cookie" else "express"
        model = str(entry.get("model") or "unknown").strip() or "unknown"
        group = str(entry.get("group") or "ok")
        retries = max(0, int(entry.get("retries") or 0))

        p_tk = max(0, int(entry.get("p_tk") or 0))
        cached_tk = max(0, min(p_tk, int(entry.get("cached_tk") or 0))) if p_tk > 0 else max(0, int(entry.get("cached_tk") or 0))
        th_tk = max(0, int(entry.get("th_tk") or 0))
        out_tk = max(0, int(entry.get("out_tk") or 0))

        ttft_val = entry.get("ttft")
        ttft_add = float(ttft_val) if (group == "ok" and isinstance(ttft_val, (int, float)) and ttft_val > 0) else 0.0
        ttft_cnt = 1 if ttft_add > 0 else 0

        tps_val = entry.get("tps")
        tps_add = float(tps_val) if (group == "ok" and isinstance(tps_val, (int, float)) and tps_val > 0) else 0.0
        tps_cnt = 1 if tps_add > 0 else 0

        req_ok = 1 if group == "ok" else 0
        req_err = 1 if group == "err" else 0
        req_total = req_ok + req_err  # cancel (499) 不计入成败总账

        with self._lock:
            if self._conn is None:
                return None
            try:
                cur = self._conn.execute(
                    "INSERT INTO call_records (created_at, payload_json) VALUES (?, ?)",
                    (ts, json.dumps(entry, ensure_ascii=False)),
                )
                row_id = int(cur.lastrowid or 0)
                if row_id > 0:
                    # 写回一致的持久化 ID 方便展开详情匹配
                    entry_with_id = dict(entry)
                    entry_with_id["id"] = row_id
                    self._conn.execute(
                        "UPDATE call_records SET payload_json = ? WHERE id = ?",
                        (json.dumps(entry_with_id, ensure_ascii=False), row_id),
                    )
                # 滚动清理超出 500 条的旧明细
                self._conn.execute(
                    """
                    DELETE FROM call_records
                    WHERE id NOT IN (
                        SELECT id FROM call_records ORDER BY id DESC LIMIT ?
                    )
                    """,
                    (MAX_CALL_RECORDS,),
                )

                if req_total > 0 or retries > 0:
                    self._conn.execute(
                        """
                        INSERT INTO daily_model_stats (
                            day, model, channel,
                            req_total, req_ok, req_err, retries,
                            prompt_tk, cached_tk, thought_tk, output_tk,
                            ttft_sum, ttft_cnt, tps_sum, tps_cnt
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(day, model, channel) DO UPDATE SET
                            req_total  = req_total  + excluded.req_total,
                            req_ok     = req_ok     + excluded.req_ok,
                            req_err    = req_err    + excluded.req_err,
                            retries    = retries    + excluded.retries,
                            prompt_tk  = prompt_tk  + excluded.prompt_tk,
                            cached_tk  = cached_tk  + excluded.cached_tk,
                            thought_tk = thought_tk + excluded.thought_tk,
                            output_tk  = output_tk  + excluded.output_tk,
                            ttft_sum   = ttft_sum   + excluded.ttft_sum,
                            ttft_cnt   = ttft_cnt   + excluded.ttft_cnt,
                            tps_sum    = tps_sum    + excluded.tps_sum,
                            tps_cnt    = tps_cnt    + excluded.tps_cnt
                        """,
                        (
                            day, model, ch,
                            req_total, req_ok, req_err, retries,
                            p_tk, cached_tk, th_tk, out_tk,
                            ttft_add, ttft_cnt, tps_add, tps_cnt,
                        ),
                    )
                self._conn.commit()
                return row_id
            except Exception as e:
                print(f"⚠️ [统计持久化] 写入调用统计失败：{e}")
                return None

    def load_recent_records(self, limit: int = MAX_CALL_RECORDS) -> List[Dict[str, Any]]:
        with self._lock:
            if self._conn is None:
                return []
            try:
                rows = self._conn.execute(
                    "SELECT id, payload_json FROM call_records ORDER BY id DESC LIMIT ?",
                    (max(1, int(limit)),),
                ).fetchall()
                out: List[Dict[str, Any]] = []
                for r in rows:
                    try:
                        item = json.loads(r["payload_json"])
                        if isinstance(item, dict):
                            item["id"] = int(r["id"])
                            out.append(item)
                    except Exception:
                        continue
                return out
            except Exception:
                return []

    def clear_call_records(self) -> None:
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute("DELETE FROM call_records")
                self._conn.commit()
            except Exception as e:
                print(f"⚠️ [统计持久化] 清空调用记录失败：{e}")

    # ---------- 概览聚合查询（today / 7d / all + 7 天趋势 + 分模型统计） ----------

    def get_overview_stats(
        self,
        range_key: str = "all",
        *,
        uptime: float = 0.0,
        records_version: int = 0,
        records_count: int = 0,
    ) -> Dict[str, Any]:
        rk = range_key if range_key in ("today", "7d", "all") else "all"
        today_d = date.today()
        today_s = today_d.isoformat()
        week_start_s = (today_d - timedelta(days=6)).isoformat()

        total = ok = err = retries = 0
        prompt_tk = cached_tk = thought_tk = output_tk = 0
        express_req = cookie_req = 0
        models_out: List[Dict[str, Any]] = []
        day_totals: Dict[str, int] = {}

        with self._lock:
            if self._conn is not None:
                try:
                    where_sql = ""
                    params: tuple = ()
                    if rk == "today":
                        where_sql = "WHERE day = ?"
                        params = (today_s,)
                    elif rk == "7d":
                        where_sql = "WHERE day >= ?"
                        params = (week_start_s,)

                    rows = self._conn.execute(
                        f"""
                        SELECT
                            model,
                            channel,
                            SUM(req_total)  AS req_total,
                            SUM(req_ok)     AS req_ok,
                            SUM(req_err)    AS req_err,
                            SUM(retries)    AS retries,
                            SUM(prompt_tk)  AS prompt_tk,
                            SUM(cached_tk)  AS cached_tk,
                            SUM(thought_tk) AS thought_tk,
                            SUM(output_tk)  AS output_tk,
                            SUM(ttft_sum)   AS ttft_sum,
                            SUM(ttft_cnt)   AS ttft_cnt,
                            SUM(tps_sum)    AS tps_sum,
                            SUM(tps_cnt)    AS tps_cnt
                        FROM daily_model_stats
                        {where_sql}
                        GROUP BY model, channel
                        """,
                        params,
                    ).fetchall()

                    by_model: Dict[str, Dict[str, Any]] = {}
                    for r in rows:
                        rt = int(r["req_total"] or 0)
                        ro = int(r["req_ok"] or 0)
                        re_ = int(r["req_err"] or 0)
                        rr = int(r["retries"] or 0)
                        pt = int(r["prompt_tk"] or 0)
                        ct = int(r["cached_tk"] or 0)
                        tt = int(r["thought_tk"] or 0)
                        ot = int(r["output_tk"] or 0)
                        ch = str(r["channel"] or "express")

                        total += rt
                        ok += ro
                        err += re_
                        retries += rr
                        prompt_tk += pt
                        cached_tk += ct
                        thought_tk += tt
                        output_tk += ot
                        if ch == "cookie":
                            cookie_req += rt
                        else:
                            express_req += rt

                        mname = str(r["model"] or "unknown")
                        agg = by_model.get(mname)
                        if agg is None:
                            agg = {
                                "model": mname,
                                "channels": set(),
                                "calls": 0,
                                "ok": 0,
                                "err": 0,
                                "retries": 0,
                                "prompt_tk": 0,
                                "cached_tk": 0,
                                "thought_tk": 0,
                                "output_tk": 0,
                                "ttft_sum": 0.0,
                                "ttft_cnt": 0,
                                "tps_sum": 0.0,
                                "tps_cnt": 0,
                            }
                            by_model[mname] = agg
                        if rt > 0:
                            agg["channels"].add(ch)
                        agg["calls"] += rt
                        agg["ok"] += ro
                        agg["err"] += re_
                        agg["retries"] += rr
                        agg["prompt_tk"] += pt
                        agg["cached_tk"] += ct
                        agg["thought_tk"] += tt
                        agg["output_tk"] += ot
                        agg["ttft_sum"] += float(r["ttft_sum"] or 0.0)
                        agg["ttft_cnt"] += int(r["ttft_cnt"] or 0)
                        agg["tps_sum"] += float(r["tps_sum"] or 0.0)
                        agg["tps_cnt"] += int(r["tps_cnt"] or 0)

                    for mname, agg in by_model.items():
                        if agg["calls"] <= 0:
                            continue
                        chs = agg["channels"]
                        if chs == {"cookie"}:
                            ch_label = "Cookie"
                        elif chs == {"express"}:
                            ch_label = "Express"
                        else:
                            ch_label = "双通道"
                        total_m_tk = agg["prompt_tk"] + agg["thought_tk"] + agg["output_tk"]
                        models_out.append({
                            "model": mname,
                            "channel": ch_label,
                            "calls": agg["calls"],
                            "ok": agg["ok"],
                            "err": agg["err"],
                            "retries": agg["retries"],
                            "prompt_tk": agg["prompt_tk"],
                            "cached_tk": agg["cached_tk"],
                            "thought_tk": agg["thought_tk"],
                            "output_tk": agg["output_tk"],
                            "total_tk": total_m_tk,
                            "avg_ttft": round(agg["ttft_sum"] / agg["ttft_cnt"], 2) if agg["ttft_cnt"] > 0 else None,
                            "avg_tps": round(agg["tps_sum"] / agg["tps_cnt"], 1) if agg["tps_cnt"] > 0 else None,
                        })
                    models_out.sort(key=lambda x: (-x["calls"], -x["total_tk"], x["model"]))

                    # 近 7 天每日请求总量（固定 7 天连续数组）
                    sp_rows = self._conn.execute(
                        "SELECT day, SUM(req_total) AS total FROM daily_model_stats WHERE day >= ? GROUP BY day",
                        (week_start_s,),
                    ).fetchall()
                    day_totals = {str(r["day"]): int(r["total"] or 0) for r in sp_rows}
                except Exception as e:
                    print(f"⚠️ [统计持久化] 查询概览统计失败：{e}")

        sparkline_7d = []
        day_labels = ["6天前", "5天前", "4天前", "3天前", "前天", "昨天", "今日"]
        for i in range(7):
            d_obj = today_d - timedelta(days=6 - i)
            d_s = d_obj.isoformat()
            sparkline_7d.append({
                "day": d_s,
                "label": day_labels[i],
                "total": int(day_totals.get(d_s, 0)),
            })

        completion_tk = thought_tk + output_tk
        return {
            "range": rk,
            "uptime": round(float(uptime), 2),
            "total": total,
            "success": ok,
            "error": err,
            "retries": retries,
            "prompt_tokens": prompt_tk,
            "cached_tokens": cached_tk,
            "thought_tokens": thought_tk,
            "output_tokens": output_tk,
            "completion_tokens": completion_tk,
            "express_requests": express_req,
            "cookie_requests": cookie_req,
            "sparkline_7d": sparkline_7d,
            "models": models_out,
            "records_version": records_version,
            "records_count": records_count,
        }


# 单例导出
metrics_store = MetricsStore()
