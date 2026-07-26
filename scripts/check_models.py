#!/usr/bin/env python3
"""对着官方文档核对能力矩阵的小工具。

每次 Google 发新模型 / 改文档时跑一遍，比翻代码快：

    python scripts/check_models.py                 # 用 vertexModels.json
    python scripts/check_models.py gemini-4-pro    # 试算任意模型名

输出的每一列都对应 model_capabilities.py 里的一条判断，
逐列和官方文档比对即可确认是否需要更新白名单。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

import model_capabilities as mc  # noqa: E402


def load_models() -> list:
    if len(sys.argv) > 1:
        return sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, "..", "vertexModels.json"),
                 os.path.join(here, "..", "app", "vertexModels.json")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("models", [])
    return []


def main() -> int:
    models = load_models()
    if not models:
        print("没有找到模型清单。用法：python scripts/check_models.py [模型名 ...]")
        return 1

    header = (f"{'模型':<26} {'家族':<6} {'思考':<7} {'档位/预算':<28} "
              f"{'temp':<5} {'top_p':<6} {'top_k':<6} {'n':<4} 生图")
    print(header)
    print("-" * len(header))
    for m in models:
        cap = mc.capabilities_summary(m)
        prof = mc.get_profile(m)
        th = cap["thinking"]
        if th["kind"] == "level":
            detail = f"{'/'.join(th['levels'])} (默认 {prof.get('default_level', '-')})"
        elif th["kind"] == "budget":
            detail = f"{th['budget_min']}-{th['budget_max']} 可关={th['can_off']}"
        else:
            detail = "-"
        allowed = prof["allowed_sampling"]
        yn = lambda k: "✓" if k in allowed else "✗"      # noqa: E731
        img = (f"{'/'.join(cap['image_sizes'])} · {len(cap['image_aspect_ratios'])} 种比例"
               if cap["is_image"] else "-")
        print(f"{m:<26} {cap['family']:<6} {str(th['kind']):<7} {detail:<28} "
              f"{yn('temperature'):<5} {yn('top_p'):<6} {yn('top_k'):<6} "
              f"{yn('candidate_count'):<4} {img}")

    print("\n核对清单：")
    print("  1. temp/top_p/top_k 打 ✓ 的模型，是否确实还没被官方标为弃用？")
    print("     （官方：自 3.6 Flash 与 3.5 Flash-Lite 起及所有更新模型均已弃用）")
    print("  2. 思考档位集合与默认值是否与官方模型页一致？")
    print("  3. 生图分辨率/比例数量是否与官方模型页一致？")
    print("  4. 是否有官方已停用的模型仍留在清单里？")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
