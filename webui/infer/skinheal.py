"""skinheal.py — 破洞修补（Hole Healing）。

解决的问题
----------
不少皮肤的第一层（基础层）里会有**孤立的透明像素**，甚至连成一小段空白：
游戏里渲染成「皮肤上有个洞」，露出底下的模型盒，非常扎眼。
本模块把它们用**周围有颜色的像素**补上。

为什么只补「第一层」
--------------------
透明在第二层（overlay：帽子/外套…）是**合法状态**——真实皮肤只有
55~58% 会画臂/腿外层，整片留空很正常。如果把 overlay 的空白也当「洞」
补掉，等于强迫模型给所有皮肤画满第二层，方向完全错了。
所以修补范围严格限制在**基础层面的 UV 矩形**内（``base_region_mask``），
overlay 区域和图集里本来就没用的角落一律不碰。

算法（针对「连续空白」优化的洋葱式填充）
----------------------------------------
1. 在基础层面内找透明像素的**连通域**（4-连通）；
2. 面积超过 ``MAX_HOLE_AREA`` 的连通域直接跳过——那几乎一定是
   合法的留白（比如旧版 64×32 的空白下半部混进来的结构），不是洞；
3. 对每个「洞」做**洋葱式填充**（onion peel）：每一轮把「至少有一个
   已填/不透明邻居」的洞像素同时填上，颜色取其 8-邻居里不透明像素的
   平均 RGB。逐层向内推进，而不是拿整个洞的周长一次性平均——
   后者在细长的连续空白上会糊成一团平均色，前者能让周围的
   渐变/条纹自然地「长」进洞里。

性能：64×64 全图最多 4096 像素，纯 numpy + 小队列，单张 <1ms。
"""

from __future__ import annotations

import numpy as np

#: 洞的最大面积（像素数）。实测真实皮肤的破损洞多为 1~6 px，
#: 连续笔画式的空白一般在 20 px 以内；超过这个面积的基础层透明区
#: 几乎都是合法结构（如未使用的图集角落），补了反而画蛇添足。
MAX_HOLE_AREA = 20

# 4-邻域（连通域生长用）与 8-邻域（取色用）
_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def base_region_mask() -> np.ndarray:
    """``(64,64)`` bool：所有**基础层**面矩形的并集（overlay 与未用区域为 False）。

    与 ``webui/infer/server.py::overlay_mask`` 同源（都走 ``skinatlas.face_index``），
    但语义相反：这里是「允许修补的区域」。面名规则 ``<盒名>.<面名>``，
    盒名带 ``_ov`` / 等于 ``hat`` 的是第二层。
    """
    import sys
    import os
    lib = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "scripts", "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    from skinatlas import OVERLAY_BOXES, face_index

    m = np.zeros((64, 64), dtype=bool)
    for name, (y0, y1, x0, x1) in face_index().items():
        box = name.split(".")[0]
        if box in OVERLAY_BOXES:
            continue
        m[y0:y1, x0:x1] = True
    return m


def heal_image(arr: np.ndarray, base_mask: np.ndarray | None = None,
               max_area: int = MAX_HOLE_AREA) -> tuple[np.ndarray, dict]:
    """修补一张 ``(64,64,4)`` uint8 RGBA 皮肤的基础层破洞。

    返回 ``(修补后的 arr, 统计 dict)``。统计里：
    ``holes`` = 补掉的连通域个数；``filled`` = 补上的像素数；
    ``skipped_big`` = 因超面积被跳过的连通域个数。
    原地语义：直接修改并返回传入的 arr（调用方持有唯一副本）。
    """
    if base_mask is None:
        base_mask = base_region_mask()
    alpha = arr[..., 3]
    hole = (alpha < 128) & base_mask
    stats = {"holes": 0, "filled": 0, "skipped_big": 0}
    if not hole.any():
        return arr, stats

    h, w = hole.shape
    opaque = (alpha >= 128)
    # 连通域标记（BFS，4-连通）
    comp_id = np.full((h, w), -1, dtype=np.int32)
    comps: list[list[tuple[int, int]]] = []
    for y, x in zip(*np.nonzero(hole)):
        if comp_id[y, x] != -1:
            continue
        cid = len(comps)
        comp_id[y, x] = cid
        queue = [(y, x)]
        members = []
        while queue:
            cy, cx = queue.pop()
            members.append((cy, cx))
            for dy, dx in _N4:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and hole[ny, nx] and comp_id[ny, nx] == -1:
                    comp_id[ny, nx] = cid
                    queue.append((ny, nx))
        comps.append(members)

    rgb = arr[..., :3].astype(np.float32)
    for members in comps:
        if len(members) > max_area:
            stats["skipped_big"] += 1
            continue
        stats["holes"] += 1
        pending = set(members)
        # 洋葱式填充：每轮填「有不透明/已填邻居」的一整层，直到填完。
        while pending:
            layer = []
            for (y, x) in pending:
                src = []
                for dy, dx in _N8:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and opaque[ny, nx]:
                        src.append(rgb[ny, nx])
                if src:
                    layer.append((y, x, np.mean(src, axis=0)))
            if not layer:
                break        # 剩余像素 8-邻域内没有任何颜色源：安全放弃，保持透明
            for y, x, c in layer:
                rgb[y, x] = c
                arr[y, x, 3] = 255
                opaque[y, x] = True
                pending.discard((y, x))
                stats["filled"] += 1
    arr[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return arr, stats


if __name__ == "__main__":
    # 自检：构造一张带洞的假皮肤 → 补 → 验证
    rng = np.random.default_rng(0)
    a = np.zeros((64, 64, 4), dtype=np.uint8)
    a[..., :3] = rng.integers(0, 256, size=(64, 64, 3))
    a[..., 3] = 255
    bm = base_region_mask()
    # 基础层内打洞：单像素 / 连续 3 横排 / 5×5=25px 超面积方块（应跳过）。
    # 大块刻意放在 body.back 面（x20-28, y20-32）内——角落 (0-8,0-8) 是
    # 图集未用区域，不属于任何面，本来就不该被修补。
    holes = [(10, 10), (20, 20), (20, 21), (20, 22)]
    holes += [(y, x) for y in range(22, 27) for x in range(22, 27)]
    for y, x in holes:
        a[y, x, 3] = 0
    out, st = heal_image(a.copy(), bm)
    ok_small = out[10, 10, 3] == 255 and out[20, 22, 3] == 255
    ok_big = all(out[y, x, 3] == 0 for y, x in holes[4:])
    ok_region = out[50, 50, 3] == 255          # overlay/未用区不受影响
    print(f"holes={st['holes']} filled={st['filled']} skipped_big={st['skipped_big']}"
          f" | 小洞已补: {ok_small} | 大块保留: {ok_big} | 区域外不动: {ok_region}")
    assert ok_small and ok_big and ok_region
    assert st["holes"] == 2 and st["filled"] == 4 and st["skipped_big"] == 1, st
    print("SKINHEAL_SELFTEST_OK")
