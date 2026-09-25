"""geom.py — Minecraft 人物模型的立方体几何 + 每个面到 UV 矩形的「角点对应关系」。

为什么要有这个文件
------------------
前端要在 canvas 上把 64×64 图集贴到一个可旋转的人形上。难点不是投影，
而是**每一个面（top/bottom/left/right/front/back）的贴图朝向**：
贴反了只有 1 个像素的错位，肉眼看「好像也像个人」，但左右会镜像、上下会倒。

推导方式（不是猜的）：把图集当成盒子展开的**平铺网（net）**，从盒子外侧看，
相邻矩形共享的那条边就是 3D 里的真实棱。逐面推一遍得到：

* ``front``  (+Z)：u→+x，v→+y（图集不镜像，这是唯一被 2D 前视图验证过的面）
* ``back``   (-Z)：u→-x，v→+y
* ``right``  (-X)：u→+z，v→+y  （``right`` 是**角色自己的右侧**，在观察者左手边）
* ``left``   (+X)：u→-z，v→+y
* ``top``    (-y)：u→+x，v→+z  （**top 矩形的下边 = 正面**，这是 MC 的已知约定）
* ``bottom`` (+y)：u→-z，v→+x

坐标约定：本文件输出的是 **y 向上** 的世界坐标，X 向右（观察者视角，正对角色时），
Z 指向观察者。角色站在 X∈[0,16]、Y∈[-16,16]、Z 居中。
已知换算：``Y_world = 16 - y_figure``，其中 ``y_figure`` 就是现存 2D 前视图
（``webui/static/app.js`` 的 ``FIG`` 表）用的坐标。**因此在 yaw=pitch=0 时，
这里的 3D 渲染必须与那个已被用户目视确认过的 2D 前视图逐像素一致**——
探针就是拿这条当验收（见 ``_probe/probe_infer.js``）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "scripts", "lib"))

from skinatlas import face_index  # noqa: E402

FIG_H = 32.0        # 前视图高度（figure 单位）
FIG_W = 16.0        # 前视图宽度

#: 盒子在**前视图坐标**下的位置：(名字, x0, y0, 宽, 高, 厚度, 是否第二层, 外扩量)
#: 外扩量来自 Minecraft 的 overlay 层：帽子 +0.5，其余 +0.25（每侧）
BOXES: tuple[tuple[str, float, float, float, float, float, float], ...] = (
    ("head",     4, 0, 8, 8, 8, 0.0),
    ("hat",      4, 0, 8, 8, 8, 0.5),
    ("body",     4, 8, 8, 12, 4, 0.0),
    ("rarm",     0, 8, 4, 12, 4, 0.0),
    ("larm",    12, 8, 4, 12, 4, 0.0),
    ("rleg",     4, 20, 4, 12, 4, 0.0),
    ("lleg",     8, 20, 4, 12, 4, 0.0),
    ("body_ov",  4, 8, 8, 12, 4, 0.25),
    ("rarm_ov",  0, 8, 4, 12, 4, 0.25),
    ("larm_ov", 12, 8, 4, 12, 4, 0.25),
    ("rleg_ov",  4, 20, 4, 12, 4, 0.25),
    ("lleg_ov",  8, 20, 4, 12, 4, 0.25),
)

IS_OVERLAY = frozenset({"hat", "body_ov", "rarm_ov", "larm_ov", "rleg_ov", "lleg_ov"})

#: 面名 → 图集矩形在盒子内的偏移（依 skinatlas.face_rects 的同一套 net 约定）
_FACE_ORDER = ("top", "bottom", "right", "front", "left", "back")


def _offsets(w: float, d: float, h: float) -> dict[str, tuple[float, float, float, float]]:
    """面名 → 相对盒子左上角的 ``(dx, dy, fw, fh)``，单位是**贴图像素**。

    与 ``skinatlas.face_rects()`` 是同一套 net 约定（[right][front][left][back]），
    只是这里给的是相对偏移，便于按 expand 缩放。
    """
    return {
        "top":    (d,     0,     w, d),
        "bottom": (d + w, 0,     w, d),
        "right":  (0,     d,     d, h),
        "front":  (d,     d,     w, h),
        "left":   (d + w, d,     d, h),
        "back":   (d + w + d, d, w, h),
    }


def _corners(face: str, hw: float, hh: float, hd: float) -> list[list[float]]:
    """面在**以盒子中心为原点的 y-up 局部坐标**下，按 ``[TL, TR, BL, BR]`` 返回 4 个角。

    TL/TR/BL/BR 指**贴图矩形**的左上/右上/左下/右下角（不是世界坐标的上下左右）。
    朝向依据见模块 docstring 的 net 推导；``front`` 是唯一被 2D 前视图验证过的锚点。
    """
    x0, x1 = -hw, hw
    y0, y1 = hh, -hh        # y-up：y0 = 顶部（图集 v 小的一侧）
    z0, z1 = -hd, hd        # z1 = 朝观察者

    if face == "front":     # +Z, u→+x, v→↓
        return [[x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1]]
    if face == "back":      # -Z, u→-x, v→↓
        return [[x1, y0, z0], [x0, y0, z0], [x1, y1, z0], [x0, y1, z0]]
    if face == "right":     # -X, u→+z, v→↓
        return [[x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1]]
    if face == "left":      # +X, u→-z, v→↓
        return [[x1, y0, z1], [x1, y0, z0], [x1, y1, z1], [x1, y1, z0]]
    if face == "top":       # -y, u→+x, v→+z（矩形下边 = 正面）
        return [[x0, y0, z0], [x1, y0, z0], [x0, y0, z1], [x1, y0, z1]]
    if face == "bottom":    # +y, u→-z, v→+x
        return [[x0, y1, z1], [x0, y1, z0], [x1, y1, z1], [x1, y1, z0]]
    raise KeyError(face)


_NORMALS = {
    "front": (0.0, 0.0, 1.0), "back": (0.0, 0.0, -1.0),
    "right": (-1.0, 0.0, 0.0), "left": (1.0, 0.0, 0.0),
    "top": (0.0, 1.0, 0.0), "bottom": (0.0, -1.0, 0.0),
}


def model_geometry() -> dict:
    """输出给前端的完整几何：盒子 + 每个面的 UV 矩形与 4 个 3D 角点（y-up 世界坐标）。

    前端只做「旋转 → 正交投影 → 仿射贴图」，不含任何 Minecraft 知识。
    OK 之后 3D 视图与 UV 布局永远同步，不会出现「前端硬编码一份 UV，
    skinatlas 改了它不跟着改」这种分裂。
    """
    fi = face_index()
    boxes = []
    for name, x0, y0, w, h, d, expand in BOXES:
        # 前视图坐标 → y-up 世界坐标：Y = 16 - y_figure（16 是模型竖直中心）
        cx = x0 + w / 2.0
        cy = 16.0 - (y0 + h / 2.0)
        hw = w / 2.0 + expand
        hh = h / 2.0 + expand
        hd = d / 2.0 + expand
        faces = []
        for face, (dx, dy, fw, fh) in _offsets(w, d, h).items():
            rect = fi.get(f"{name}.{face}")
            if rect is None and expand:
                rect = fi.get(f"{name.replace('_ov', '')}.{face}")   # 兜底：不该发生
            if rect is None:
                continue
            ry0, ry1, rx0, rx1 = rect
            offs = _corners(face, hw, hh, hd)
            corners = [[round(cx + ox, 4), round(cy + oy, 4), round(oz, 4)]
                       for ox, oy, oz in offs]
            faces.append({
                "face": face,
                # 图集矩形（像素）。轴序**刻意保持 skinatlas.face_index 的原样**
                # ``[y0, y1, x0, x1]``——全项目（skinatlas / webui/static/app.js 的 FIG）
                # 都是这个口径，这里若图省事改成 [x0,y0,x1,y1] 会变成
                # 「同名不同序」的坑（曾经就是这么埋的）。
                "rect": [ry0, ry1, rx0, rx1],
                "corners": corners,                 # [TL, TR, BL, BR]
                "normal": list(_NORMALS[face]),
            })
        boxes.append({
            "name": name,
            "overlay": name in IS_OVERLAY,
            "expand": expand,
            "size": [w, h, d],
            "center": [round(cx, 4), round(cy, 4), 0.0],
            "faces": faces,
        })
    return {"fig_w": FIG_W, "fig_h": FIG_H,
            "faces_total": sum(len(b["faces"]) for b in boxes),
            "boxes": boxes}


def overlay_faces() -> list[str]:
    """第二层要跳过的面名（与前端 ``OVERLAY_FACES`` 同源，供探针断言）。"""
    fi = face_index()
    out = []
    for name, *_ in BOXES:
        if name not in IS_OVERLAY:
            continue
        for face in _FACE_ORDER:
            if f"{name}.{face}" in fi:
                out.append(f"{name}.{face}")
    return out


if __name__ == "__main__":
    import json
    g = model_geometry()
    print(json.dumps({"boxes": len(g["boxes"]), "faces": g["faces_total"],
                      "overlay_faces": len(overlay_faces()),
                      "head": g["boxes"][0]}, ensure_ascii=False, indent=1))
