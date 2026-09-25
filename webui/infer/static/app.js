/* app.js — drmage 实验台前端（零依赖，无 CDN）
 *
 * 三块自研渲染：
 *   R3D   —— 正交投影 + 仿射贴图的可旋转人形（几何由后端 geom.py 给，前端不含 MC 知识）
 *   FLAT  —— 前视拼图（只有 front 面，即现存监控面板里那套已被目视确认的口径）
 *   UV    —— 图集 + 逐面网格
 * 3D 在 yaw=pitch=0 时应与前视拼图**逐像素一致**，探针就是拿这条当验收。
 */
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const S = {
  meta: null, geo: null, faceIndex: {}, overlayFaces: new Set(),
  condMode: 'spec', quantOpts: [], items: [], realItems: [],
  job: null, poll: null, thumb3D: true, cur: null,
  view: { yaw: 0, pitch: 0, zoom: 1 },
  // 'all' | 'base' —— **唯一**的「看几层」状态源。
  // 结果区的下拉、弹窗里那个「第二层」复选框、导出 zip 的 layer 参数，
  // 三处都读它、也都能改它。分成三份状态就会出现
  // 「缩略图看了两层、导出只有一层」这种对不上的情况。
  layer: 'all',
};

/* ==========================================================================
   工具
   ========================================================================== */
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k === 'txt') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) e.appendChild(k);
  return e;
}
const fmt = (n, d = 2) => (n === null || n === undefined || Number.isNaN(n)) ? '—' : Number(n).toFixed(d);
const int_ = (n) => (n === null || n === undefined) ? '—' : Number(n).toLocaleString('en-US');

function reportErr(msg) {
  const el = $('#err');
  el.hidden = false;
  el.textContent = msg;
  console.error('[infer]', msg);
}
function clearErr() { $('#err').hidden = true; }

async function api(path, opt) {
  const r = await fetch(path, opt);
  let j = null;
  try { j = await r.json(); } catch (e) { /* 可能是 zip */ }
  if (!r.ok) throw new Error((j && j.error) || (r.status + ' ' + r.statusText));
  return j;
}

/* 图片对象缓存（同一 key 只拉一次） */
const _imgs = new Map();
function loadImg(url) {
  if (_imgs.has(url)) return _imgs.get(url);
  const p = new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error('图片加载失败 ' + url));
    im.src = url;
  });
  _imgs.set(url, p);
  return p;
}

/* ==========================================================================
   3D 渲染器
   ========================================================================== */
/* 面贴图用的「切片」缓存。
   为什么要切片而不是给 setTransform 加 clip：
   仿射变换会把**整张 64×64 图集**铺进面所在的平面，只有 rect 那一块落在面内，
   其余全在面外——不裁剪就会把整张图当贴图糊上去（曾经就是这个 bug，
   表现为 3D 里看不到人形、只有一整块纹理）。
   用 clip 裁剪会引入 clip 自身的抗锯齿，与前视拼图 raster() 的
   drawImage 边缘口径对不上；所以改用「先把 rect 切出来，
   再让变换的源 (0,0) 落在面左上角」，边缘由同一个 drawImage 决定，口径一致。
   切片挂在 Image 对象上，换图自然失效。 */
const _tileCache = new WeakMap();
function faceTile(img, rx0, ry0, rw, rh) {
  let m = _tileCache.get(img);
  if (!m) { m = new Map(); _tileCache.set(img, m); }
  const key = rx0 + ',' + ry0 + ',' + rw + ',' + rh;
  let t = m.get(key);
  if (!t) {
    t = document.createElement('canvas');
    t.width = rw; t.height = rh;
    const tc = t.getContext('2d');
    tc.imageSmoothingEnabled = false;
    tc.drawImage(img, rx0, ry0, rw, rh, 0, 0, rw, rh);
    m.set(key, t);
  }
  return t;
}

const R3D = {
  /* 把 yaw/pitch/zoom 作用到世界坐标点，返回屏幕坐标与深度 */
  proj(p, yaw, pitch, s, cx, cy) {
    const cyaw = Math.cos(yaw), syaw = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x = p[0] - 8, y = p[1], z = p[2];      // 绕模型中心 (8,0,0) 转
    const x1 = x * cyaw + z * syaw;
    const z1 = -x * syaw + z * cyaw;
    const y2 = y * cp - z1 * sp;
    const z2 = y * sp + z1 * cp;
    return [cx + x1 * s, cy - y2 * s, z2];
  },
  /** 渲染到 canvas。``opt``：{yaw,pitch,zoom,withOverlay,bleed} */
  render(canvas, img, opt = {}) {
    const base = opt.base || 6.2;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const zoom = opt.zoom || 1;
    const yaw = opt.yaw || 0, pitch = opt.pitch || 0;
    const cssW = 16 * base * zoom, cssH = 32 * base * zoom;
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    const devW = Math.max(1, Math.round(cssW * dpr));
    const devH = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== devW || canvas.height !== devH) {
      canvas.width = devW; canvas.height = devH;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, devW, devH);
    if (!S.geo) return { faces: 0 };
    const s = base * zoom * dpr;
    const cx = 8 * s, cy = 16 * s;
    // 旋转时相邻面的仿射填充会留下亚像素缝 → 沿面法方向外扩一点点盖住。
    // yaw=pitch=0 时**不外扩**，这样它可以与前视拼图逐像素对齐（探针验收点）。
    const rot = Math.abs(yaw) > 1e-6 || Math.abs(pitch) > 1e-6;
    const bleed = opt.bleed !== undefined ? opt.bleed : (rot ? 0.6 * dpr : 0);

    const cyaw = Math.cos(yaw), syaw = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const draw = [];
    for (const box of S.geo.boxes) {
      if (!(opt.withOverlay !== false) && box.overlay) continue;
      for (const f of box.faces) {
        const n = f.normal;
        const nz = n[1] * sp + (-n[0] * syaw + n[2] * cyaw) * cp;
        if (nz <= 1e-6) continue;                       // 背面剔除
        const p = f.corners.map((q) => this.proj(q, yaw, pitch, s, cx, cy));
        const zc = (p[0][2] + p[1][2] + p[2][2] + p[3][2]) / 4;
        draw.push({ f, p, z: zc });
      }
    }
    draw.sort((a, b) => a.z - b.z);                     // 远 → 近（画家算法）
    let m = 0;
    for (const it of draw) {
      const p = it.p;
      let q = p;
      if (bleed) {
        const mx = (p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4;
        const my = (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4;
        q = p.map((pt) => {
          const dx = pt[0] - mx, dy = pt[1] - my;
          const L = Math.hypot(dx, dy) || 1;
          return [pt[0] + dx / L * bleed, pt[1] + dy / L * bleed];
        });
      }
      // rect 是 [y0, y1, x0, x1]（与 face_index 同一轴序，别想当然写成 x 在前）
      const [ry0, ry1, rx0, rx1] = it.f.rect;
      const rw = rx1 - rx0, rh = ry1 - ry0;
      if (rw <= 0 || rh <= 0) continue;
      // 仿射：贴图坐标 (u,v) → 屏幕。正交投影下面还是平行四边形，所以仿射是**精确**的。
      const a = (q[1][0] - q[0][0]) / rw, b = (q[1][1] - q[0][1]) / rw;
      const c = (q[2][0] - q[0][0]) / rh, d = (q[2][1] - q[0][1]) / rh;
      const e = q[0][0], f_ = q[0][1];
      const tile = faceTile(img, rx0, ry0, rw, rh);
      ctx.save();
      ctx.setTransform(a, b, c, d, e, f_);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tile, 0, 0);
      ctx.restore();
      m++;
    }
    return { faces: m, w: devW, h: devH };
  },
};

/* 前视拼图：只画 front 面（与监控面板同一口径，作为 3D 的对照基准） */
const FRONT_FACES = ['head.front', 'body.front', 'rarm.front', 'larm.front',
  'rleg.front', 'lleg.front', 'hat.front', 'body_ov.front', 'rarm_ov.front',
  'larm_ov.front', 'rleg_ov.front', 'lleg_ov.front'];
const FIG_POS = {
  'head.front': [4, 0], 'body.front': [4, 8], 'rarm.front': [0, 8],
  'larm.front': [12, 8], 'rleg.front': [4, 20], 'lleg.front': [8, 20],
  'hat.front': [4, 0], 'body_ov.front': [4, 8], 'rarm_ov.front': [0, 8],
  'larm_ov.front': [12, 8], 'rleg_ov.front': [4, 20], 'lleg_ov.front': [8, 20],
};
function raster(canvas, img, withOverlay) {
  canvas.width = 336; canvas.height = 672;
  canvas.style.width = '100%'; canvas.style.height = 'auto';
  const c = canvas.getContext('2d');
  c.imageSmoothingEnabled = false;
  c.clearRect(0, 0, 336, 672);
  for (const name of FRONT_FACES) {
    if (!withOverlay && S.overlayFaces.has(name)) continue;
    const r = S.faceIndex[name];
    const pos = FIG_POS[name];
    if (!r || !pos) continue;
    const [y0, y1, x0, x1] = r;
    c.drawImage(img, x0, y0, x1 - x0, y1 - y0,
      pos[0] * 21, pos[1] * 21, (x1 - x0) * 21, (y1 - y0) * 21);
  }
}

/* UV 展开 + 面网格 */
function paintUV(canvas, img, withGrid) {
  const sc = 5, W = 64 * sc;
  canvas.width = W; canvas.height = W;
  canvas.style.width = '100%'; canvas.style.height = 'auto';
  const c = canvas.getContext('2d');
  c.imageSmoothingEnabled = false;
  c.fillStyle = '#101319';
  c.fillRect(0, 0, W, W);
  c.drawImage(img, 0, 0, 64, 64, 0, 0, W, W);
  if (withGrid) {
    c.lineWidth = 1;
    for (const [name, r] of Object.entries(S.faceIndex)) {
      const ov = S.overlayFaces.has(name);
      c.strokeStyle = ov ? 'rgba(255,120,220,.55)' : 'rgba(90,220,255,.42)';
      c.strokeRect(r[2] * sc + .5, r[0] * sc + .5, (r[3] - r[2]) * sc, (r[1] - r[0]) * sc);
    }
  }
}

/* ==========================================================================
   单元格（结果 / 对照共用）
   ========================================================================== */
function makeCell(item, opt = {}) {
  const card = h('div', { class: 'cell' });
  const cv = h('canvas');
  card.appendChild(cv);
  if (opt.badge) card.appendChild(h('div', { class: 'tag', txt: opt.badge }));
  const m = item.per || null;
  const line = opt.meta !== false
    ? (m ? `${m.unique_colors} 色 · 可见 ${(m.visible * 100).toFixed(0)}%`
      : (item.cond ? (item.cond.tone || '') : ''))
    : '';
  if (line) card.appendChild(h('div', { class: 'meta', txt: line }));
  card.addEventListener('click', () => openModal(item));
  cv.__item = item;
  cv.__card = card;
  _cells.push({ cv, card, item, opt });
  return card;
}
const _cells = [];

const withOverlay = () => S.layer !== 'base';

async function drawCell(cv, item) {
  const im = await loadImg('/api/img?key=' + item.key);
  const ov = withOverlay();
  if (S.thumb3D) R3D.render(cv, im, { base: 5.9, yaw: -0.42, pitch: 0.13, zoom: 1.05, withOverlay: ov });
  else raster(cv, im, ov);
}

/** 重画所有格子（结果区 / 对照栏 / 平面↔3D 切换都走这里）。返回 Promise。 */
function paintCells() {
  // 两个 grid 都是先 innerHTML='' 再重建的，旧格子已经脱离文档。
  // 顺手回收，免得 _cells 跨多次生成无限增长，也免得去重画死画布。
  for (let i = _cells.length - 1; i >= 0; i--) {
    if (!_cells[i].cv.isConnected) _cells.splice(i, 1);
  }
  // 这里**绝不能**再写 `catch (e) { /* ignore */ }`。
  // 曾经就是那个空 catch 把「_cells 里存的是 {cv,item,opt}、这里却读 c.card」
  // 这个 TypeError 吞了整整一轮开发——表现是**所有缩略图全空白**、
  // 而控制台一片安静、探针全绿。有错就让它响。
  return Promise.all(_cells.map((c) =>
    drawCell(c.cv, c.item).catch((e) => console.error('缩略图绘制失败', c.item && c.item.key, e))));
}

/* ==========================================================================
   生成 / 轮询
   ========================================================================== */
function collectReq() {
  const spec = {
    hue_deg: +$('#sHue').value,
    hue_spread: +$('#sSpread').value,
    sat_mean: +$('#sSat').value,
    val_mean: +$('#sVal').value,
    dark_ratio: +$('#sDark').value,
    bright_ratio: +$('#sBright').value,
    neutral_ratio: +$('#sNeutral').value,
    high_sat_ratio: +$('#sSat').value,
    tone: $('#selTone').value,
    saturation_class: $('#selSatCls').value,
    complexity_class: $('#selCpx').value,
    overlay_used: $('#cbOverlay').checked,
    model_type: 'classic',
  };
  const q = S.quantOpts[+$('#selQuant').value] || { value: 'off' };
  const quant = { mode: q.value, k: q.value === 'adaptive' ? +$('#sK').value : (q.k || 0) };
  if (q.palette) quant.palette = q.palette;
  const heal = $('#cbHeal').checked;
  const base = {
    n: +$('#sN').value,
    ddim: +$('#sDdim').value,
    eta: +$('#sEta').value,
    seed: +$('#inSeed').value,
    alpha: $('#selAlpha').value,
    quant: quant,
    heal: heal,
    ckpt: $('#ckptSel').value,
    force: $('#cbForce').checked,
  };
  if (S.condMode === 'lottery') {
    return Object.assign(base, {
      cond: {
        mode: 'lottery',
        n_pool: +$('#selPool').value,
        batch: +$('#selBatch').value,
        final_k: +$('#inFinalK').value,
        tone: $('#cbToneLot').checked ? $('#selToneLot').value : null,
      },
    });
  }
  const useReal = S.condMode === 'real';
  return Object.assign(base, {
    cond: useReal
      ? { mode: 'real', tone: $('#cbTone').checked ? $('#selTone').value : null }
      : { mode: 'spec', spec: spec },
  });
}

async function doGenerate() {
  clearErr();
  const btn = $('#btnGen');
  btn.disabled = true;
  $('#prog').hidden = false;
  const req = collectReq();
  // 「真实抽样」模式下，条件条数 = 真实张数
  if (S.condMode === 'real') req.n = +$('#sRealN').value;
  try {
    const r = await api('/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (r.error) throw new Error(r.error);
    S.job = r.job;
    pollJob();
  } catch (e) {
    reportErr('生成失败：' + e.message);
    btn.disabled = false;
    $('#prog').hidden = true;
  }
}

function pollJob() {
  if (S.poll) clearTimeout(S.poll);
  S.poll = setTimeout(async () => {
    try {
      const j = await api('/api/job?id=' + S.job);
      if (typeof j.progress === 'number') {           // 抽奖模式的分批进度
        const bar = $('#prog i');
        if (bar) bar.style.width = (j.progress * 100).toFixed(0) + '%';
      }
      if (j.state === 'done') {
        S.items = j.items || [];
        await renderResults(j);
        $('#btnGen').disabled = false;
        $('#prog').hidden = true;
        const bar = $('#prog i'); if (bar) bar.style.width = '0%';
        $('#btnExport').disabled = !S.items.length;
        if (S.condMode === 'real') runReal();       // 对照条与生成条件同分布
      } else if (j.state === 'error') {
        reportErr('生成失败：' + (j.error || '未知错误'));
        $('#btnGen').disabled = false;
        $('#prog').hidden = true;
      } else {
        pollJob();
      }
    } catch (e) {
      reportErr('查询任务失败：' + e.message);
      $('#btnGen').disabled = false;
      $('#prog').hidden = true;
    }
  }, 400);
}

const SCORE_CN = {
  color_realism: '颜色真实度', block_structure: '色块结构',
  limb_symmetry: '肢体镜像', face_coherence: '脸部连贯',
  cond_adherence: '条件贴合',
  learned: '判别器 P(real)', rule_total: '规则分（参考）',
  richness_rank: '丰富度（池内排名）', coherence_rank: '连贯度（池内排名）',
  coverage_factor: '覆盖率因子',
};

function renderResults(j) {
  const grid = $('#genGrid');
  const rail = $('#finalRail');
  grid.innerHTML = '';
  rail.innerHTML = '';
  if (!S.items.length) {
    grid.appendChild(h('div', { class: 'empty', txt: '没有结果' }));
    rail.hidden = true;
    return;
  }
  const su = j.summary || {};
  const lot = j.lottery || null;

  if (lot) {
    // —— 抽奖模式：最终输出条（top-K）+ 候选网格（全部，按分数降序）——
    const finals = S.items.filter((i) => i.final_rank);
    const cands = S.items;
    rail.hidden = false;
    for (const it of finals) {
      rail.appendChild(makeCell(it, {
        badge: `🏆 第${it.final_rank}名 · ${it.score.toFixed(3)}`,
      }));
    }
    $('#finalStat').textContent =
      `${lot.final_k} / ${lot.n_actual} 张候选 · 抽奖 ${fmt(lot.seconds, 1)}s`
      + `（${fmt(lot.per_image_seconds, 2)}s/张）`
      + ` · 候选分：中位 ${fmt(lot.score_median, 3)} / 最低 ${fmt(lot.score_min, 3)}`
      + (lot.heal ? ' · 已修补' : '');
    for (const it of cands) {
      grid.appendChild(makeCell(it, {
        badge: it.final_rank ? `🏆#${it.final_rank} ${it.score.toFixed(3)}`
          : `候选 ${it.score.toFixed(3)}`,
      }));
    }
    $('#statLine').textContent =
      `${cands.length} 张候选 · 均分 ${fmt(lot.score_mean, 3)} · top ${fmt(lot.score_top, 3)}`
      + ` · DDIM ${su.ddim_steps} · seed ${su.seed}`
      + (lot.micro_batch !== lot.batch ? ` · 显存微批 ${lot.micro_batch}` : '');
  } else {
    rail.hidden = true;
    $('#finalStat').textContent = '抽奖模式的评分 top-K 会出现在这里';
    for (const it of S.items) grid.appendChild(makeCell(it, { badge: '#' + it.idx }));
    $('#statLine').textContent =
      `${S.items.length} 张 · ${fmt(su.seconds, 1)}s（${fmt(su.per_image_seconds, 2)}s/张）`
      + ` · 平均 ${int_(su.unique_colors_mean)} 色 · 可见 ${fmt((su.visible_mean || 0) * 100, 1)}%`
      + ` · DDIM ${su.ddim_steps} · seed ${su.seed}`;
  }
  return paintCells();
}

async function runReal() {
  try {
    const n = S.condMode === 'real' ? +$('#sRealN').value : 12;
    const tone = $('#cbTone').checked ? $('#selTone').value : '';
    const r = await api(`/api/real?n=${n}&seed=${(Math.random() * 1e6) | 0}`
      + (tone ? `&tone=${tone}` : ''));
    S.realItems = r.items || [];
    const rail = $('#realRail');
    rail.innerHTML = '';
    if (!S.realItems.length) {
      rail.appendChild(h('div', { class: 'empty small', txt: '没有符合条件的真实样本' }));
      $('#realStat').textContent = '—';
      return;
    }
    for (const it of S.realItems) {
      rail.appendChild(makeCell(it, { meta: false, badge: '#' + it.real_index }));
    }
    $('#realStat').textContent = `${S.realItems.length} 张 · 检查渲染是否正常、条件是否对得上`;
    await paintCells();
  } catch (e) {
    reportErr('取真实对照失败：' + e.message);
  }
}

/* ==========================================================================
   大图 / 3D 弹窗
   ========================================================================== */
async function openModal(item) {
  S.cur = item;
  $('#modal').hidden = false;
  const im = await loadImg('/api/img?key=' + item.key);
  S.curImg = im;
  S.view = { yaw: -0.42, pitch: 0.13, zoom: 1 };
  // 弹窗里的「第二层」复选框是 S.layer 的视图：用户可能先在结果区改了层再点进来的。
  $('#cbOv').checked = withOverlay();
  drawModal();
  paintUV($('#cUV'), im, $('#cbGrid').checked);
  const m = item.per || {};
  const rows = [];
  const add = (k, v) => rows.push(`<div class="k">${k}</div><div class="v">${v}</div>`);
  // 来源标注：抽奖候选是**生成图**（条件借自真实 val），不能标成「真实」——
  // 曾因此把生成图标成「真实 #15」，让人误以为真实皮肤也只有这么低分。
  const isLot = item.origin === 'lottery_gen';
  add('来源', isLot
    ? `生成（条件：真实 val #${item.real_index}）`
    : (item.real_index !== null && item.real_index !== undefined
        ? `真实 val #${item.real_index}` : `生成 #${item.idx}`));
  if (m.visible !== undefined) add('可见率', (m.visible * 100).toFixed(1) + '%');
  if (m.unique_colors !== undefined) add('唯一色数', int_(m.unique_colors));
  if (m.near_black !== undefined && m.near_black !== null) {
    add('近黑率', fmt(m.near_black, 3));
  }
  if (m.mean_rgb) add('平均 RGB', m.mean_rgb.join(', '));
  if (m.healed_px) add('破洞修补', `补了 ${m.healed_px} 个像素`);
  if (item.score !== undefined && item.score !== null) {
    add('评分总分', item.score.toFixed(4)
      + (item.final_rank ? `（最终输出 第${item.final_rank}名）` : '（候选）'));
    if (item.score_parts) {
      for (const [k, v] of Object.entries(item.score_parts)) {
        if (v === null || v === undefined) continue;
        add(SCORE_CN[k] || k, v.toFixed(3));
      }
    }
    if (item.score_stats && item.score_stats.adh_dist !== undefined) {
      add('条件贴合距离', item.score_stats.adh_dist.toFixed(4)
        + '（真实皮肤 ≈0，越大越不听条件）');
    }
  }
  const c = item.cond || {};
  if (c.tone) {
    add('色调', c.tone);
    add('饱和档', c.saturation_class || '—');
    add('复杂度', c.complexity_class || '—');
    add('主导色相', (c.dominant_hue_deg ?? '—') + '°');
    add('明度', fmt(c.val_mean, 3));
    add('饱和度', fmt(c.sat_mean, 3));
  }
  const mo = S.meta && S.meta.loaded;
  if (mo) {
    add('权重', mo.rel.split('/').slice(-2).join('/'));
    add('参数量', mo.params_m + 'M');
    add('训练步数', mo.step ? int_(mo.step) : '—');
  }
  $('#mBody').innerHTML =
    `<div class="kv">${rows.join('')}</div>`
    + `<p class="hint">透明像素的 RGB 已归零（游戏内不渲染）。下载的 PNG 是标准 64×64 RGBA，
       alpha 严格 0/255。</p>`;
  const mi = isLot
    ? `抽奖候选 #${item.idx}（条件 val #${item.real_index}）`
    : (item.real_index !== null && item.real_index !== undefined
        ? `真实 #${item.real_index}` : `生成 #${item.idx}`);
  $('#mTitle').textContent = '皮肤 ' + mi;
  $('#btnCopy').hidden = !c.tone;
}

function closeModal() {
  $('#modal').hidden = true;
  S.cur = null; S.curImg = null;
  if (S.autoTimer) { clearInterval(S.autoTimer); S.autoTimer = null; $('#cbAuto').checked = false; }
}

function drawModal() {
  if (!S.curImg) return;
  R3D.render($('#c3d'), S.curImg, {
    base: 21, yaw: S.view.yaw, pitch: S.view.pitch, zoom: S.view.zoom,
    withOverlay: withOverlay(),
  });
}

/* 「看几层」的**唯一**入口：改状态 → 同步三处 UI → 重画。
 *
 * 三处 UI 指的是：结果区下拉 #selLayer、弹窗复选框 #cbOv、导出按钮的提示文字。
 * 全部走这一个函数，就不会出现某一处忘了同步、显示和实际不一致的情况。
 * 注意给 checkbox.checked 赋值**不会**触发 change 事件，所以从 #cbOv 进来时
 * 再赋值一次是安全的，不会绕成死循环。
 */
function setLayer(v, opt = {}) {
  S.layer = (v === 'base') ? 'base' : 'all';
  $('#selLayer').value = S.layer;
  $('#cbOv').checked = S.layer === 'all';
  syncExportLabel();
  if (S.curImg) drawModal();
  // opt.skipCells：调用方自己会重画时（比如 paintCells 的调用链）避免重复劳动
  return opt.skipCells ? undefined : paintCells();
}

/** 导出按钮的文字带上当前层，免得「我明明看着两层，导出来只有一层」这种困惑。 */
function syncExportLabel() {
  const b = $('#btnExport');
  b.textContent = S.layer === 'base' ? '⬇ 导出 zip（仅第一层）' : '⬇ 导出 zip（含第二层）';
  b.title = S.layer === 'base'
    ? 'zip 里的 PNG 会把第二层像素整片置为透明'
    : 'zip 里是完整的 64×64 RGBA（两层都在）';
}
/* 拖拽旋转 / 滚轮缩放 */
function init3DInput() {
  const cv = $('#c3d');
  let drag = null;
  const down = (e) => {
    drag = { x: e.clientX, y: e.clientY, yaw: S.view.yaw, pitch: S.view.pitch };
    cv.setPointerCapture && cv.setPointerCapture(e.pointerId);
  };
  const move = (e) => {
    if (!drag) return;
    S.view.yaw = drag.yaw + (e.clientX - drag.x) * 0.011;
    S.view.pitch = Math.max(-1.35, Math.min(1.35,
      drag.pitch + (e.clientY - drag.y) * 0.011));
    drawModal();
  };
  const up = () => { drag = null; };
  cv.addEventListener('pointerdown', down);
  cv.addEventListener('pointermove', move);
  cv.addEventListener('pointerup', up);
  cv.addEventListener('pointercancel', up);
  cv.addEventListener('pointerleave', up);
  cv.addEventListener('wheel', (e) => {
    e.preventDefault();
    S.view.zoom = Math.max(0.45, Math.min(2.6, S.view.zoom * (e.deltaY < 0 ? 1.12 : 0.89)));
    drawModal();
  }, { passive: false });
  cv.addEventListener('dblclick', () => {
    S.view = { yaw: -0.42, pitch: 0.13, zoom: 1 };
    $('#cbAuto').checked = false;
    if (S.autoTimer) { clearInterval(S.autoTimer); S.autoTimer = null; }
    drawModal();
  });
}

/* ==========================================================================
   控件绑定
   ========================================================================== */
function bindRange(id, out, fmtFn) {
  const el = $(id), o = $(out);
  const upd = () => { o.textContent = fmtFn ? fmtFn(+el.value) : el.value; };
  el.addEventListener('input', upd);
  upd();
}
const IDV = (v) => v;

function specSig() {
  // 「条件变了 → 真实对照条该换一批」的粗略信号，避免每次生成都重拉
  return [$('#selTone').value, $('#cbTone').checked, $('#sRealN').value].join('|');
}

async function boot() {
  try {
    let meta = await api('/api/meta');
    // 首屏就把权重载进来：临时平台上「点了就能出图」比「启动快 3 秒」重要得多。
    if (!meta.loaded) {
      $('#brandSub').textContent = '正在载入权重…';
      try {
        await api('/api/reload', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ckpt: meta.default_ckpt }),
        });
        meta = await api('/api/meta');
      } catch (e) { /* 载入失败也继续，点生成时会再试并给出明确报错 */ }
    }
    S.meta = meta;
    S.geo = meta.geometry;
    S.faceIndex = meta.face_index || {};
    S.overlayFaces = new Set(meta.overlay_faces || []);
    S.quantOpts = meta.quant_opts || [];

    $('#brandSub').textContent =
      `${meta.loaded ? meta.loaded.rel.replace('models/', '') : '权重未载入'}`
      + ` · ${meta.geo ? meta.geo.faces_total : 72} 面 · 服务 ${meta.now}`;
    const tr = meta.training || {};
    const cg = $('#chipGpu'), ct = $('#chipTrain');
    if (meta.gpu && meta.gpu.cuda) {
      cg.textContent = `GPU ${meta.gpu.free_gb} / ${meta.gpu.total_gb} GB 可用`;
      cg.className = 'chip ' + (meta.gpu.free_gb < meta.min_free_vram_gb ? 'err' : 'ok');
    } else {
      cg.textContent = 'GPU 不可用（走 CPU，会很慢）';
      cg.className = 'chip warn';
    }
    if (tr.status === 'running') {
      ct.textContent = `训练中 · ${tr.tag} ep${tr.epoch}`;
      ct.className = 'chip warn';
    } else {
      ct.textContent = `训练 ${tr.status_cn || tr.status || '—'}`;
      ct.className = 'chip';
    }

    // 权重下拉
    const sel = $('#ckptSel');
    sel.innerHTML = '';
    for (const c of meta.ckpts) {
      const label = `${c.rel.replace('models/', '')}`
        + `${c.preferred ? ' ★' : ''} · ${c.kind} · ${c.size_mb}MB · ${c.mtime_cn}`;
      sel.appendChild(h('option', { value: c.rel, txt: label }));
    }
    sel.value = meta.default_ckpt;

    // 条件枚举
    const cd = meta.cond;
    for (const k of cd.tone) $('#selTone').appendChild(h('option', { value: k, txt: k }));
    for (const k of cd.saturation_class) $('#selSatCls').appendChild(h('option', { value: k, txt: k }));
    for (const k of cd.complexity_class) $('#selCpx').appendChild(h('option', { value: k, txt: k }));
    $('#selTone').value = cd.defaults.tone;
    $('#selSatCls').value = cd.defaults.saturation_class;
    $('#selCpx').value = cd.defaults.complexity_class;

    for (const [k, note] of Object.entries(meta.alpha_modes || {})) {
      $('#selAlpha').appendChild(h('option', { value: k, txt: `${k} — ${note}` }));
    }
    // 默认 retrieval（与训练侧同分布的真实 mask）。engine.DEFAULT_ALPHA_MODE 同此口径；
    // template 是已知会出椒盐/穿孔的旧行为，只留给 A/B。
    $('#selAlpha').value = 'retrieval';

    // 抽奖模式的色调筛选下拉与「滑块合成」的色调共用枚举
    const cdTone = meta.cond.tone || [];
    for (const k of cdTone) $('#selToneLot').appendChild(h('option', { value: k, txt: k }));
    $('#selToneLot').value = cd.defaults.tone;

    for (let i = 0; i < S.quantOpts.length; i++) {
      const q = S.quantOpts[i];
      $('#selQuant').appendChild(h('option', { value: i, txt: q.label }));
    }
    // 默认选「自适应 K=64」（实测最贴真实：色数 64 对真实 66）；找不到就维持第一项
    const di = S.quantOpts.findIndex(x => x.value === 'adaptive' && x.k === 64);
    if (di >= 0) $('#selQuant').value = String(di);
    syncQuant();

    bindRange('#sHue', '#vHue', (v) => v + '°');
    bindRange('#sSpread', '#vSpread', (v) => v.toFixed(2));
    bindRange('#sSat', '#vSat', (v) => v.toFixed(2));
    bindRange('#sVal', '#vVal', (v) => v.toFixed(2));
    bindRange('#sDark', '#vDark', (v) => v.toFixed(2));
    bindRange('#sBright', '#vBright', (v) => v.toFixed(2));
    bindRange('#sNeutral', '#vNeutral', (v) => v.toFixed(2));
    bindRange('#sRealN', '#vRealN', IDV);
    bindRange('#sN', '#vN', IDV);
    bindRange('#sDdim', '#vDdim', IDV);
    bindRange('#sEta', '#vEta', (v) => v.toFixed(2));
    bindRange('#sK', '#vK', IDV);

    $$('#condMode button').forEach((b) => b.addEventListener('click', () => {
      $$('#condMode button').forEach((x) => x.classList.toggle('on', x === b));
      S.condMode = b.dataset.v;
      const real = S.condMode === 'real';
      const lot = S.condMode === 'lottery';
      $('#specBox').hidden = real || lot;
      $('#realBox').hidden = !real;
      $('#lotteryBox').hidden = !lot;
      // 抽奖模式的批量走它自己的下拉（这里的批量滑杆只服务另两个模式）
      $('#batchRow').hidden = lot;
      $('#condHint').textContent = real
        ? '从 val 集抽真实条件，每张条件都不同 → 看模型复现真实分布的能力。'
        : (lot ? '真实条件抽 N 张 → 评分 → 只交付分数最高的几张。'
               : '用下面的滑块拼一份条件向量，整批复制 → 看模型听不听指挥。');
      if (lot) updLotEst();
    }));

    const clampFinalK = () => {
      const lim = Math.min(+$('#selPool').value, +$('#selBatch').value) - 1;
      const el = $('#inFinalK');
      el.max = String(lim);
      let v = Math.max(1, Math.min(lim, Math.round(+el.value || 1)));
      if (+el.value !== v) el.value = String(v);
      return v;
    };
    async function updLotEst() {
      try {
        const p = new URLSearchParams({
          n_pool: $('#selPool').value, batch: $('#selBatch').value,
          ddim: $('#sDdim').value,
        });
        const r = await api('/api/lottery_estimate?' + p.toString());
        $('#lotEst').innerHTML =
          `预计耗时 <b>${r.seconds}s</b>（区间 ${r.seconds_low}~${r.seconds_high}s，`
          + `${r.per_image}s/张${r.calibrated_from_jobs ? '，按近期实测校准' : '，按基准常数估计'}）`
          + ` · 候选 ${$('#selPool').value} → 输出 ${clampFinalK()}`;
      } catch (e) {
        $('#lotEst').textContent = '预计耗时 —（服务暂不可达）';
      }
    }
    ['#selPool', '#selBatch', '#sDdim'].forEach(
      (id) => $(id).addEventListener('change', updLotEst));
    $('#inFinalK').addEventListener('change', clampFinalK);
    $('#cbToneLot').addEventListener('change', () => {
      $('#toneLotWrap').hidden = !$('#cbToneLot').checked;
    });

    $('#selQuant').addEventListener('change', syncQuant);
    $('#sPreq'); // no-op
    $('#btnGen').addEventListener('click', doGenerate);
    $('#btnSeed').addEventListener('click', () => {
      $('#inSeed').value = (Math.random() * 1e6) | 0;
    });
    $('#btnReal').addEventListener('click', runReal);
    $('#cbTone').addEventListener('change', runReal);
    $('#selTone').addEventListener('change', () => {
      if (S.condMode === 'real') runReal();
    });
    $('#btnReload').addEventListener('click', async () => {
      try {
        clearErr();
        const r = await api('/api/reload', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ckpt: $('#ckptSel').value }),
        });
        $('#brandSub').textContent = `已重载 ${r.loaded.rel} · ${r.loaded.params_m}M`;
      } catch (e) { reportErr('重载失败：' + e.message); }
    });
    $('#ckptSel').addEventListener('change', () => {
      $('#brandSub').textContent = '权重将在下次生成时载入：' + $('#ckptSel').value;
    });
    $('#btnTheme').addEventListener('click', () => {
      const cur = document.documentElement.dataset.theme;
      document.documentElement.dataset.theme = cur === 'dark' ? 'light' : 'dark';
    });
    $('#btnExport').addEventListener('click', async () => {
      // 抽奖模式只导出「最终输出」（top-K）；其余模式导出全部生成结果
      const src = (S.condMode === 'lottery')
        ? S.items.filter((i) => i.final_rank) : S.items;
      const keys = src.map((i) => i.key).join(',');
      if (!keys) return;
      const btn = $('#btnExport');
      btn.disabled = true;
      try {
        // fetch + blob 触发下载（与单张下载同一路径）；服务端同时留档，
        // 浏览器拦截下载时磁盘上也有副本（路径在 X-Saved-Path 头里）
        const r = await fetch('/api/export?keys=' + keys + '&layer=' + S.layer);
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.status);
        const saved = r.headers.get('X-Saved-Path');
        const b = await r.blob();
        const a = h('a', { href: URL.createObjectURL(b), download: S.layer === 'base' ? 'drmage_skins_base.zip' : 'drmage_skins.zip' });
        document.body.appendChild(a); a.click(); a.remove();
        $('#statLine').textContent += ` ｜ 已导出 ${src.length} 张`
          + (saved ? `（服务端留档：${saved}）` : '');
      } catch (e) {
        reportErr('导出失败：' + e.message);
      } finally {
        btn.disabled = false;
      }
    });
    $('#selLayer').addEventListener('change', (e) => setLayer(e.target.value));
    syncExportLabel();
    $('#btnAll3D').addEventListener('click', async (e) => {
      S.thumb3D = !S.thumb3D;
      e.target.textContent = S.thumb3D ? '换平面缩略图' : '换 3D 缩略图';
      await paintCells();
    });

    // modal
    $('#btnClose').addEventListener('click', closeModal);
    $('#modal').addEventListener('click', (e) => {
      if (e.target.id === 'modal') closeModal();
    });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
    $$('.mctl [data-view]').forEach((b) => b.addEventListener('click', () => {
      const v = b.dataset.view;
      const map = {
        front: [-0.42, 0.13], right: [-Math.PI / 2, 0.1], back: [Math.PI - 0.42, 0.13],
        left: [Math.PI / 2, 0.1], top: [-0.42, 1.25], iso: [-0.78, 0.42],
      };
      const m = map[v] || [0, 0];
      S.view.yaw = m[0]; S.view.pitch = m[1];
      $$('.mctl [data-view]').forEach((x) => x.classList.toggle('on', x === b));
      drawModal();
    }));
    $('#cbOv').addEventListener('change', (e) => {
      // 弹窗里这个复选框是 S.layer 的另一个视图，不是独立开关。
      // 从它进来时 setLayer 内部还会把 #cbOv.checked 再赋一次同一个值——
      // 给 checked 赋值不触发 change，所以不会来回弹。
      setLayer(e.target.checked ? 'all' : 'base');
      paintUV($('#cUV'), S.curImg, $('#cbGrid').checked);
    });
    $('#cbGrid').addEventListener('change', () => {
      if (S.curImg) paintUV($('#cUV'), S.curImg, $('#cbGrid').checked);
    });
    $('#cbAuto').addEventListener('change', (e) => {
      if (S.autoTimer) { clearInterval(S.autoTimer); S.autoTimer = null; }
      if (e.target.checked) {
        S.autoTimer = setInterval(() => {
          S.view.yaw += 0.045;
          drawModal();
        }, 40);
      }
    });
    $('#btnDl').addEventListener('click', async () => {
      if (!S.cur) return;
      // 注意作用域：这里只有 S.cur（openModal 里的 item/isLot 不在这个闭包里）——
      // 曾经直接抄弹窗里的变量名，点下载直接 ReferenceError，按钮静默失效。
      const cur = S.cur;
      const isLot = cur.origin === 'lottery_gen';
      const r = await fetch('/api/img?key=' + cur.key);
      const b = await r.blob();
      const a = h('a', {
        href: URL.createObjectURL(b),
        download: `drmage_${isLot ? 'lot' + cur.idx
        : (cur.real_index !== null && cur.real_index !== undefined ? 'real' + cur.real_index : 'gen' + cur.idx)}.png`,
      });
      document.body.appendChild(a); a.click(); a.remove();
    });
    $('#btnCopy').addEventListener('click', () => {
      const c = (S.cur && S.cur.cond) || {};
      navigator.clipboard && navigator.clipboard.writeText(JSON.stringify(c, null, 1));
      $('#btnCopy').textContent = '已复制';
      setTimeout(() => { $('#btnCopy').textContent = '复制条件'; }, 1200);
    });

    init3DInput();
    await runReal();
    // 开局先给一批，别让用户对着空页面（也验证整条链路）
    await doGenerate();
    window.__S = S;
    window.__ready = true;
  } catch (e) {
    reportErr('初始化失败：' + e.message);
    window.__ready = 'error';
  }
}

function syncQuant() {
  const q = S.quantOpts[+$('#selQuant').value] || { value: 'off' };
  const adaptive = q.value === 'adaptive';
  $('#sK').disabled = !adaptive;
  $('#sK').value = q.k || 48;
  $('#vK').textContent = $('#sK').value;
  $('#quantNote').innerHTML = q.note
    ? q.note + (q.value === 'off' ? '' : '<br>量化只在<b>可见像素</b>上做，透明区不动。')
    : '量化是<b>后处理</b>，不是模型结构。';
}

// 探针钩子
window.__render3D = (cv, img, opt) => R3D.render(cv, img, opt);
window.__raster = (cv, img, withOverlay) => raster(cv, img, withOverlay);

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
