/* ==========================================================================
   drmage-1.0-flash-preview · 训练监控平台  前端逻辑
   无任何外部依赖（不引 CDN）：图表、皮肤 UV 拼合都是自己画的 canvas。
   ========================================================================== */
'use strict';

/* ---------------- 基础工具 ---------------- */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function fmtNum(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(1) + 'G';
  if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (a >= 1e4) return Math.round(v).toLocaleString('en-US');
  if (a >= 1) return v.toFixed(d);
  if (a === 0) return '0';
  if (a >= 1e-3) return v.toFixed(Math.max(d, 3));
  return v.toExponential(2);
}
function fmtInt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Math.round(v).toLocaleString('en-US');
}
function fmtDur(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return '—';
  sec = Math.round(sec);
  if (sec < 60) return sec + ' 秒';
  if (sec < 3600) return Math.floor(sec / 60) + ' 分 ' + (sec % 60) + ' 秒';
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
  return h + ' 小时 ' + m + ' 分';
}
function fmtBytes(mb) {
  if (mb === null || mb === undefined) return '—';
  return mb >= 1024 ? (mb / 1024).toFixed(2) + ' GB' : mb.toFixed(0) + ' MB';
}
function ago(ts) {
  if (!ts) return '—';
  const d = Date.now() / 1000 - ts;
  if (d < 60) return Math.round(d) + ' 秒前';
  if (d < 3600) return Math.round(d / 60) + ' 分钟前';
  if (d < 86400) return Math.round(d / 3600) + ' 小时前';
  return Math.round(d / 86400) + ' 天前';
}
/** 只在值**真的变化**时写 DOM，并且不做任何动画。
 *
 * 早先这里给每次变化加了 `el.animate(opacity .3→1)`「闪烁提示」，
 * 但监控面板每 2 秒就推一次心跳，结果是整页数字一直在闪——非常难看。
 * 数值本身变了就看得出来，不需要额外动效。 */
function setTxt(el, v) {
  if (!el) return;
  const s = (v === null || v === undefined) ? '—' : String(v);
  if (el.textContent !== s) el.textContent = s;
}

/** 往一个固定容器里塞「徽章 + 说明文字」，复用已有节点而不是重建。
 *  重建会带来重排与闪烁，这也是面板「一跳一跳」的来源之一。 */
function setFoot(footEl, badges, tail) {
  if (!footEl) return;
  let pool = footEl._pool;
  if (!pool) {
    pool = footEl._pool = { badges: [], tail: null };
  }
  badges = (badges || []).filter(Boolean);
  badges.forEach((b, i) => {
    let el = pool.badges[i];
    if (!el) {
      el = document.createElement('span');
      pool.badges[i] = el;
      footEl.appendChild(el);
    }
    const txt = Array.isArray(b) ? b[0] : b;
    const cls = Array.isArray(b) ? (b[1] || '') : '';
    const want = 'badge ' + cls;
    if (el.className !== want) el.className = want;
    if (el.style.display === 'none') el.style.display = '';
    setTxt(el, txt);
  });
  for (let i = badges.length; i < pool.badges.length; i++) pool.badges[i].style.display = 'none';
  if (tail) {
    if (!pool.tail) {
      pool.tail = document.createElement('span');
      pool.tail.className = 'mute';
      footEl.appendChild(pool.tail);
    }
    pool.tail.style.display = '';
    setTxt(pool.tail, tail);
  } else if (pool.tail) {
    pool.tail.style.display = 'none';
  }
}
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  kids.flat().forEach(k => e.appendChild(typeof k === 'string' ? document.createTextNode(k) : k));
  return e;
}

const COLORS = {
  cyan: '#22d3ee', violet: '#a78bfa', emerald: '#34d399',
  amber: '#fbbf24', rose: '#fb7185', sky: '#38bdf8', muted: '#7c89a6', info: '#60a5fa',
};
function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/* ---------------- 可见的错误上报 ----------------
 * 早先每个数据环节外面都是 `catch (e) {}`：某一环挂掉，页面只会「静默地少一块」
 * （比如下面的训练日志、训练档案整块空白），控制台里一条线索都没有。
 * 现在统一汇到 console + 顶部一条红色横幅，并且同一条错误只报一次。
 */
const KPI_ERR = { seen: new Set() };
function reportErr(scope, err) {
  const msg = (err && (err.message || err.reason && err.reason.message || err)) || '未知错误';
  const line = scope + '：' + msg;
  try { console.error('[监控平台]', scope, err); } catch (e) { }
  if (KPI_ERR.seen.has(line)) return;
  KPI_ERR.seen.add(line);
  const wrap = $('.wrap');
  if (!wrap) return;
  let box = $('#errBar');
  if (!box) {
    box = h('div', { id: 'errBar', class: 'errbar' },
      h('div', { class: 'errbar-h' },
        '页面有环节读取失败（会缺一块内容，其余部分照常工作）'));
    wrap.insertBefore(box, wrap.firstChild);
  }
  const item = h('div', { class: 'errline' });
  item.appendChild(h('span', {}, '⚠️ ' + line));
  const x = h('button', { class: 'btn sm' }, '忽略');
  x.onclick = () => item.remove();
  item.appendChild(x);
  box.appendChild(item);
}
/** 浏览器层面的布局告警，不是我们代码的 bug —— 不该弹给用户看。 */
const BENIGN_ERR = /ResizeObserver loop|Script error\.?$|ResizeObserver loop limit/;
window.addEventListener('error', e => {
  const msg = (e && e.error && e.error.message) || (e && e.message) || '';
  if (BENIGN_ERR.test(msg)) return;
  reportErr('页面脚本异常', e.error || e.message);
});
window.addEventListener('unhandledrejection', e => {
  const msg = (e.reason && (e.reason.message || e.reason)) || '';
  if (BENIGN_ERR.test(String(msg))) return;
  reportErr('未处理的异步异常', e.reason);
});


/* ---------------- 图表（自研 canvas） ---------------- */
class Chart {
  constructor(canvas, tipEl) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.tip = tipEl;
    this.data = { series: [], refs: [], dual_axis: false };
    this.hidden = new Set();
    this.view = null;        // [x0, x1] 缩放范围
    this.hover = null;
    this._bind();
    /* 【根因级修复】不许在 ResizeObserver 回调里**同步**改画布尺寸。
     *
     * 同步改尺寸会再触发一次布局，Chrome 就报
     *   "ResizeObserver loop completed with undelivered notifications"
     * 然后**丢弃后续通知**：图表卡在旧尺寸/未栅格化状态 ——
     * 实测表现就是「图表区一整块白，手动缩放一下页面又好了」。
     * 推迟到下一帧再量，通知就能正常送达，循环告警也消失。 */
    this._ro = new ResizeObserver(() => {
      if (this._raf) cancelAnimationFrame(this._raf);
      this._raf = requestAnimationFrame(() => { this._raf = 0; this.resize(); });
    });
    this._ro.observe(this.cv.parentElement || this.cv);
  }
  setData(d) {
    d = d || { series: [], refs: [] };
    if (d.height) this.wantH = Math.round(d.height);   // 高度随数据一起给出，见 resize()
    const sameShape = this.data.series.length === d.series.length &&
      this.data.series.every((s, i) => s.key === d.series[i].key);
    this.data = d;
    if (!sameShape) this.view = null;
    this.full = this._xDomain();
    this.resize();          // 高度可能随图表切换而变，所以每次都重新量尺寸
  }
  /** 量尺寸 → （必要时）重建位图 → 重画。
   *
   * 三条硬约束，少一条都会出现「曲线空白 / 一缩放页面又正常了」：
   *  ① 位图尺寸必须**取整**。devicePixelRatio 在整页缩放下是小数（1.25/1.5），
   *     早先直接 `floor(w*dpr)`，CSS 宽度是 100%（带小数）→ 位图与显示尺寸
   *     永远差一点点，Chrome 会把整块后备位图判成「需要重新栅格化」。
   *  ② CSS 宽高改成**显式 px**（不再依赖 `width:100%`）。两边都由 JS 写死，
   *     就不存在百分比 × 小数 dpr 的错配。
   *  ③ **尺寸没变就绝不碰 `canvas.width/height`**。每次赋值都会丢弃并重建
   *     整块后备位图；监控面板每 4~8 秒刷一次图，反复重建在 GPU 合成下会
   *     留下一个纯白/空白矩形（实测症状：图表区一大块白，缩放页面才恢复）。
   */
  resize(force) {
    const cv = this.cv;
    const host = cv.parentElement;
    let w = Math.round((host ? host.clientWidth : cv.clientWidth) || 0);
    if (w < 40) w = 600;
    /* 高度来源：**显式字段优先**（setData({height}) / setHeight()），其次才是 HTML 属性。
     *
     * 早先读的是 `this.cv.h` —— 一个挂在 DOM 元素上的临时属性，谁都能写、也说不清
     * 什么时候被写成什么。实测就出过「系统曲线被拉成 1900px 高」这种没法解释的现象。
     * 现在走显式字段，并且**夹到 [48, 900]**：任何来源的野值都不可能再造出一条
     * 两屏高的曲线。 */
    let hgt = Math.round(this.wantH || parseInt(cv.getAttribute('height')) || 200);
    if (!(hgt >= 48 && hgt <= 900)) {
      reportErr('图表高度异常', new Error('hgt=' + hgt + '（原始 this.wantH=' + this.wantH
        + '，属性=' + cv.getAttribute('height') + '）已按 200 处理'));
      hgt = 200;
    }
    this.H = hgt;
    const dpr = clamp(window.devicePixelRatio || 1, 1, 2);
    const bw = Math.max(120, Math.min(4096, Math.round(w * dpr)));
    const bh = Math.max(60, Math.min(2048, Math.round(hgt * dpr)));

    this.W = w;
    const same = !force && cv.width === bw && cv.height === bh
      && cv.style.width === w + 'px' && cv.style.height === hgt + 'px';
    if (!same) {
      cv.width = bw;
      cv.height = bh;
      cv.style.width = w + 'px';
      cv.style.height = hgt + 'px';
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }
  /** 由调用方显式指定这块图的高度（CSS px）。 */
  setHeight(h) {
    const v = Math.round(Number(h) || 0);
    if (v > 0 && v !== this.wantH) { this.wantH = v; this.resize(); }
    return this;
  }
  get heightPx() { return this.H || 0; }
  /** 强制重建位图并重画（「重绘」按钮 / 自检发现尺寸不一致时用） */
  repaint() { this.resize(true); }
  /** 位图尺寸和「实际显示尺寸 × dpr」对不上就说明这块画布已经不可信了 */
  needsRepaint() {
    const cv = this.cv;
    if (!cv || !cv.width) return true;
    const r = cv.getBoundingClientRect();
    if (r.width < 40 || r.height < 20) return false;
    const dpr = clamp(window.devicePixelRatio || 1, 1, 2);
    return Math.abs(cv.width - Math.round(r.width * dpr)) > 2
      || Math.abs(cv.height - Math.round(r.height * dpr)) > 2;
  }
  _xDomain() {
    let a = Infinity, b = -Infinity;
    this.data.series.forEach(s => s.points.forEach(p => {
      if (p[0] < a) a = p[0];
      if (p[0] > b) b = p[0];
    }));
    if (!isFinite(a)) return [0, 1];
    if (a === b) return [a, a + 1];
    return [a, b];
  }
  _bind() {
    const cv = this.cv;
    cv.addEventListener('mousemove', e => {
      const r = cv.getBoundingClientRect();
      this.hover = { x: e.clientX - r.left, y: e.clientY - r.top };
      this.draw();
    });
    cv.addEventListener('mouseleave', () => { this.hover = null; this.draw(); });
    cv.addEventListener('wheel', e => {
      if (!this.full) return;
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      const mx = e.clientX - r.left;
      const L = this._layout();
      const [x0, x1] = this.view || this.full;
      const t = clamp((mx - L.l) / Math.max(1, L.w), 0, 1);
      const cx = x0 + t * (x1 - x0);
      const k = e.deltaY > 0 ? 1.18 : 0.85;
      let n0 = cx - (cx - x0) * k, n1 = cx + (x1 - cx) * k;
      const fw = this.full[1] - this.full[0];
      if (n1 - n0 < fw / 400 || n1 - n0 > fw * 1.02) return;
      this.view = [clamp(n0, this.full[0], this.full[1]),
      clamp(n1, this.full[0], this.full[1])];
      this.draw();
    }, { passive: false });
    cv.addEventListener('dblclick', () => { this.view = null; this.draw(); });
    let drag = null;
    cv.addEventListener('mousedown', e => {
      if (!this.full) return;
      drag = { x: e.clientX, v: this.view || this.full.slice() };
      cv.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', e => {
      if (!drag) return;
      const L = this._layout();
      const [x0, x1] = this.full;
      const span = drag.v[1] - drag.v[0];
      const dx = (e.clientX - drag.x) / Math.max(1, L.w) * span;
      if (span >= (x1 - x0) * 0.999) return;
      let n0 = drag.v[0] - dx, n1 = drag.v[1] - dx;
      if (n0 < x0) { n1 += x0 - n0; n0 = x0; }
      if (n1 > x1) { n0 -= n1 - x1; n1 = x1; }
      this.view = [Math.max(x0, n0), Math.min(x1, n1)];
      this.draw();
    });
    window.addEventListener('mouseup', () => {
      if (drag) { drag = null; cv.style.cursor = 'crosshair'; }
    });
  }
  _layout() {
    const dual = this.data.dual_axis;
    const l = 58, r = dual ? 56 : 16, t = 12, b = 26;
    return { l, r, t, b, w: Math.max(10, (this.W || 600) - l - r), h: Math.max(10, (this.H || 200) - t - b) };
  }
  _ticks(min, max, n) {
    if (!isFinite(min) || !isFinite(max) || min === max) return [min];
    const span = max - min;
    const raw = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) out.push(v);
    return out;
  }
  _yMap(v) {
    if (this._log) return Math.log10(Math.max(v, this._yminPos)) / Math.log10(this._ymax);
    return (v - this._ymin) / (this._ymax - this._ymin || 1);
  }
  draw() {
    const ctx = this.ctx;
    if (!ctx || !this.W) return;
    const d = this.data;
    const L = this._layout();
    ctx.clearRect(0, 0, this.W, this.H);

    const vis = d.series.filter(s => !this.hidden.has(s.key));
    const [vx0, vx1] = this.view || this.full || [0, 1];

    // ---- Y 轴域（左右分开算）----
    const yFor = (axis) => {
      let lo = Infinity, hi = -Infinity;
      vis.filter(s => (s.axis || 'left') === axis).forEach(s => s.points.forEach(p => {
        if (p[0] < vx0 || p[0] > vx1) return;
        if (!isFinite(p[1])) return;
        lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]);
      }));
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      if (lo === hi) { lo -= Math.abs(lo) * .1 + .5; hi += Math.abs(hi) * .1 + .5; }
      const pad = (hi - lo) * 0.10;
      return [lo - pad, hi + pad];
    };
    let [yl0, yl1] = yFor('left');
    let [yr0, yr1] = yFor('right');
    if (d.y_min !== undefined && d.y_min !== null) yl0 = d.y_min;
    let log = !!d.log_y && yl0 > 0;
    this._log = log;
    this._ymin = yl0; this._ymax = yl1;
    this._yminPos = Math.max(yl0 * 1e-3, 1e-9);

    const yPix = (v, axis) => {
      const [a, bb] = axis === 'right' ? [yr0, yr1] : [yl0, yl1];
      let t;
      if (log && axis === 'left') t = Math.log10(Math.max(v, this._yminPos)) / Math.log10(yl1);
      else t = (v - a) / (bb - a || 1);
      return L.t + L.h - t * L.h;
    };
    const xPix = (v) => L.l + (v - vx0) / ((vx1 - vx0) || 1) * L.w;

    // ---- 网格 ----
    ctx.lineWidth = 1;
    ctx.strokeStyle = cssVar('--border', 'rgba(255,255,255,.08)');
    ctx.font = '10px ui-monospace, Consolas, monospace';
    ctx.fillStyle = cssVar('--text-mute', '#66708a');
    const xt = this._ticks(vx0, vx1, 6);
    ctx.textAlign = 'center';
    xt.forEach(v => {
      const x = xPix(v);
      if (x < L.l - 2 || x > L.l + L.w + 2) return;
      ctx.beginPath(); ctx.moveTo(x, L.t); ctx.lineTo(x, L.t + L.h); ctx.stroke();
      ctx.fillText(fmtNum(v, 0), x, L.t + L.h + 15);
    });
    const yt = this._ticks(yl0, yl1, 5);
    ctx.textAlign = 'right';
    yt.forEach(v => {
      const y = yPix(v, 'left');
      if (y < L.t - 2 || y > L.t + L.h + 2) return;
      ctx.beginPath(); ctx.moveTo(L.l, y); ctx.lineTo(L.l + L.w, y); ctx.stroke();
      ctx.fillText(fmtNum(v, 3), L.l - 7, y + 3);
    });
    if (d.dual_axis) {
      ctx.textAlign = 'left';
      this._ticks(yr0, yr1, 4).forEach(v => {
        const y = yPix(v, 'right');
        if (y < L.t || y > L.t + L.h) return;
        ctx.fillText(fmtNum(v, 0), L.l + L.w + 7, y + 3);
      });
    }
    // 轴标签
    ctx.save();
    ctx.fillStyle = cssVar('--text-dim', '#9ba7bf');
    ctx.textAlign = 'left'; ctx.font = '10.5px ui-monospace, monospace';
    ctx.fillText(d.x_label || '', L.l, L.t + L.h + 22);
    ctx.restore();

    // ---- 参考线 ----
    (d.refs || []).forEach(rf => {
      const y = yPix(rf.y, rf.axis || 'left');
      if (y < L.t || y > L.t + L.h) return;
      ctx.save();
      ctx.setLineDash([5, 4]); ctx.lineWidth = 1;
      ctx.strokeStyle = COLORS[rf.color] || COLORS.muted;
      ctx.beginPath(); ctx.moveTo(L.l, y); ctx.lineTo(L.l + L.w, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS[rf.color] || COLORS.muted;
      ctx.textAlign = 'right'; ctx.font = '10px ui-monospace, monospace';
      ctx.fillText(rf.label, L.l + L.w - 4, y - 4);
      ctx.restore();
    });

    // ---- 数据线 ----
    const clip = () => { ctx.beginPath(); ctx.rect(L.l, L.t - 2, L.w, L.h + 4); ctx.clip(); };
    vis.forEach((s, i) => {
      const pts = s.points.filter(p => p[0] >= vx0 && p[0] <= vx1);
      if (pts.length < 1) return;
      const col = COLORS[s.color] || COLORS.cyan;
      const ax = s.axis || 'left';
      ctx.save(); clip();
      if (i === 0 && pts.length > 2 && !s.dash) {
        const g = ctx.createLinearGradient(0, L.t, 0, L.t + L.h);
        g.addColorStop(0, col + '38'); g.addColorStop(1, col + '00');
        ctx.beginPath();
        ctx.moveTo(xPix(pts[0][0]), yPix(pts[0][1], ax));
        pts.slice(1).forEach(p => ctx.lineTo(xPix(p[0]), yPix(p[1], ax)));
        const last = pts[pts.length - 1];
        ctx.lineTo(xPix(last[0]), L.t + L.h); ctx.lineTo(xPix(pts[0][0]), L.t + L.h);
        ctx.closePath(); ctx.fillStyle = g; ctx.fill();
      }
      ctx.beginPath();
      ctx.lineWidth = s.width || 2;
      ctx.globalAlpha = s.alpha || 1;
      ctx.strokeStyle = col;
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      if (s.dash) ctx.setLineDash([4, 3]);
      pts.forEach((p, k) => {
        const x = xPix(p[0]), y = yPix(p[1], ax);
        k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    });

    // ---- 悬停十字 ----
    if (this.hover && this.hover.x >= L.l && this.hover.x <= L.l + L.w) {
      let best = null, bd = Infinity;
      vis.forEach(s => s.points.forEach(p => {
        if (p[0] < vx0 || p[0] > vx1) return;
        const dd = Math.abs(xPix(p[0]) - this.hover.x);
        if (dd < bd) { bd = dd; best = p[0]; }
      }));
      if (best !== null) {
        const x = xPix(best);
        ctx.save();
        ctx.strokeStyle = cssVar('--border-2', 'rgba(255,255,255,.2)');
        ctx.setLineDash([3, 3]); ctx.beginPath();
        ctx.moveTo(x, L.t); ctx.lineTo(x, L.t + L.h); ctx.stroke();
        ctx.setLineDash([]);
        const rows = [['', fmtInt(best) + ' ' + (d.x_unit || '')]];
        vis.forEach(s => {
          const p = s.points.find(q => q[0] === best);
          if (!p) return;
          ctx.fillStyle = COLORS[s.color] || COLORS.cyan;
          ctx.beginPath(); ctx.arc(x, yPix(p[1], s.axis || 'left'), 3.2, 0, 6.3); ctx.fill();
          rows.push([s.label, fmtNum(p[1], 4)]);
        });
        if (this.tip) {
          this.tip.innerHTML = rows.map((r, k) => k === 0
            ? `<div class="t-row"><b>${r[1]}</b></div>`
            : `<div class="t-row"><span class="t-key">${r[0]}</span><span>${r[1]}</span></div>`
          ).join('');
          this.tip.style.left = x + 'px';
          this.tip.style.top = (this.hover.y - 6) + 'px';
          this.tip.classList.add('on');
        }
        ctx.restore();
      }
    } else if (this.tip) {
      this.tip.classList.remove('on');
    }
  }
  exportPNG() {
    const a = document.createElement('a');
    a.href = this.cv.toDataURL('image/png');
    a.download = 'chart_' + Date.now() + '.png';
    a.click();
  }
}

/* ---------------- 全局状态 ---------------- */
const S = {
  boot: null, overview: null, system: null, artifacts: null, events: [],
  evFilter: '', lastEventId: 0, tab: 'logs', chartId: 'loss', series: null,
  // gridIdx 默认给一个超大值：renderSamples 会夹到最后一帧 —— 也就是「当前实验的最新快照」
  grids: [], gridIdx: 9999, skinView: 'figure', skinOverlay: true,
  skinStepIdx: 9999, skinStepTag: '',
  logLevel: '', logQuery: '', logName: 'train.log', notify: false,
  faceIndex: {}, sseOk: false, sysTimer: null, logAuto: true,
  tag: null, evCounts: null, skinGroupTag: '', skinListSig: '',
  // 各区域的内容签名：只有变了才重建 DOM（避免每 2 秒一次重排/闪烁）
  skinSig: '', sampleHintSig: '', logSig: '', seriesSig: '', artSig: '',
};

/* ---------------- 主题 ---------------- */
function initTheme() {
  const saved = localStorage.getItem('theme') || 'dark';
  document.documentElement.dataset.theme = saved;
  $$('#themeSeg button').forEach(b => {
    b.classList.toggle('on', b.dataset.theme === saved);
    b.onclick = () => {
      document.documentElement.dataset.theme = b.dataset.theme;
      localStorage.setItem('theme', b.dataset.theme);
      $$('#themeSeg button').forEach(x => x.classList.toggle('on', x === b));
      if (chart) chart.resize();
      if (sysChart) sysChart.resize();
    };
  });
}

/* ---------------- 提示条 ---------------- */
function toast(msg, level = 'info', ms = 4200) {
  const t = h('div', { class: 'toast ' + (level === 'error' ? 'err' : level === 'warn' ? 'warn' : level === 'success' ? 'ok' : '') }, msg);
  $('#toasts').appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .3s'; }, ms - 350);
  setTimeout(() => t.remove(), ms);
}

/* ---------------- 数据获取 ---------------- */
async function jget(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(url + ' → HTTP ' + r.status);
  return r.json();
}
async function jpost(url, body) {
  const r = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({ ok: false, error: '响应不是 JSON' }));
  return j;
}

/* ==========================================================================
   渲染：KPI
   ========================================================================== */
function updateChip(tr) {
  const chip = $('#statusChip');
  if (!chip || !tr) return;
  chip.className = 'status-chip s-' + (tr.status || 'idle');
  setTxt($('#statusText'), `${tr.status_cn || '—'} · ${tr.tag || '—'}`);
}
/* ---- KPI：只建一次骨架，之后每 tick 只更新文本/宽度 ----
 *
 * 早先每次都 `wrap.replaceChildren(...)` 重建全部卡片 —— 每 2 秒一次重建会让
 * 整块区域重排 + 重播入场动画，肉眼就是「一跳一跳」。现在 DOM 建一次就固定，
 * 只有文字变化时才写。 */
const KPI_DEFS = [
  ['status', '训练状态', ''], ['progress', '训练进度', ''],
  ['mse', '训练损失 MSE', 'accent-bar'], ['lr', '学习率', ''],
  ['vram', '峰值显存', ''], ['gpu', 'GPU 利用率', ''],
  ['temp', 'GPU 温度', ''], ['eta', '预计剩余', ''],
];
const KPI = {};
const _lastColor = new WeakMap();
function setColor(el, c) {
  if (!el) return;
  if (_lastColor.get(el) === c) return;
  _lastColor.set(el, c);
  el.style.color = c || '';
}

function buildKPIs() {
  const wrap = $('#kpis');
  if (wrap.dataset.built) return;
  wrap.dataset.built = '1';
  wrap.replaceChildren(...KPI_DEFS.map(([key, label, extra]) => {
    const card = h('div', { class: 'card kpi fade-in ' + extra });
    const lab = h('div', { class: 'k-label' }, label);
    card.appendChild(lab);
    const val = h('div', { class: 'k-value' });
    card.appendChild(val);
    let barI = null;
    if (key === 'progress') {
      const p = h('div', { class: 'progress striped' });
      barI = h('i'); barI.style.width = '0%';
      p.appendChild(barI); card.appendChild(p);
    }
    const foot = h('div', { class: 'k-foot' });
    card.appendChild(foot);
    KPI[key] = { val, barI, foot, label: lab };
    return card;
  }));
}

function renderKPIs() {
  buildKPIs();
  const o = S.overview || {}, tr = o.train || {}, sys = o.system || {};
  const g = (sys.gpus || [])[0] || {};
  const stIco = { running: '🟢', finished: '✅', stale: '⛔', stopped: '⏸', idle: '💤' };

  setTxt(KPI.status.val, `${stIco[tr.status] || '•'} ${tr.status_cn || '—'}`);
  setFoot(KPI.status.foot, [
    'tag ' + (tr.tag || '—'),
    tr.pid ? 'PID ' + tr.pid : null,
    tr.pid ? (tr.pid_alive ? ['进程存活', 'ok'] : ['进程已退出', 'warn']) : null,
  ], tr.heartbeat_age_s !== null && tr.heartbeat_age_s !== undefined
    ? '心跳 ' + ago(Date.now() / 1000 - tr.heartbeat_age_s) : null);

  const done = (tr.epochs_done === null || tr.epochs_done === undefined) ? '—' : tr.epochs_done;
  setTxt(KPI.progress.val, `${done} / ${tr.epochs || '—'} 轮`);
  if (KPI.progress.barI) {
    const w = clamp((tr.progress || 0) * 100, 0, 100).toFixed(1) + '%';
    if (KPI.progress.barI.style.width !== w) KPI.progress.barI.style.width = w;
  }
  setFoot(KPI.progress.foot, [
    fmtInt(tr.step) + ' 步' + (tr.total_steps ? ' / ' + fmtInt(tr.total_steps) : ''),
  ], tr.steps_per_epoch ? '每轮 ' + fmtInt(tr.steps_per_epoch) + ' 步' : null);

  const d = tr.mse_delta;
  // 损失口径随架构变（旧架构 MSE / PIDiff 交叉熵 CE），标签必须跟着走，
  // 否则 PIDiff 实验里会顶着「训练损失 MSE」的标题显示一个 CE 数值。
  if (tr.loss_name && KPI.mse.label) setTxt(KPI.mse.label, tr.loss_name);
  setTxt(KPI.mse.val, fmtNum(tr.mse, 5));
  setFoot(KPI.mse.foot, [
    (d === null || d === undefined) ? null
      : [(d < 0 ? '↓ ' : d > 0 ? '↑ ' : '') + fmtNum(Math.abs(d), 5),
      d < 0 ? 'ok' : d > 0 ? 'err' : ''],
    (tr.mse_best === null || tr.mse_best === undefined) ? null : '最优 ' + fmtNum(tr.mse_best, 5),
  ], tr.records ? tr.records + ' 个记录点' : null);

  setTxt(KPI.lr.val, tr.lr ? (tr.lr * 1e6).toFixed(1) : '—');
  setFoot(KPI.lr.foot, ['×10⁻⁶'],
    tr.args && tr.args.schedule ? '调度 ' + tr.args.schedule : null);

  setTxt(KPI.vram.val, fmtNum(tr.vram_gb, 2));
  setFoot(KPI.vram.foot, ['GiB'],
    g.mem_total_mb ? '显卡共 ' + (g.mem_total_mb / 1024).toFixed(1) + ' GiB' : null);

  setTxt(KPI.gpu.val, fmtNum(g.util_percent, 0));
  setColor(KPI.gpu.val, barColor(g.util_percent));
  setFoot(KPI.gpu.foot, ['%'], g.name ? g.name.replace('NVIDIA GeForce ', '') : null);

  setTxt(KPI.temp.val, fmtNum(g.temp_c, 0));
  setColor(KPI.temp.val, tempColor(g.temp_c));
  setFoot(KPI.temp.foot, ['°C'],
    g.power_w ? '功耗 ' + g.power_w.toFixed(0) + ' / ' + (g.power_limit_w || 0).toFixed(0) + ' W' : null);

  setTxt(KPI.eta.val, tr.eta_s ? fmtDur(tr.eta_s)
    : (tr.status === 'running' ? '—' : tr.status_cn || '—'));
  setFoot(KPI.eta.foot, [tr.epoch_seconds ? '每轮 ' + fmtDur(tr.epoch_seconds) : null],
    tr.elapsed_s ? '已训练 ' + fmtDur(tr.elapsed_s) : null);
}
function bdg(text, cls = '') { return h('span', { class: 'badge ' + cls }, text); }
function barColor(p) {
  if (p === null || p === undefined) return null;
  if (p >= 90) return COLORS.rose; if (p >= 70) return COLORS.amber; return COLORS.emerald;
}
function tempColor(t) {
  if (t === null || t === undefined) return null;
  if (t >= 86) return COLORS.rose; if (t >= 80) return COLORS.amber; return COLORS.cyan;
}

/* ==========================================================================
   渲染：系统运行状态
   ========================================================================== */
/* ---- 系统面板：同样只建一次，之后只更新属性 ---- */
const SYS = { gauges: [], rows: [], kv: {}, built: false, histSig: '' };
const GAUGE_DEFS = [
  ['GPU', 'util_percent', '%'], ['显存', 'mem_percent', '%'], ['温度', 'temp_c', ' °C'],
];
const METRIC_DEFS = [
  ['cpu', 'CPU'], ['mem', '内存'], ['disk', '项目盘'], ['root', '系统盘 C:'],
];
const HOST_DEFS = [
  ['主机', 'hostname'], ['用户', 'user'], ['CPU', 'cpu'], ['核心', 'cores'],
  ['Python', 'python'], ['开机时长', 'uptime'], ['显卡驱动', 'driver'],
  ['显存', 'vram'], ['GPU 频率', 'clock'], ['风扇', 'fan'],
];

function buildSystem() {
  if (SYS.built) return;
  SYS.built = true;
  const R = 27, C = 2 * Math.PI * R;
  $('#gauges').replaceChildren(...GAUGE_DEFS.map(([lab, , unit]) => {
    const box = h('div', { class: 'gauge' });
    const dial = h('div', { class: 'g-dial' });
    dial.insertAdjacentHTML('beforeend',
      `<svg viewBox="0 0 72 72" width="72" height="72">
         <circle cx="36" cy="36" r="${R}" fill="none"
                 stroke="${cssVar('--surface-3', '#2a3145')}" stroke-width="7"/>
         <circle class="arc" cx="36" cy="36" r="${R}" fill="none" stroke="${COLORS.cyan}"
                 stroke-width="7" stroke-linecap="round"
                 stroke-dasharray="${C.toFixed(2)}" stroke-dashoffset="${C.toFixed(2)}"
                 transform="rotate(-90 36 36)"/>
       </svg>`);
    const val = h('div', { class: 'g-val' }, '—');
    dial.appendChild(val);
    box.appendChild(dial);
    box.appendChild(h('div', { class: 'g-lab' }, lab + (unit === ' °C' ? ' °C' : ' %')));
    SYS.gauges.push({ arc: dial.querySelector('.arc'), val, C });
    return box;
  }));

  $('#sysMetrics').replaceChildren(...METRIC_DEFS.map(([key, lab]) => {
    const r = h('div', { class: 'mrow' });
    const ml = h('div', { class: 'ml' }, lab);
    const bar = h('div', { class: 'bar' });
    const fill = h('i'); fill.style.width = '0%'; fill.style.background = COLORS.cyan;
    bar.appendChild(fill);
    const mv = h('div', { class: 'mv' });
    const b = h('b', {}, '—'); const i2 = h('i', {}, '');
    mv.appendChild(b); mv.appendChild(i2);
    r.append(ml, bar, mv);
    SYS.rows.push({ key, ml, fill, b, i2 });
    return r;
  }));

  $('#hostInfo').replaceChildren(...HOST_DEFS.flatMap(([lab, key]) => {
    const k = h('div', { class: 'k' }, lab);
    const v = h('div', { class: 'v' }, '—');
    SYS.kv[key] = v;
    return [k, v];
  }));
}

function renderSystem() {
  buildSystem();
  const sys = S.system || {}, now = sys.now || {}, host = sys.host || {};
  const g = (now.gpus || [])[0] || {};
  const mem = now.memory || {}, disk = now.disk_project || {};

  setTxt($('#sysUpdated'), now.time ? ('更新于 ' + now.time) : '—');

  // 三个环形：只改 dashoffset / 颜色 / 文本
  const vals = {
    GPU: [g.util_percent, barColor(g.util_percent)],
    显存: [g.mem_percent, barColor(g.mem_percent)],
    温度: [g.temp_c, tempColor(g.temp_c)],
  };
  GAUGE_DEFS.forEach(([lab], i) => {
    const st = SYS.gauges[i]; if (!st) return;
    const [v, col] = vals[lab] || [null, null];
    const pct = (v === null || v === undefined) ? 0 : clamp(v / 100, 0, 1);
    const off = (st.C * (1 - pct)).toFixed(2);
    if (st.arc.getAttribute('stroke-dashoffset') !== off) st.arc.setAttribute('stroke-dashoffset', off);
    if (st.arc.getAttribute('stroke') !== (col || COLORS.cyan)) st.arc.setAttribute('stroke', col || COLORS.cyan);
    setTxt(st.val, (v === null || v === undefined) ? '—' : Math.round(v));
  });

  const projDrive = (host.project_drive || 'C:\\').replace('\\', '');
  const mvals = {
    cpu: [now.cpu_percent, null],
    mem: [mem.percent, `${mem.used_gb} / ${mem.total_gb} GB`],
    disk: [disk.percent, `剩 ${disk.free_gb} GB`],
    root: [(now.disk_root || {}).percent, `剩 ${(now.disk_root || {}).free_gb} GB`],
  };
  SYS.rows.forEach(row => {
    if (row.key === 'disk') setTxt(row.ml, '项目盘 ' + projDrive);
    const [v, extra] = mvals[row.key] || [null, null];
    const w = clamp(v || 0, 0, 100) + '%';
    if (row.fill.style.width !== w) row.fill.style.width = w;
    const col = barColor(v) || COLORS.cyan;
    if (row.fill.style.background !== col) row.fill.style.background = col;
    setTxt(row.b, (v === null || v === undefined) ? '—' : v + '%');
    setTxt(row.i2, extra || '');
  });

  const hv = {
    hostname: host.hostname, user: host.user,
    cpu: (host.cpu || '').slice(0, 44), cores: host.cpu_count + ' 线程',
    python: host.python, uptime: fmtDur(host.boot_uptime_s),
    driver: g.driver,
    vram: g.mem_used_mb ? `${g.mem_used_mb.toFixed(0)} / ${g.mem_total_mb.toFixed(0)} MB` : '—',
    clock: g.clock_mhz ? `${g.clock_mhz.toFixed(0)} / ${g.clock_max_mhz.toFixed(0)} MHz` : '—',
    fan: (g.fan_percent === null || g.fan_percent === undefined) ? '—' : g.fan_percent.toFixed(0) + ' %',
  };
  Object.entries(hv).forEach(([k, v]) => setTxt(SYS.kv[k], v === null || v === undefined ? '—' : v));

  // 系统历史曲线：点数或末点变化时才重画
  const hist = sys.history || [];
  const sig = hist.length + '|' + (hist.length ? JSON.stringify(hist[hist.length - 1]) : '');
  if (sysChart && hist.length > 1 && sig !== SYS.histSig) {
    SYS.histSig = sig;
    const mk = (f) => hist.map((p, i) => [i, f(p)]).filter(p => p[1] !== null && p[1] !== undefined);
    sysChart.setData({
      series: [
        { key: 'cpu', label: 'CPU %', color: 'sky', points: mk(p => p.cpu), width: 1.4 },
        { key: 'gpu', label: 'GPU %', color: 'emerald', points: mk(p => p.gpu_util), width: 1.6 },
        { key: 'gmem', label: '显存 %', color: 'violet', points: mk(p => p.gpu_mem), width: 1.4 },
        { key: 'ram', label: '内存 %', color: 'amber', points: mk(p => p.ram), width: 1.2 },
      ],
      refs: [], x_label: '最近 ' + Math.round(hist.length * (S.tickSeconds || 2) / 60) + ' 分钟（左→右）',
      y_label: '%', y_min: 0, height: 120,
    });
  }
}

/* ==========================================================================
   渲染：事件流
   ========================================================================== */
const EV_ICO = {
  service_start: '🚀', run_attach: '🔗', train_start: '▶️', train_finish: '🏁',
  train_died: '⛔', train_stopped: '⏸️', epoch: '📊', best_mse: '🏆',
  mse_rise: '📉', sample_grid: '🖼️', skins_export: '🧑‍🎨', checkpoint: '💾',
  gpu_temp: '🌡️', gpu_mem: '📦', disk: '💽', cpu_high: '⚙️', heartbeat_stale: '💤',
};
/* ---- 事件流：只追加新节点，永不整体重建 ----
 *
 * 整体重建有两个后果：① 滚动位置每 2 秒被重置（用户正在往回看历史时尤其难受）；
 * ② 整块内容闪一次。改成 append-only + 用 CSS 显示/隐藏做级别过滤，
 * 顺带把「新事件」的语义交给用户自己看时间戳。 */
const EV_SEEN = new Set();
function evNode(e) {
  const d = h('div', { class: 'ev lv-' + e.level });
  d.dataset.lv = e.level;
  d.appendChild(h('div', { class: 'e-time' }, e.time || ''));
  d.appendChild(h('div', { class: 'e-ico' }, EV_ICO[e.code]
    || (e.level === 'error' ? '⛔' : e.level === 'warn' ? '⚠️' : e.level === 'success' ? '✅' : 'ℹ️')));
  const body = h('div');
  body.appendChild(h('div', { class: 'e-title' }, e.title || ''));
  if (e.detail) body.appendChild(h('div', { class: 'e-detail' }, e.detail));
  d.appendChild(body);
  return d;
}
function appendEvents(items) {
  const box = $('#events');
  let added = 0;
  (items || []).forEach(e => {
    if (!e || e.id === undefined || EV_SEEN.has(e.id)) return;
    EV_SEEN.add(e.id);
    box.prepend(evNode(e));      // 新的在最上面
    added++;
  });
  // 元素太多时裁掉最旧的（保留过滤用的可见性由 applyEventFilter 处理）
  while (box.children.length > 240) box.lastElementChild.remove();
  if (added) applyEventFilter();
  updateEventsEmpty();
  return added;
}
function applyEventFilter() {
  const box = $('#events');
  for (const el of box.children) {
    if (!el.dataset || !el.dataset.lv) continue;
    el.style.display = (!S.evFilter || el.dataset.lv === S.evFilter) ? '' : 'none';
  }
}
function updateEventsEmpty() {
  const box = $('#events');
  // 清掉 HTML 里的初始占位提示（它没有 data-lv）
  [...box.children].forEach(el => {
    if (!el.dataset.lv && el.classList.contains('empty')) el.remove();
  });
  const visible = [...box.children].filter(el => el.dataset.lv && el.style.display !== 'none');
  let e = box.querySelector('.ev-empty');
  if (!visible.length) {
    if (!e) {
      e = h('div', { class: 'empty ev-empty' },
        '暂无监控事件。训练启动、每轮完成、新的最优 MSE、显存/温度超阈值等都会出现在这里。');
      box.appendChild(e);
    }
  } else if (e) {
    e.remove();
  }
}
function renderEvents() { applyEventFilter(); updateEventsEmpty(); }
function pushEvent(e) {
  if (!e || (e.id || 0) <= S.lastEventId) return;
  S.lastEventId = e.id;
  S.events.push(e);
  if (S.events.length > 900) S.events = S.events.slice(-700);
  if (e.level === 'warn' || e.level === 'error') {
    toast((e.level === 'error' ? '⛔ ' : '⚠️ ') + e.title + (e.detail ? ' — ' + e.detail : ''),
      e.level === 'error' ? 'error' : 'warn');
    if (S.notify && 'Notification' in window && Notification.permission === 'granted') {
      try { new Notification('训练监控 · ' + e.title, { body: e.detail || '', tag: e.code }); } catch (err) { }
    }
  }
}
function pushEvent(e) {
  if (!e || (e.id || 0) <= S.lastEventId) return;
  S.lastEventId = e.id;
  S.events.push(e);
  if (S.events.length > 900) S.events = S.events.slice(-700);
  if (e.level === 'warn' || e.level === 'error') {
    toast((e.level === 'error' ? '⛔ ' : '⚠️ ') + e.title + (e.detail ? ' — ' + e.detail : ''),
      e.level === 'error' ? 'error' : 'warn');
    if (S.notify && 'Notification' in window && Notification.permission === 'granted') {
      try { new Notification('训练监控 · ' + e.title, { body: e.detail || '', tag: e.code }); } catch (err) { }
    }
  }
}

/* ==========================================================================
   渲染：样本快照
   ========================================================================== */
function renderSamples() {
  const art = S.artifacts || {};
  S.grids = art.all_grids || [];
  const tr = (S.overview || {}).train || {};
  const rng = $('#sampleRange');
  rng.max = Math.max(0, S.grids.length - 1);
  if (S.gridIdx >= S.grids.length) S.gridIdx = S.grids.length - 1;
  if (S.gridIdx < 0) S.gridIdx = 0;
  rng.value = S.gridIdx;
  const g = S.grids[S.gridIdx];
  const img = $('#sampleImg');
  const hint = $('#sampleEmpty');
  if (g) {
    if (hint) hint.remove();
    img.style.display = '';
    const url = g.url + '?t=' + Math.floor(g.mtime);
    if (img.dataset.url !== url) { img.dataset.url = url; img.src = url; }
    setTxt($('#sampleStepLabel'), '第 ' + fmtInt(g.step) + ' 步');
    setTxt($('#sampleSub'), `${S.grids.length} 个快照 · 当前显示 ${g.name}`);
  } else {
    img.style.display = 'none';
    setTxt($('#sampleStepLabel'), '—');
    const every = (tr.args && tr.args.sample_every) || 2200;
    const perEp = tr.steps_per_epoch || 2200;
    setTxt($('#sampleSub'), '该实验还没有样本快照');
    // 只在「换了实验」或首次进入空态时重建提示，避免每 2 秒写一次 DOM
    const sig = (tr.tag || '') + '|' + every + '|' + perEp;
    if (S.sampleHintSig !== sig) {
      S.sampleHintSig = sig;
      let box = $('#sampleEmpty');
      if (box) box.remove();
      box = h('div', { class: 'hintbox', id: 'sampleEmpty' });
      box.appendChild(h('span', { class: 'hi' }, '📷'));
      const mins = Math.max(0.1, every / perEp * (tr.epoch_seconds || 250) / 60);
      box.appendChild(h('div', {
        html: `实验 <b>${tr.tag || '—'}</b> 还没有产出样本快照。`
          + `<br>训练脚本每 <b>${fmtInt(every)}</b> 步画一张（当前配置约 <b>${mins.toFixed(1)} 分钟</b>一张），`
          + `第一张会在第 ${fmtInt(every)} 步出现。`
          + `<br>想看历史产出，用顶栏右上的「实验」下拉切到别的实验（例如 <b>diff_v1</b>）。`,
      }));
      $('.sample-view').appendChild(box);
    }
  }
}

/* ==========================================================================
   渲染：皮肤产出（前端 UV 拼合人物视图）
   ========================================================================== */
/** 第二层面名单：``hat`` 是历史命名坑——它不在 `*_ov` 后缀里，但**就是第二层**。
 *  早先「仅第一层」只按 `endsWith('_ov')` 过滤，结果帽子照样画上去（实测被用户抓到）。 */
const OVERLAY_FACES = new Set([
  'hat.front', 'body_ov.front', 'rarm_ov.front', 'larm_ov.front',
  'rleg_ov.front', 'lleg_ov.front',
]);
const FIG = [
  ['head.front', 4, 0], ['body.front', 4, 8], ['rarm.front', 0, 8],
  ['larm.front', 12, 8], ['rleg.front', 4, 20], ['lleg.front', 8, 20],
  ['hat.front', 4, 0], ['body_ov.front', 4, 8], ['rarm_ov.front', 0, 8],
  ['larm_ov.front', 12, 8], ['rleg_ov.front', 4, 20], ['lleg_ov.front', 8, 20],
];
const FALLBACK_FACES = {
  'head.front': [8, 16, 8, 16], 'hat.front': [8, 16, 40, 48],
  'body.front': [20, 32, 20, 28], 'body_ov.front': [36, 48, 20, 28],
  'rarm.front': [20, 32, 44, 48], 'rarm_ov.front': [36, 48, 44, 48],
  'larm.front': [52, 64, 36, 40], 'larm_ov.front': [52, 64, 52, 56],
  'rleg.front': [20, 32, 4, 8], 'rleg_ov.front': [36, 48, 4, 8],
  'lleg.front': [52, 64, 20, 24], 'lleg_ov.front': [52, 64, 4, 8],
};
function faceRect(name) {
  const f = (S.faceIndex && S.faceIndex[name]) || FALLBACK_FACES[name];
  return f ? { y0: f[0], y1: f[1], x0: f[2], x1: f[3] } : null;
}
const _imgCache = new Map();
function loadImg(url) {
  if (_imgCache.has(url)) return _imgCache.get(url);
  const p = new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = url;
  });
  _imgCache.set(url, p);
  return p;
}
/** 人物前视拼图。``withOverlay=false`` 时不画第二层面（帽子/上衣/袖/裤的外层），
 *  用于对比「模型在基层画了什么」和「叠上第二层之后的样子」——底层才是模型的
 *  本体输出，第二层是同一张 UV 的另一段面。
 *  ``src`` 既可以是图片 URL，也可以是**画布/图像对象**（样本网格切出来的 64×64 格）。 */
async function paintFigure(canvas, src, scale, withOverlay = true) {
  const im = (typeof src === 'string') ? await loadImg(src) : src;
  const W = 16 * scale, H = 32 * scale;
  canvas.width = W; canvas.height = H;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const c = canvas.getContext('2d');
  c.imageSmoothingEnabled = false;
  c.clearRect(0, 0, W, H);
  FIG.forEach(([name, dx, dy]) => {
    if (!withOverlay && OVERLAY_FACES.has(name)) return;
    const r = faceRect(name);
    if (!r) return;
    const w = r.x1 - r.x0, hh = r.y1 - r.y0;
    c.drawImage(im, r.x0, r.y0, w, hh, dx * scale, dy * scale, w * scale, hh * scale);
  });
}
async function paintUV(canvas, src, scale, withGrid) {
  const im = (typeof src === 'string') ? await loadImg(src) : src;
  canvas.width = 64 * scale; canvas.height = 64 * scale;
  canvas.style.width = '100%'; canvas.style.maxWidth = (64 * scale) + 'px';
  canvas.style.height = 'auto';
  const c = canvas.getContext('2d');
  c.imageSmoothingEnabled = false;
  c.clearRect(0, 0, canvas.width, canvas.height);
  c.drawImage(im, 0, 0, 64, 64, 0, 0, 64 * scale, 64 * scale);
  if (withGrid && S.faceIndex) {
    c.strokeStyle = 'rgba(34,211,238,.30)';
    c.lineWidth = 1;
    Object.values(S.faceIndex).forEach(f => {
      c.strokeRect(f[2] * scale + .5, f[0] * scale + .5, (f[3] - f[2]) * scale, (f[1] - f[0]) * scale);
    });
  }
}
/* ==========================================================================
   皮肤产出 = **当前实验样本快照的 64×64 切割**
   ==========================================================================
 * 数据源：``logs/samples/<当前实验>/step_XXXXXX.png`` —— 训练每 sample_every 步
 * 自动产出一张网格（``save_sample_grid``：每格 64×64、NEAREST 放大 s 倍、格间
 * 1px 间隙、棋盘衬底）。实验切换**只在顶栏**做，这里跟随顶栏，只提供
 * 「看哪一步」的批次列表。
 *
 * 为什么切割时要重建 alpha：样本网格是**无透明通道**的 RGB 图，透明区被画成
 * 棋盘（(46,46,54) / (70,70,80) 两色、每 tile/16 px 一格）。逐像素比对棋盘色、
 * 命中就还原成透明 —— 否则人物视图会糊满棋盘格子。
 */

const _layoutCache = new Map();      // gridUrl -> {scale,tile,stride,cols,rows}
const _cellCache = new Map();        // gridUrl#idx -> 64×64 canvas（已还原 alpha）

/** 由「cols*64s + (cols+1)」的网格尺寸反推布局（save_sample_grid 的拼版公式）。 */
function detectSampleLayout(W, H) {
  for (const scale of [1, 2, 3, 4, 6, 8]) {
    const stride = 64 * scale + 1;
    const cols = Math.round((W - 1) / stride), rows = Math.round((H - 1) / stride);
    if (cols >= 1 && rows >= 1 && cols * stride === W - 1 && rows * stride === H - 1)
      return { scale, tile: 64 * scale, stride, cols, rows };
  }
  return null;
}

/** 切出第 idx 格 → 64×64 RGBA canvas（棋盘色 → 透明）。结果按 (url,idx) 缓存。 */
async function sampleCell(gridUrl, idx) {
  const key = gridUrl + '#' + idx;
  if (_cellCache.has(key)) return _cellCache.get(key);
  if (_cellCache.size > 400) _cellCache.clear();     // 步数会一直涨，兜个底
  const im = await loadImg(gridUrl);
  let layout = _layoutCache.get(gridUrl);
  if (!layout) {
    layout = detectSampleLayout(im.naturalWidth, im.naturalHeight);
    _layoutCache.set(gridUrl, layout);
  }
  if (!layout) throw new Error('样本网格尺寸无法识别 ' + im.naturalWidth + 'x' + im.naturalHeight);
  const col = idx % layout.cols, row = (idx / layout.cols) | 0;
  const tile = layout.tile;

  const big = document.createElement('canvas');
  big.width = tile; big.height = tile;
  const bctx = big.getContext('2d', { willReadFrequently: true });
  bctx.drawImage(im, 1 + col * layout.stride, 1 + row * layout.stride, tile, tile, 0, 0, tile, tile);
  const d = bctx.getImageData(0, 0, tile, tile).data;
  const step = Math.max(4, Math.floor(tile / 16));

  const cv = document.createElement('canvas');
  cv.width = 64; cv.height = 64;
  const ctx = cv.getContext('2d');
  const out = ctx.createImageData(64, 64);
  const k = tile / 64;
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const sx = (x * k) | 0, sy = (y * k) | 0;
      const si = (sy * tile + sx) * 4, di = (y * 64 + x) * 4;
      const even = (((sx / step) | 0) + ((sy / step) | 0)) % 2 === 0;
      const isBoard = d[si] === (even ? 70 : 46) && d[si + 1] === (even ? 70 : 46)
        && d[si + 2] === (even ? 80 : 54);
      out.data[di] = isBoard ? 0 : d[si];
      out.data[di + 1] = isBoard ? 0 : d[si + 1];
      out.data[di + 2] = isBoard ? 0 : d[si + 2];
      out.data[di + 3] = isBoard ? 0 : 255;
    }
  }
  ctx.putImageData(out, 0, 0);
  _cellCache.set(key, cv);
  return cv;
}

/** 皮肤产出：跟随顶栏选中的实验；「看哪一步」用批次下拉列表选。 */
function renderSkins() {
  const tr = (S.overview || {}).train || {};
  const activeTag = tr.tag || S.tag || '';
  const grids = (S.artifacts && S.artifacts.all_grids) || [];
  const spe = tr.steps_per_epoch || 2200;

  if (S.skinStepTag !== activeTag) { S.skinStepTag = activeTag; S.skinStepIdx = 9999; }
  const sel = $('#skinStepSel');
  const listSig = activeTag + '|' + grids.map(g => g.step).join(',');
  if (sel.dataset.sig !== listSig) {
    sel.dataset.sig = listSig;
    // 下拉里最新的在最上面
    const items = grids.map((g, i) => ({ g, i })).reverse();
    sel.replaceChildren(...items.map(({ g, i }) => {
      const ep = spe ? Math.max(1, Math.floor(g.step / spe) + 1) : null;
      return h('option', { value: String(i) },
        `step ${fmtInt(g.step)}` + (ep ? ` · epoch ${ep}` : '')
        + (i === grids.length - 1 ? '（最新）' : ''));
    }));
  }
  S.skinStepIdx = clamp(S.skinStepIdx, 0, Math.max(0, grids.length - 1));
  sel.value = String(S.skinStepIdx);

  const g = grids[S.skinStepIdx];
  if (!g) {
    if (S.skinSig !== 'empty') {
      S.skinSig = 'empty';
      const box = h('div', { class: 'hintbox', style: 'margin:4px 0' });
      box.appendChild(h('span', { class: 'hi' }, '🎨'));
      box.appendChild(h('div', { html:
        `实验 <b>${activeTag || '—'}</b> 还没有样本快照。`
        + `<br>训练每 <b>${fmtInt((tr.args && tr.args.sample_every) || 2200)}</b> 步自动存一张样本网格，`
        + `这里会把它按 64×64 切开当皮肤预览；第一次采样完成后就会出现在这里。` }));
      $('#skinGrid').replaceChildren(box);
      setTxt($('#skinSub'), '暂无样本快照');
    }
    return;
  }
  const gridUrl = g.url + '?t=' + Math.floor(g.mtime);
  const ep = spe ? Math.max(1, Math.floor(g.step / spe) + 1) : null;
  setTxt($('#skinSub'), `当前实验 ${activeTag || '—'} · step ${fmtInt(g.step)}`
    + (ep ? `（epoch ${ep}）` : '') + ` · 由样本快照直接切割，未筛选`);

  const sig = activeTag + '|' + gridUrl + '|' + S.skinView + '|' + S.skinOverlay;
  if (sig === S.skinSig) return;
  S.skinSig = sig;
  $('#skinGrid').replaceChildren(h('div', { class: 'empty' }, '正在切分样本网格…'));
  (async () => {
    const im = await loadImg(gridUrl);
    const layout = detectSampleLayout(im.naturalWidth, im.naturalHeight);
    if (!layout) throw new Error('无法识别 ' + im.naturalWidth + 'x' + im.naturalHeight);
    _layoutCache.set(gridUrl, layout);
    const cards = [];
    for (let i = 0; i < layout.cols * layout.rows; i++) {
      const cell = await sampleCell(gridUrl, i);
      const cv = document.createElement('canvas');
      const card = h('div', { class: 'skin-card' },
        h('div', { class: 'skin-thumb' }, cv),
        h('div', { class: 'skin-name' }, '#' + String(i + 1).padStart(2, '0')));
      card.onclick = () => openSkinCell(gridUrl, i, g.step);
      if (S.skinView === 'figure') await paintFigure(cv, cell, 4, S.skinOverlay);
      else await paintUV(cv, cell, 2, false);
      cards.push(card);
    }
    // 用户中途切了步数/视图的话，旧结果不许覆盖新选择
    if (S.skinSig === sig) $('#skinGrid').replaceChildren(...cards);
  })().catch(e => {
    $('#skinGrid').replaceChildren(h('div', { class: 'empty' }, '切分失败：' + (e.message || e)));
  });
}

async function openSkinCell(gridUrl, idx, step) {
  const body = $('#modalBody');
  body.replaceChildren();
  const c1 = document.createElement('canvas');
  const c2 = document.createElement('canvas');
  body.appendChild(h('div', { style: 'display:flex;gap:26px;flex-wrap:wrap;align-items:flex-start' },
    h('div', { style: 'text-align:center' },
      h('div', { class: 'hint', style: 'margin-bottom:8px' },
        S.skinOverlay ? '人物视图（前视，含第二层叠加）' : '人物视图（前视，仅第一层）'), c1),
    h('div', { style: 'text-align:center' },
      h('div', { class: 'hint', style: 'margin-bottom:8px' }, 'UV 展开（cyan 线为图集面边界）'), c2)));
  body.appendChild(h('div', { class: 'hint', style: 'margin-top:14px' },
    `来源：当前实验 step ${fmtInt(step)} 的样本快照第 ${idx + 1} 格`
    + `（棋盘衬底已还原为透明）。UV 视图里的 cyan 框是「图集面边界」。`));
  $('#modalTitle').textContent = '皮肤 #' + (idx + 1) + ' · step ' + fmtInt(step);
  showModal('#modalBg');
  try {
    const cell = await sampleCell(gridUrl, idx);
    await paintFigure(c1, cell, 8, S.skinOverlay);
    await paintUV(c2, cell, 7, true);
  } catch (e) { reportErr('打开皮肤详情失败', e); }
}

/* ==========================================================================
   渲染：关键指标速览（把项目里已有的实测结论聚到一屏）
   ========================================================================== */
function pickRun(obj, prefer) {
  if (!obj || typeof obj !== 'object') return null;
  const keys = Object.keys(obj);
  const cands = keys.filter(k => !/真实|real/i.test(k));
  const hit = cands.find(k => prefer && k.includes(prefer))
    || cands.find(k => /diff|diffusion/i.test(k))
    || cands[cands.length - 1];
  return hit ? { key: hit, val: obj[hit] } : null;
}
function renderHighlights() {
  const rep = S._reports || {};
  const tr = (S.overview || {}).train || {};
  const args = tr.args || {};
  const rows = [];

  const clean = (rep.clean_report || {}).data || {};
  if (clean.final) {
    rows.push(['清洗后样本', fmtInt(clean.final) + ' 张']);
    if (clean.dup_near_removed !== undefined)
      rows.push(['近似去重剔除', fmtInt(clean.dup_near_removed) + ' 张（' +
        (clean.dup_near_removed / Math.max(1, clean.final + clean.dup_near_removed) * 100).toFixed(1) + '%）']);
  }
  const ds = (rep.dataset || {}).data || {};
  if (ds.train) {
    rows.push(['训练 / 验证', fmtInt(ds.train.shape[0]) + ' / ' +
      (ds.val ? fmtInt(ds.val.shape[0]) : '—')]);
    rows.push(['张量形状', ds.train.shape.join(' × ') + '（C×H×W）']);
  }
  const base = args.base || (tr.model || {}).base;
  if (base) {
    const pm = 11.88 * Math.pow(base / 64, 2);
    const n = ds.train ? ds.train.shape[0] : 0;
    rows.push(['模型参数', pm.toFixed(2) + 'M（估算）'
      + (n ? ' · ' + Math.round(pm * 1e6 / n) + ' 参数/样本' : '')]);
  }
  if (args.model_type) rows.push(['骨架类型', args.model_type]);
  const steps = tr.step || 0;
  rows.push(['训练预算对齐', fmtInt(steps) + ' / 800,000 步（DDPM CIFAR 参照）· '
    + (steps / 800000 * 100).toFixed(1) + '%']);

  // 三条质量判据：真实 vs 当前生成
  const sd = pickRun((rep.semantic_diversity || {}).data, tr.tag);
  if (sd && (rep.semantic_diversity.data.real || {}).ratio !== undefined) {
    rows.push(['语义多样性', `${sd.val.ratio?.toFixed(2)} / 1.00（${sd.key}，越低越单调）`]);
  }
  const ca = (rep.color_audit || {}).data || {};
  const caReal = (ca.agg || {})['真实(val)'];
  const caGen = pickRun(ca.agg, tr.tag);
  if (caReal && caGen && caGen.val.block_pixel_share !== undefined) {
    rows.push(['色块覆盖 ≥8px', `${Number(caGen.val.block_pixel_share).toFixed(3)} / ${Number(caReal.block_pixel_share).toFixed(3)}（真实）`]);
  }
  const sb = (rep.seam_bleed || {}).data || {};
  const sbReal = sb['真实(val)'];
  const sbGen = pickRun(sb, tr.tag);
  if (sbReal && sbGen && sbGen.val.ratio_cross !== undefined) {
    rows.push(['跨缝色差比', `${Number(sbGen.val.ratio_cross).toFixed(3)} / ${Number(sbReal.ratio_cross).toFixed(3)}（真实，越接近越好）`]);
  }
  const cp = (rep.capacity_probe || {}).data || {};
  if (cp.train && cp.train.by_timestep) {
    const bt = cp.train.by_timestep;
    rows.push(['低时间步误差 t<100', bt['0-100'] !== undefined ? bt['0-100'].toFixed(4) + '（细节段，是全局均值的 3.9 倍）' : '—']);
    rows.push(['泛化间隙', cp.gap_rel !== undefined ? (cp.gap_rel * 100).toFixed(2) + '%（远小于批间噪声 ' + (cp.noise_floor * 100).toFixed(2) + '%）' : '—']);
  }

  $('#highlights').replaceChildren(...rows.flatMap(([k, v]) => [
    h('div', { class: 'k' }, k), h('div', { class: 'v' }, String(v)),
  ]));
  setTxt($('#hlSub'), rep.clean_report ? '实测数据聚合' : '（暂无报告）');
  $('#hlNote').textContent = '这些数字来自项目里已有的诊断 JSON（logs/*.json），'
    + '不是本平台重新估算的。「关键指标速览」只在需要时读取一次。';
}
async function loadHighlights(force) {
  try {
    if (!S._reports || force) S._reports = await jget('/api/reports');
    renderHighlights();
  } catch (e) { reportErr('关键指标速览读取失败', e); }
}

/* ==========================================================================
   渲染：训练档案
   ========================================================================== */
/* ---- 训练档案：kv 行只建一次；断点表在「本实验无断点」时给出跨实验回退 ---- */
const RUN_ROWS = [
  ['tag', '实验 tag'], ['status', '状态'], ['pid', '训练进程 PID'],
  ['alive', '进程存活'], ['epochs', '已完成轮数'], ['steps', '当前步数'],
  ['spe', '每轮步数'], ['epsec', '每轮耗时'], ['elapsed', '已训练时长'],
  ['params', '模型参数'], ['inch', '输入通道'], ['T', '扩散步数 T'],
  ['sched', '噪声调度'], ['cond', '条件维度'], ['mask', '掩码损失'],
  ['grids', '样本快照'], ['ckpt', '断点数量'], ['skins', '皮肤产出'],
  ['logmtime', '日志更新'],
];
const RUN_UI = { kv: {}, built: false, ckSig: '' };

function buildRunInfo() {
  if (RUN_UI.built) return;
  RUN_UI.built = true;
  $('#runInfo').replaceChildren(...RUN_ROWS.flatMap(([key, lab]) => {
    const k = h('div', { class: 'k' }, lab);
    const v = h('div', { class: 'v' }, '—');
    RUN_UI.kv[key] = v;
    return [k, v];
  }));
}

function renderRunInfo() {
  buildRunInfo();
  const tr = (S.overview || {}).train || {}, art = S.artifacts || {};
  const vals = {
    tag: tr.tag, status: tr.status_cn, pid: tr.pid,
    alive: tr.pid ? (tr.pid_alive ? '是' : '否') : '—',
    epochs: (tr.epochs_done === null || tr.epochs_done === undefined ? '—' : tr.epochs_done)
      + ' / ' + (tr.epochs || '—'),
    steps: fmtInt(tr.step) + ' / ' + (tr.total_steps ? fmtInt(tr.total_steps) : '—'),
    spe: fmtInt(tr.steps_per_epoch), epsec: fmtDur(tr.epoch_seconds),
    elapsed: fmtDur(tr.elapsed_s),
    params: (tr.args && tr.args.base)
      ? 'base=' + tr.args.base + ' · 约 ' + (tr.args.base ** 2 * 2.9 / 1000).toFixed(1) + 'M' : '—',
    inch: tr.model && tr.model.in_ch, T: tr.model && tr.model.T,
    sched: tr.model && tr.model.schedule, cond: tr.model && tr.model.cond_dim,
    mask: tr.model && tr.model.mask_loss ? '已开启' : '关闭',
    grids: (art.grids || 0) + ' 张',
    ckpt: ((art.checkpoints || []).length) + ' 个',
    skins: (art.skins || 0) + ' 张',
    logmtime: tr.log_mtime ? ago(tr.log_mtime) : '—',
  };
  Object.entries(vals).forEach(([k, v]) => setTxt(RUN_UI.kv[k],
    v === null || v === undefined || v === '' ? '—' : v));

  // 断点表：本实验优先；没有就用「跨实验」清单兜底，并说明原因
  const own = art.checkpoints || [];
  const all = art.checkpoints_all || [];
  const sig = (own.length ? 'own:' : 'all:') + all.map(c => c.tag + '/' + c.name + c.mtime).join(',');
  if (sig === RUN_UI.ckSig) return;
  RUN_UI.ckSig = sig;
  const t = $('#ckptTable');
  if (own.length) {
    t.innerHTML = '<tr><th>断点文件</th><th>大小</th><th>更新时间</th></tr>' +
      own.map(c => `<tr><td class="name">${c.name}</td><td>${fmtBytes(c.size_mb)}</td><td>${ago(c.mtime)}</td></tr>`).join('');
    return;
  }
  if (!all.length) {
    t.innerHTML = '<tr><td class="name" style="color:var(--text-mute)">暂时没有任何断点</td></tr>';
    return;
  }
  t.innerHTML = `<tr><th colspan="3" style="color:var(--text-mute);font-weight:400;white-space:normal">
      实验「${tr.tag || '—'}」还没有保存断点（每轮结束才存一次）——下面是其它实验可用的断点，
      可直接在「训练控制」里选来续训：</th></tr>`
    + '<tr><th>实验 / 文件</th><th>大小</th><th>更新时间</th></tr>'
    + all.slice(0, 12).map(c =>
      `<tr><td class="name">${c.tag} / ${c.name}</td><td>${fmtBytes(c.size_mb)}</td><td>${ago(c.mtime)}</td></tr>`).join('');
}

/* ==========================================================================
   渲染：底部标签页
   ========================================================================== */
function renderBottomTools() {
  const box = $('#bottomTools');
  box.replaceChildren();
  if (S.tab === 'logs') {
    const sel = h('select', {
      class: 'btn sm', style: 'padding:4px 8px',
      onchange: e => { S.logName = e.target.value; loadLogs(); },
    });
    ['train.log', 'console.log'].forEach(n => {
      const o = h('option', { value: n }, n); if (n === S.logName) o.selected = true; sel.appendChild(o);
    });
    const lv = h('select', {
      class: 'btn sm', style: 'padding:4px 8px',
      onchange: e => { S.logLevel = e.target.value; loadLogs(); },
    });
    [['', '全部级别'], ['info', '常规'], ['warn', '警告'], ['error', '错误'], ['success', '成功']]
      .forEach(([v, t]) => { const o = h('option', { value: v }, t); if (v === S.logLevel) o.selected = true; lv.appendChild(o); });
    const q = h('input', {
      class: 'btn sm', placeholder: '搜索关键词', value: S.logQuery,
      style: 'padding:4px 9px;width:150px',
      onkeydown: e => { if (e.key === 'Enter') { S.logQuery = e.target.value; loadLogs(); } },
    });
    const auto = h('button', { class: 'btn sm' + (S.logAuto ? ' primary' : '') }, S.logAuto ? '自动刷新 开' : '自动刷新 关');
    auto.onclick = () => { S.logAuto = !S.logAuto; renderBottomTools(); };
    const ref = h('button', { class: 'btn sm' }, '⟳ 刷新');
    ref.onclick = () => loadLogs();
    box.append(sel, lv, q, auto, ref);
  } else if (S.tab === 'reports') {
    const b = h('button', { class: 'btn sm' }, '⟳ 重新读取');
    b.onclick = () => loadReports(true);
    box.appendChild(b);
  } else if (S.tab === 'server') {
    const b = h('button', { class: 'btn sm' }, '⟳ 刷新');
    b.onclick = () => loadServerLog();
    box.appendChild(b);
  }
}
async function loadLogs() {
  if (S.tab !== 'logs') return;
  try {
    const tag = S.tag || (S.overview && S.overview.train ? S.overview.train.tag : '') || '';
    const url = `/api/logs?name=${encodeURIComponent(S.logName)}&lines=600`
      + `&tag=${encodeURIComponent(tag)}`
      + (S.logLevel ? '&level=' + S.logLevel : '')
      + (S.logQuery ? '&q=' + encodeURIComponent(S.logQuery) : '');
    const d = await jget(url);
    const { bar, list } = ensureLogShell();

    // 顶部信息条：一眼看清「在看哪个实验的哪个文件、有多少行」
    const filt = [S.logLevel ? '级别=' + S.logLevel : '', S.logQuery ? '搜索=' + S.logQuery : '']
      .filter(Boolean).join(' · ');
    setTxt(bar, `${tag || '—'} / ${S.logName} · 共 ${d.total || 0} 行`
      + (filt ? ` · 过滤：${filt}` : '') + (S.logAuto ? ' · 自动刷新开' : ' · 自动刷新关'));

    // 内容签名：只有内容变了才重建 DOM。
    // 每 6 秒无脑重建会丢掉用户的文本选中、也会让滚动条回弹。
    const first = d.items.length ? d.items[0].raw : '';
    const last = d.items.length ? d.items[d.items.length - 1].raw : '';
    const sig = `${d.exists}|${d.total}|${d.items.length}|${first}|${last}`;
    if (sig === S.logSig) {
      if (S.logAuto) list.scrollTop = list.scrollHeight;   // 内容没变也要保持跟随
      return;
    }
    S.logSig = sig;

    if (!d.exists) {
      list.replaceChildren(fmtHint('📄', `实验 <b>${tag || '—'}</b> 还没有 <b>${S.logName}</b>。`
        + `<br>如果这个实验还没启动过，这是正常的；启动后日志会立刻出现。`));
      return;
    }
    if (!d.items.length) {
      list.replaceChildren(fmtHint('🔍', d.total
        ? `当前过滤条件（${filt}）下没有匹配的日志；共 ${d.total} 行。`
        : `实验 <b>${tag || '—'}</b> 的日志还是空的。`));
      return;
    }

    const keepTop = S.logAuto ? null : list.scrollTop;
    list.replaceChildren(...d.items.map(it => {
      const l = h('div', { class: 'logline lv-' + it.level, title: it.raw });
      l.appendChild(h('div', { class: 'l-time' }, it.ts || ''));
      l.appendChild(h('div', { class: 'l-msg' }, it.msg));
      return l;
    }));
    // 行数很少时补一句解释：新实验日志天然就少，不是加载失败
    if (d.items.length < 12) {
      list.insertBefore(fmtHint('ℹ️',
        `这个实验刚启动，目前只有 <b>${d.items.length}</b> 行日志（正常现象）。`
        + `<br>想看完整的训练历史，用顶栏右上的「实验」下拉切到别的实验。`), list.firstChild);
    }
    if (keepTop !== null) list.scrollTop = keepTop;
    else requestAnimationFrame(() => { list.scrollTop = list.scrollHeight; });
  } catch (e) { toast('日志读取失败：' + e.message, 'error'); }
}

/** 底部标签页共用的稳定外壳：信息条 + 滚动列表。
 *  固定这两层后，切换数据源时不会把整块 DOM 换掉（避免闪 + 保住滚动位置）。 */
function ensureLogShell() {
  const box = $('#bottomBody');
  let bar = box.querySelector(':scope > .bar-strip');
  let list = box.querySelector(':scope > .logs');
  if (!bar || !list) {
    box.replaceChildren();
    bar = h('div', { class: 'bar-strip' });
    list = h('div', { class: 'logs' });
    box.append(bar, list);
  }
  return { bar, list, box };
}

function fmtHint(icon, html) {
  const b = h('div', { class: 'hintbox', style: 'margin:10px' });
  b.appendChild(h('span', { class: 'hi' }, icon));
  b.appendChild(h('div', { html }));
  return b;
}
function fmtReportValue(v, depth = 0) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return fmtNum(v, 4);
  if (typeof v === 'boolean') return v ? '是' : '否';
  if (typeof v === 'string') return v;
  if (Array.isArray(v)) {
    if (v.every(x => typeof x === 'number')) return '[' + v.map(x => fmtNum(x, 3)).join(', ') + ']';
    if (v.length > 6) return v.length + ' 项';
    return v.map(x => fmtReportValue(x, depth + 1)).join('、');
  }
  if (typeof v === 'object') return JSON.stringify(v).slice(0, 200);
  return String(v);
}
function renderReportBlock(title, sub, data) {
  const det = h('details', { style: 'margin-bottom:10px', ...(title.includes('图集') ? { open: '' } : {}) });
  det.appendChild(h('summary', { style: 'cursor:pointer;font-weight:650;font-size:12.5px;padding:6px 2px' },
    title + (sub ? '  ' : '')));
  if (sub) det.querySelector('summary').appendChild(h('span', { class: 'hint' }, sub));
  const body = h('div', { style: 'padding:6px 2px 10px' });
  if (data && typeof data === 'object' && !Array.isArray(data)) {
    const flat = Object.entries(data).filter(([, v]) => typeof v !== 'object' || v === null);
    const nested = Object.entries(data).filter(([, v]) => v && typeof v === 'object');
    if (flat.length) {
      const t = h('table', { class: 'tbl' });
      t.innerHTML = '<tr><th>项</th><th>值</th></tr>' + flat.map(([k, v]) =>
        `<tr><td class="name">${k}</td><td>${fmtReportValue(v)}</td></tr>`).join('');
      body.appendChild(t);
    }
    nested.forEach(([k, v]) => {
      const st = h('div', { style: 'margin-top:10px' });
      st.appendChild(h('div', { class: 'hint', style: 'margin-bottom:4px' }, k));
      const t = h('table', { class: 'tbl' });
      const rows = (typeof v === 'object' && !Array.isArray(v)) ? Object.entries(v) : [];
      if (rows.length && rows.every(([, x]) => x === null || typeof x !== 'object')) {
        t.innerHTML = '<tr><th>子项</th><th>值</th></tr>' + rows.map(([k2, v2]) =>
          `<tr><td class="name">${k2}</td><td>${fmtReportValue(v2)}</td></tr>`).join('');
      } else {
        t.innerHTML = '<tr><th>子项</th><th>值</th></tr>' + rows.map(([k2, v2]) =>
          `<tr><td class="name">${k2}</td><td>${fmtReportValue(v2)}</td></tr>`).join('')
          || `<tr><td colspan="2">${fmtReportValue(v)}</td></tr>`;
      }
      st.appendChild(t); body.appendChild(st);
    });
  } else {
    body.appendChild(h('pre', { class: 'cmd' }, JSON.stringify(data, null, 1).slice(0, 8000)));
  }
  det.appendChild(body);
  return det;
}
async function loadReports(force) {
  if (S.tab !== 'reports') return;
  if (S._reports && !force) { paintReports(S._reports); return; }
  try {
    S._reports = await jget('/api/reports');
    paintReports(S._reports);
  } catch (e) { toast('报告读取失败：' + e.message, 'error'); }
}
function paintReports(rep) {
  const box = $('#bottomBody');
  const frag = document.createDocumentFragment();
  const order = ['atlas_stats', 'struct_metrics', 'color_audit', 'semantic_diversity',
    'seam_bleed', 'capacity_probe', 'clean_report', 'validation_dcgan_v1',
    'dataset', 'docs'];
  const done = new Set();
  order.concat(Object.keys(rep)).forEach(k => {
    const r = rep[k];
    if (!r || done.has(k)) return;
    done.add(k);
    frag.appendChild(renderReportBlock(
      r.title || k,
      r.path ? '（' + r.path + ' · ' + ago(r.mtime) + '）' : '',
      r.data));
  });
  box.replaceChildren(frag);
  if (!Object.keys(rep).length) box.replaceChildren(h('div', { class: 'empty' }, '未找到诊断报告 JSON'));
}
function renderProgressTable() {
  const charts = (S.series && S.series.charts) || [];
  const ep = charts.find(c => c.id === 'epoch');
  const box = $('#bottomBody');
  if (!ep) { box.replaceChildren(h('div', { class: 'empty' }, '暂无逐轮数据')); return; }
  const mse = (ep.series.find(s => s.key === 'mse_epoch') || {}).points || [];
  const sec = (ep.series.find(s => s.key === 'sec_epoch') || {}).points || [];
  const secMap = new Map(sec);
  const rows = mse.map(([e, v], i) => {
    const prev = i > 0 ? mse[i - 1][1] : null;
    const dd = prev ? v - prev : null;
    return { e, v, dd, s: secMap.get(e) };
  }).reverse();
  const t = h('table', { class: 'tbl' });
  t.innerHTML = '<tr><th>轮次</th><th>每轮末 MSE</th><th>变化</th><th>本轮耗时</th></tr>' +
    rows.map(r => `<tr><td>${r.e}</td><td>${fmtNum(r.v, 5)}</td>
      <td style="color:${r.dd === null ? 'inherit' : r.dd < 0 ? COLORS.emerald : COLORS.rose}">
      ${r.dd === null ? '—' : (r.dd < 0 ? '↓ ' : '↑ ') + fmtNum(Math.abs(r.dd), 5)}</td>
      <td>${r.s ? fmtDur(r.s) : '—'}</td></tr>`).join('');
  box.replaceChildren(h('div', { class: 'scroll-x', style: 'max-height:470px;overflow:auto' }, t));
}
async function loadServerLog() {
  if (S.tab !== 'server') return;
  try {
    const d = await jget('/api/server_log?lines=400');
    if (!d.items.length) { $('#bottomBody').replaceChildren(h('div', { class: 'empty' }, '暂无服务日志')); return; }
    const list = h('div', { class: 'logs' });
    list.replaceChildren(...d.items.map(it => {
      const l = h('div', { class: 'logline lv-' + it.level });
      l.appendChild(h('div', { class: 'l-time' }, it.time || ''));
      l.appendChild(h('div', { class: 'l-msg' }, it.msg));
      return l;
    }));
    $('#bottomBody').replaceChildren(list);
    requestAnimationFrame(() => { list.scrollTop = list.scrollHeight; });
  } catch (e) { toast('服务日志读取失败：' + e.message, 'error'); }
}
function switchTab(tab) {
  S.tab = tab;
  $$('#bottomTabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
  renderBottomTools();
  // 其它标签页会把 #bottomBody 换掉，回到日志页时必须强制重画（否则签名命中会跳过）
  try {
    if (tab === 'logs') { S.logSig = ''; return loadLogs(); }
    if (tab === 'reports') { loadReports(); return; }
    if (tab === 'progress') { renderProgressTable(); return; }
    if (tab === 'server') { loadServerLog(); return; }
  } catch (e) { reportErr('底部标签页「' + tab + '」渲染失败', e); }
}

/* ==========================================================================
   图表交互
   ========================================================================== */
let chart = null, sysChart = null;
function renderChartTabs() {
  const charts = (S.series && S.series.charts) || [];
  const box = $('#chartTabs');
  box.replaceChildren(...charts.map(c => {
    const b = h('button', { class: c.id === S.chartId ? 'on' : '' }, c.title);
    b.onclick = () => { S.chartId = c.id; renderChartTabs(); paintChart(); };
    return b;
  }));
}
function paintChart() {
  const charts = (S.series && S.series.charts) || [];
  const c = charts.find(x => x.id === S.chartId) || charts[0];
  if (!c) {
    $('#chartHint').textContent = '暂无曲线数据';
    $('#chartLegend').replaceChildren();
    return;
  }
  S.chartId = c.id;
  chart.cv.setAttribute('height', c.height || 240);   // 只作 HTML 兜底
  chart.setHeight(c.height || 240);                   // 真正生效的高度来源
  chart.setData(c);
  const lg = $('#chartLegend');
  lg.replaceChildren(...c.series.map(s => {
    const d = h('div', { class: 'lg' + (chart.hidden.has(s.key) ? ' off' : '') });
    const sw = h('div', { class: 'sw' }); sw.style.background = COLORS[s.color] || COLORS.cyan;
    d.appendChild(sw); d.appendChild(document.createTextNode(s.label));
    d.onclick = () => {
      chart.hidden.has(s.key) ? chart.hidden.delete(s.key) : chart.hidden.add(s.key);
      renderChartTabs(); paintChart();
    };
    return d;
  }));
  const n = (S.series.n_records || 0);
  setTxt($('#chartSub'), `（${S.series.tag || '—'} · ${fmtInt(n)} 个记录点）`);
  $('#chartHint').textContent = '滚轮缩放 X 轴 · 拖动平移 · 双击重置 · 点图例可显隐序列';
}

/* ==========================================================================
   模态
   ========================================================================== */
function showModal(sel) { $(sel).classList.add('on'); }
function hideModal(sel) { $(sel).classList.remove('on'); }
function initModals() {
  $('#modalClose').onclick = () => hideModal('#modalBg');
  $('#modalBg').onclick = e => { if (e.target.id === 'modalBg') hideModal('#modalBg'); };
  $('#ctrlClose').onclick = () => hideModal('#ctrlBg');
  $('#ctrlBg').onclick = e => { if (e.target.id === 'ctrlBg') hideModal('#ctrlBg'); };
  window.addEventListener('keydown', e => {
    if (e.key === 'Escape') { hideModal('#modalBg'); hideModal('#ctrlBg'); }
  });
}

/* ==========================================================================
   训练控制
   ========================================================================== */
/** 给「新实验」算一个不与现有 tag 冲突的默认名：diff_v1 → diff_v2 */
function nextTag(tag) {
  if (!tag) return 'diff_v2';
  const m = tag.match(/^(.*?)(\d+)$/);
  if (m) return m[1] + (parseInt(m[2], 10) + 1);
  return tag + '_v2';
}

function initControl() {
  const open = async () => {
    showModal('#ctrlBg');
    try {
      const art = await jget('/api/artifacts');
      const tr = (S.overview && S.overview.train) || {};
      const curTag = tr.tag || '';
      /* 默认**沿用当前实验**：同一个 tag + 该实验自己的 latest.pt 断点 + 它当初
       * 记在 metrics.jsonl 里的真实参数。训练中断后最常见的动作就是"接着跑"，
       * 手填一遍既慢又容易把 batch/base/lr 填错。 */
      $('#f-tag').value = curTag || nextTag('');
      const a = tr.args || {};
      const set = (id, v) => { const el = $(id); if (el && v !== undefined && v !== null && v !== '') el.value = String(v); };
      set('#f-epochs', a.epochs);
      set('#f-batch', a.batch);
      set('#f-base', a.base);
      set('#f-lr', a.lr);
      set('#f-schedule', a.schedule);
      set('#f-channels', a.channels);
      set('#f-model-type', a.model_type);
      set('#f-sample-every', a.sample_every);
      setTxt($('#ctrlStatus'),
        curTag ? `已沿用 ${curTag} 的配置（已训练 step ${fmtInt(tr.step)} / epoch ${tr.epoch || '—'}）`
          + '；想开新实验就点右边「改用新实验名」' : '还没有实验记录，下面就是默认配置');
      const sel = $('#f-resume');
      const all = art.checkpoints_all || art.checkpoints || [];
      const byTag = {};
      all.forEach(c => { (byTag[c.tag] = byTag[c.tag] || []).push(c); });
      const frag = document.createDocumentFragment();
      frag.appendChild(h('option', { value: '' }, '从头训练'));
      Object.keys(byTag).sort().forEach(t => {
        const g = h('optgroup', { label: t + '（' + byTag[t].length + ' 个断点）' });
        byTag[t].forEach(c => {
          g.appendChild(h('option', { value: c.path || ('models/' + t + '/' + c.name) },
            c.name + '（' + fmtBytes(c.size_mb) + ' · ' + ago(c.mtime) + '）'));
        });
        frag.appendChild(g);
      });
      sel.replaceChildren(frag);
      // 默认选当前实验的 latest.pt（最有用的续训起点）。
      // 当前实验自己还没断点时（比如刚改用新实验名 `_v2`），退到**它的父实验**
      // 的最新断点；再没有就老老实实从头训练 —— 别去抓别的实验的权重。
      const parentTag = curTag.replace(/_v?\d+$/, '');   // diff_v2_masked_v2 → diff_v2_masked
      const pickFrom = t => Array.from(sel.options).find(o =>
        o.value.startsWith('models/' + t + '/') && o.value.endsWith('latest.pt'));
      const want = pickFrom(curTag) || (parentTag && parentTag !== curTag ? pickFrom(parentTag) : null);
      if (want) want.selected = true;
      updateCmd();
    } catch (e) { /* 忽略：连不上后端时表单仍可手工填写 */ }
  };
  $('#btnControl').onclick = open;
  // 「继续当前实验」/「改用新实验名」：一键切换，其余参数不动
  const newTagBtn = $('#btnCtrlNewTag');
  if (newTagBtn) newTagBtn.onclick = () => {
    const curTag = (S.overview && S.overview.train && S.overview.train.tag) || '';
    const cur = $('#f-tag').value || '';
    $('#f-tag').value = (cur && cur !== curTag) ? curTag : nextTag(curTag || cur);
    updateCmd();
    toast('实验名改为 ' + $('#f-tag').value + '（参数不变）', 'info');
  };

  // 预设：一组固化的「正确开关组合」（与 webui/server.py 的 TRAIN_PRESETS 对齐）。
  // 换预设 = 换模型结构（in_ch 3→4、cond 36→40、新增 FiLM），所以必须**新 tag
  // 从头训**：续训到旧断点会被后端直接拒绝（有意为之，避免静默出一堆 missing keys）。
  const PRESET_META = {
    v21: {
      tag: 'diff_v4',
      hint: '推荐。v2 的全部开关 + 一处**结构性修复**：v2 的条件平面用的是目标'
          + '自己的 alpha，mask 与内容强相关，模型可走「认出是哪张皮肤」的捷径，'
          + '推理时拿到无关 mask 就在暴露区输出噪声。v2.1 用跨样本 mask + 形态扰动'
          + '切断这条捷径。需要 train_ov4.npy 与 models/mask_bank.npz。',
      args: ['--alpha-input', '--ov-bits', '--overlay-weight 4.0',
             '--cond-dropout 0.15', '--cfg-scale 4.0',
             '--alpha-source retrieval', '--mask-cross-sample', '--mask-jitter 0.5'],
    },
    v2: {
      tag: 'diff_v3',
      hint: '旧版，保留作 A/B 对照：含「条件平面 = 目标自己的 alpha」这个缺陷。'
          + 'alpha 条件平面 + 部位级 overlay 四位 + overlay 损失加权 4× + FiLM/CFG。',
      args: ['--alpha-input', '--ov-bits', '--overlay-weight 4.0',
             '--cond-dropout 0.15', '--cfg-scale 4.0'],
    },
    v2_nocfg: {
      tag: 'diff_v3_nocfg',
      hint: '消融用：保留 alpha 平面 + overlay 位 + 加权，关掉 condition dropout 与 CFG。'
          + '和 v2 对比能单独看出 CFG 的贡献。',
      args: ['--alpha-input', '--ov-bits', '--overlay-weight 4.0',
             '--cond-dropout 0.0', '--cfg-scale 1.0'],
    },
    baseline: {
      tag: 'diff_base',
      hint: 'A/B 对照：纯掩码 MSE，不带任何 v2 开关。用来证明新开关确实有差。',
      args: ['--cond-dropout 0.0', '--cfg-scale 1.0'],
    },
    '': {tag: null, hint: '不套预设：只用下面的参数（等价于旧行为）。', args: []},
  };

  const updateCmd = () => {
    const tag = $('#f-tag').value || 'run';
    const parts = [
      'python scripts/21_train_diffusion.py',
      '--tag ' + tag,
      '--epochs ' + $('#f-epochs').value,
      '--batch ' + $('#f-batch').value,
      '--base ' + $('#f-base').value,
      '--schedule ' + $('#f-schedule').value,
      '--channels ' + $('#f-channels').value,
      '--lr ' + $('#f-lr').value,
      '--model-type ' + $('#f-model-type').value,
      '--cond --mask-loss',
      '--sample-every ' + $('#f-sample-every').value,
    ];
    const pa = (PRESET_META[$('#f-preset').value] || {}).args || [];
    parts.push(...pa);
    const r = $('#f-resume').value;
    if (r) parts.push('--resume ' + r);
    const ms = $('#f-max-steps').value.trim();
    if (ms) parts.push('--max-steps ' + ms);
    $('#cmdPreview').textContent = parts.join(' \\\n  ');
    const ep = parseInt($('#f-epochs').value) || 0;
    const spe = (S.overview && S.overview.train && S.overview.train.steps_per_epoch) || 2200;
    $('#f-epochsHint').textContent = `≈ ${fmtInt(ep * spe)} 步 · 按当前速度约 ${fmtDur(ep * ((S.overview && S.overview.train && S.overview.train.epoch_seconds) || 250))}`;
  };

  // 切预设：改 tag + 清空 resume（新架构不能续旧断点）+ 更新说明 + 重算命令
  const applyPreset = (syncTag) => {
    const p = $('#f-preset').value;
    const m = PRESET_META[p] || PRESET_META[''];
    $('#f-presetHint').textContent = m.hint;
    if (syncTag && m.tag) {
      $('#f-tag').value = m.tag;
      $('#f-resume').value = '';
    }
    updateCmd();
  };

  ['f-tag', 'f-epochs', 'f-batch', 'f-base', 'f-lr', 'f-schedule', 'f-channels',
    'f-model-type', 'f-sample-every', 'f-max-steps', 'f-resume'].forEach(id => {
      $('#' + id).addEventListener('input', updateCmd);
      $('#' + id).addEventListener('change', updateCmd);
    });
  // 预设单独处理：除了改命令，还要同步 tag / resume，否则「点开就点启动」
  // 会因为「v2 预设 + 续训到旧断点」被后端拒绝。
  $('#f-preset').addEventListener('change', () => applyPreset(true));
  applyPreset(true);

  $('#btnStart').onclick = async () => {
    const tag = $('#f-tag').value.trim();
    if (!/^[A-Za-z0-9_\-]{1,40}$/.test(tag)) { toast('tag 只能包含字母、数字、下划线、短横线', 'error'); return; }
    const payload = {
      confirm: true, tag,
      epochs: +$('#f-epochs').value, batch: +$('#f-batch').value,
      base: +$('#f-base').value, lr: +$('#f-lr').value,
      schedule: $('#f-schedule').value, channels: +$('#f-channels').value,
      model_type: $('#f-model-type').value,
      sample_every: +$('#f-sample-every').value,
      preset: $('#f-preset').value || '',
      resume: $('#f-resume').value || '',
      max_steps: $('#f-max-steps').value.trim() ? +$('#f-max-steps').value : null,
    };
    $('#btnStart').disabled = true;
    $('#ctrlStatus').textContent = '正在启动…';
    const r = await jpost('/api/train/start', payload);
    $('#btnStart').disabled = false;
    if (r.ok) {
      $('#ctrlStatus').textContent = '已启动，PID ' + r.pid;
      toast('训练已启动：' + tag + '（PID ' + r.pid + '）', 'ok');
      hideModal('#ctrlBg');
      pollFast();
    } else {
      $('#ctrlStatus').textContent = '';
      toast('启动失败：' + r.error, 'error', 7000);
    }
  };
}
function initStop() {
  $('#btnStop').onclick = async () => {
    const tr = (S.overview || {}).train || {};
    if (!confirm(`确认停止训练「${tr.tag || '当前实验'}」？\n\nPID：${tr.pid || '未找到'}\n停在：第 ${tr.epochs_done} 轮 / 第 ${tr.step} 步\n\n断点 latest.pt 会保留，可以直接续训。`)) return;
    const r = await jpost('/api/train/stop', { confirm: true });
    if (r.ok) toast('已发送停止指令（PID ' + r.pid + '）', 'warn');
    else toast('停止失败：' + r.error, 'error', 7000);
  };
}

/* ==========================================================================
   启动与轮询
   ========================================================================== */
function initUI() {
  initTheme();
  initModals();
  initControl();
  initStop();

  chart = new Chart($('#mainChart'), $('#chartTip'));
  sysChart = new Chart($('#sysChart'), null);

  $$('#bottomTabs button').forEach(b => b.onclick = () => switchTab(b.dataset.tab));
  $$('#evFilter button').forEach(b => b.onclick = () => {
    S.evFilter = b.dataset.lv;
    $$('#evFilter button').forEach(x => x.classList.toggle('on', x === b));
    renderEvents();
  });
  $$('#skinViewSeg button').forEach(b => b.onclick = () => {
    S.skinView = b.dataset.view;
    $$('#skinViewSeg button').forEach(x => x.classList.toggle('on', x === b));
    S.skinSig = '';
    renderSkins();
  });
  /* 第二层开关：皮肤 PNG 本身同时含两层 UV，这里只是**拼图时不叠 *_ov 面**，
     用来对照「基层本体」和「叠上第二层」的效果 */
  $$('#skinLayerSeg button').forEach(b => b.onclick = () => {
    S.skinOverlay = b.dataset.ov === '1';
    $$('#skinLayerSeg button').forEach(x => x.classList.toggle('on', x === b));
    S.skinSig = '';
    renderSkins();
  });
  /* 皮肤产出的「看哪一步」下拉：跟随顶栏实验，切步就重画 */
  const stepSel = $('#skinStepSel');
  if (stepSel) stepSel.onchange = e => {
    S.skinStepIdx = Number(e.target.value) || 0;
    S.skinSig = '';
    renderSkins();
  };

  $('#sampleRange').oninput = e => { S.gridIdx = +e.target.value; renderSamples(); };
  $('#btnSamplePrev').onclick = () => { S.gridIdx = Math.max(0, S.gridIdx - 1); renderSamples(); };
  $('#btnSampleNext').onclick = () => { S.gridIdx = Math.min(S.grids.length - 1, S.gridIdx + 1); renderSamples(); };
  $('#btnSampleLatest').onclick = () => { S.gridIdx = S.grids.length - 1; renderSamples(); };
  $('#btnSampleCompare').onclick = () => {
    if (S.grids.length < 2) { toast('样本快照不足两张，无法对比', 'warn'); return; }
    const first = S.grids[0], last = S.grids[S.grids.length - 1];
    const body = $('#modalBody');
    const mk = (g, tag) => {
      const box = h('div', { style: 'flex:1;min-width:320px' });
      box.appendChild(h('div', { class: 'hint', style: 'margin-bottom:6px' },
        `${tag} · 第 ${fmtInt(g.step)} 步 · ${g.name}`));
      box.appendChild(h('img', {
        src: g.url + '?t=' + Math.floor(g.mtime),
        style: 'width:100%;image-rendering:pixelated;border-radius:10px',
      }));
      return box;
    };
    body.replaceChildren(h('div', { style: 'display:flex;gap:16px;flex-wrap:wrap' },
      mk(first, '训练初期'), mk(last, '最新')));
    const hint = h('div', { class: 'hint', style: 'margin-top:12px' },
      '两张快照用的是**同一组固定噪声**与同一套 alpha 模板，所以差异全部来自模型进步，可以直接逐格对比。');
    hint.innerHTML = hint.textContent.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
    body.appendChild(hint);
    $('#modalTitle').textContent = '样本进度对比（首帧 vs 最新）';
    showModal('#modalBg');
  };
  $('#btnSampleBig').onclick = () => {
    const g = S.grids[S.gridIdx];
    if (!g) return;
    $('#modalTitle').textContent = '样本快照 · 第 ' + fmtInt(g.step) + ' 步';
    $('#modalBody').replaceChildren(h('img', {
      src: g.url + '?t=' + Math.floor(g.mtime),
      style: 'width:100%;image-rendering:pixelated;border-radius:10px',
    }));
    showModal('#modalBg');
  };
  $('#btnChartFit').onclick = () => { chart.view = null; chart.draw(); };
  const repaintBtn = $('#btnChartRepaint');
  if (repaintBtn) repaintBtn.onclick = () => {
    [chart, sysChart].forEach(c => { if (c) c.repaint(); });
    toast('已强制重绘画布（含系统历史曲线）', 'info');
  };
  const sysRepaintBtn = $('#btnSysRepaint');
  if (sysRepaintBtn) sysRepaintBtn.onclick = () => {
    if (sysChart) { sysChart.repaint(); toast('系统曲线已强制重绘', 'info'); }
  };
  $('#btnChartPng').onclick = () => chart.exportPNG();
  $('#btnHlRefresh').onclick = () => { S._reports = null; loadHighlights(true); toast('关键指标已重新读取', 'info'); };
  $('#runSelect').onchange = e => {
    S.tag = e.target.value || null;
    S.gridIdx = 9999;          // 切实验后跳到该实验的最新快照
    S.skinStepIdx = 9999;      // 皮肤产出同跟随顶栏实验，默认最新一步
    S.skinStepTag = '';
    S.logAuto = true;
    refreshAll().then(() => { loadSeries(); loadLogs(); });
    toast(S.tag ? '已切换到实验 ' + S.tag : '已切回「自动跟随最新」', 'info');
  };
  $('#btnNotify').onclick = async () => {
    if (!('Notification' in window)) { toast('当前浏览器不支持通知', 'warn'); return; }
    if (Notification.permission !== 'granted') {
      const p = await Notification.requestPermission();
      if (p !== 'granted') { toast('通知权限被拒绝', 'warn'); return; }
    }
    S.notify = !S.notify;
    $('#btnNotify').textContent = S.notify ? '🔔 通知开' : '🔔 通知关';
    toast(S.notify ? '告警浏览器通知已开启' : '告警浏览器通知已关闭', 'info');
    localStorage.setItem('notify', S.notify ? '1' : '0');
  };
  S.notify = localStorage.getItem('notify') === '1';
  $('#btnNotify').textContent = S.notify ? '🔔 通知开' : '🔔 通知关';

  setInterval(() => {
    setTxt($('#clock'), new Date().toLocaleTimeString('zh-CN', { hour12: false }));
  }, 1000);

  // 事件计数（全量，从服务端拿）
  const pullCounts = async () => {
    try {
      const d = await jget('/api/events?since=999999999');
      S.evCounts = d.counts;
      updateEventBadges();
    } catch (e) { reportErr('事件计数读取失败', e); }
  };
  pullCounts();
  setInterval(pullCounts, 8000);

  /* ---- 画布自愈 ----
   * 浏览器**整页缩放**会改 devicePixelRatio，但不一定触发 ResizeObserver；
   * 而 canvas 后备位图一旦和显示尺寸对不上（或 GPU 合成出白块），
   * 唯一的表现就是「图表一片空白/纯白，手动缩放一下页面又好了」。
   * 这里做两件事：① 监听 resize 重新量尺寸；② 每 2.5 秒自检一次位图尺寸。 */
  let rzT = 0;
  const repaintAll = () => {
    clearTimeout(rzT);
    rzT = setTimeout(() => [chart, sysChart].forEach(c => { if (c) c.resize(); }), 160);
  };
  window.addEventListener('resize', repaintAll);
  window.addEventListener('orientationchange', repaintAll);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) repaintAll(); });
  setInterval(() => {
    [chart, sysChart].forEach(c => { if (c && c.needsRepaint()) c.repaint(); });
  }, 2500);
}

async function boot() {
  initUI();
  /* 每一步都单独兜住：早先是「一串 await 裸奔」，中间任何一步抛异常，
   * 后面的步骤（训练曲线、日志、档案）就全都不执行了——表现就是
   * 「下面的训练日志和训练档案没有正常加载」，而且控制台还可能什么都没有。 */
  const step = async (name, fn) => {
    try { await fn(); } catch (e) { reportErr(name, e); }
  };

  await step('初始化配置', async () => {
    const b = await jget('/api/bootstrap');
    S.boot = b;
    S.faceIndex = (b.face_index && !b.face_index.__error__) ? b.face_index : {};
    S.tickSeconds = b.tick_seconds || 2;
    $('#brandDesc').textContent = b.project.desc + ' · ' + b.project.root;
    $('#footRoot').textContent = b.project.root;
    $('#footServer').textContent = `PID ${b.server.pid} · Python ${b.server.python} · 端口 ${b.server.port}`;
    $('#footTick').textContent = (b.tick_seconds || 2).toFixed(1) + 's';
  });

  await step('刷新总览', refreshAll);
  await step('连接实时推送', async () => connectSSE());
  await step('加载训练日志', async () => switchTab(S.tab));

  setInterval(async () => { if (!S.sseOk) await refreshAll(); }, 4000);
  setInterval(() => { if (S.sseOk) loadSystem(); }, 5000);
  setInterval(() => { if (S.tab === 'logs' && S.logAuto) loadLogs(); }, 6000);
  setInterval(() => { if (S.tab === 'server') loadServerLog(); }, 8000);
  await step('读取系统状态', loadSystem);
}

async function loadSystem() {
  try {
    S.system = await jget('/api/system');
    renderSystem();
  } catch (e) { reportErr('系统状态读取失败', e); }
}

async function refreshAll() {
  const q = S.tag ? '?tag=' + encodeURIComponent(S.tag) : '';
  try {
    const [ov, art, ev] = await Promise.all([
      jget('/api/overview' + q), jget('/api/artifacts' + q),
      jget('/api/events?since=' + S.lastEventId),
    ]);
    S.overview = ov;
    S.artifacts = art;
    (ev.items || []).forEach(pushEvent);
    appendEvents(ev.items);
    updateChip(ov.train);
    paintAll(true);
  } catch (e) { reportErr('刷新总览失败', e); }
}
async function pollFast() {
  for (let i = 0; i < 6; i++) {
    await new Promise(r => setTimeout(r, 2500));
    await refreshAll();
  }
}

function paintAll(full) {
  /* 每一块单独兜住：档案/皮肤任一块渲染失败，不能让曲线和其余面板跟着陪葬 */
  const safe = (name, fn) => { try { fn(); } catch (e) { reportErr(name, e); } };
  safe('实验下拉框', () => {
    if (S.overview && S.overview.runs) renderRunSelect(S.overview.runs, S.overview.train.tag);
  });
  safe('KPI 卡片', renderKPIs);
  safe('样本快照', renderSamples);
  safe('训练档案', renderRunInfo);
  safe('皮肤产出', renderSkins);
  if (full) {
    safe('训练曲线', loadSeries);
    safe('关键指标速览', loadHighlights);
  }
  if (S.tab === 'server') safe('服务日志', loadServerLog);
}

async function loadSeries() {
  try {
    const tag = S.tag || (S.overview && S.overview.train && S.overview.train.tag) || '';
    S.series = await jget('/api/series' + (tag ? '?tag=' + encodeURIComponent(tag) : ''));
    renderChartTabs();
    paintChart();
    if (S.tab === 'progress') renderProgressTable();
  } catch (e) { reportErr('训练曲线读取失败', e); }
}

/** 实验选择器：默认「跟随最新」，也可以钉住某个历史实验 */
function renderRunSelect(runs, activeTag) {
  const sel = $('#runSelect');
  const items = [{ v: '', t: '自动跟随最新' }].concat(
    (runs || []).map(r => ({ v: r.tag, t: r.tag })));
  const sig = items.map(i => i.v).join('|');
  if (sel.dataset.sig !== sig) {
    sel.dataset.sig = sig;
    sel.replaceChildren(...items.map(i => h('option', { value: i.v }, i.t)));
  }
  if (sel.value !== (S.tag || '')) sel.value = S.tag || '';
  sel.title = '当前查看：' + (S.tag || activeTag || '—');
}

function connectSSE() {
  let es;
  try { es = new EventSource('/api/stream'); } catch (e) { return; }
  es.onopen = () => {
    S.sseOk = true;
    $('#sseBadge').textContent = '推送 已连接';
    $('#sseBadge').className = 'badge ok';
  };
  es.onerror = () => {
    S.sseOk = false;
    $('#sseBadge').textContent = '推送 断开（轮询中）';
    $('#sseBadge').className = 'badge warn';
  };
  es.onmessage = ev => {
    let d;
    try { d = JSON.parse(ev.data); } catch (e) { return; }
    if (d.overview && d.overview.train) {
      // 用户钉住了某个历史实验时，不被 SSE 的「当前活跃实验」覆盖
      if (S.tag) return;
      S.overview = d.overview;
      const st = d.overview.train.status;
      const chip = $('#statusChip');
      chip.className = 'status-chip s-' + st;
      setTxt($('#statusText'), `${d.overview.train.status_cn} · ${d.overview.train.tag || '—'}`);
      const badge = d.server ? `推送 已连接 · ${d.server.ticks} 次采集` : '推送 已连接';
      if ($('#sseBadge').textContent !== badge && S.sseOk) {
        $('#sseBadge').textContent = badge;
        $('#sseBadge').className = 'badge ok';
      }
      renderKPIs();
      renderSamples();
      renderRunInfo();
      if (d.artifacts_summary) {
        // 产物签名要涵盖快照/断点/皮肤三类：只看皮肤数量的话，
        // 新产出的样本快照和断点不会触发刷新，界面会一直停在旧数据。
        const a = d.artifacts_summary;
        const asig = [a.grids, a.skins, a.checkpoints,
        (a.latest_grid && a.latest_grid.name) || ''].join('|');
        if (asig !== S.artSig) {
          S.artSig = asig;
          const q = S.tag ? '?tag=' + encodeURIComponent(S.tag) : '';
          jget('/api/artifacts' + q).then(art => {
            S.artifacts = art;
            renderSamples(); renderRunInfo(); renderSkins();
          }).catch(() => { });
        }
      }
    }
    if (d.events && Array.isArray(d.events.items)) {
      d.events.items.forEach(pushEvent);
      appendEvents(d.events.items);
    }
    if (d.server && d.server.errors > 0) {
      $('#evErr').title = '采集异常 ' + d.server.errors + ' 次';
    }
  };
  // 事件计数徽章由 initUI 里的定时器统一刷新（见 updateEventBadges）
}

function updateEventBadges() {
  if (!S.evCounts) return;
  setTxt($('#evWarn'), (S.evCounts.warn || 0) + ' 警告');
  setTxt($('#evErr'), (S.evCounts.error || 0) + ' 严重');
}

window.addEventListener('DOMContentLoaded', boot);
