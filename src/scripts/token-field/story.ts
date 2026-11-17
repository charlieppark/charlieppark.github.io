// Wires the TokenStory DOM (chapters, picker, question chips, mask switches, legends,
// floating labels, live readouts) to the 3D TokenField.
import fig1 from '../../data/fig1.json';
import { fromU8, loadAttention, type Box, type SceneData, type VideoData } from './attention';
import type { FieldState, TokenField } from './field';
import { LEGEND, legendHTML } from './legend';
import {
  coverage, evidenceStats, inside, narrate, verdict, VERDICT,
  type Condition, type EvidenceStats, type FindItem, type PaperMeta, type PortraitProbe, type ProblemItem, type WhoseItem,
} from './narrate';
import 'lenis/dist/lenis.css';

const $ = <T extends Element>(root: ParentNode, sel: string) => root.querySelector<T>(sel)!;
const esc = (t: string) => t.replace(/[<>&]/g, '');

function webglAvailable() {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch { return false; }
}

const loadImage = (src: string) =>
  new Promise<HTMLImageElement>((res, rej) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = src;
  });

/** Union of boxes as per-token coverage (clamped to 1). */
function unionCoverage(boxes: Box[], gw: number, gh: number) {
  const w = new Float32Array(gw * gh);
  for (const b of boxes) coverage(b, gw, gh).forEach((v, i) => { w[i] = Math.min(1, w[i] + v); });
  return w;
}

export async function mountStory() {
  const stage = document.querySelector<HTMLElement>('[data-stage]');
  if (!stage) return;
  const story = stage.closest<HTMLElement>('.story')!;
  const scenes = fig1.scenes as SceneData[];
  const video = fig1.video as unknown as VideoData; // JSON infers grid as number[], not a tuple
  const problems = (fig1.problem ?? []) as unknown as ProblemItem[];
  const papers = JSON.parse(story.dataset.papers ?? '[]') as PaperMeta[];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const chapters = [...story.querySelectorAll<HTMLElement>('[data-chapter]')];
  const rail = [...stage.querySelectorAll<HTMLElement>('[data-rail]')];
  const pickerBtns = [...stage.querySelectorAll<HTMLButtonElement>('[data-scene]')];
  const picker = $<HTMLElement>(stage, '.picker');
  const credit = $<HTMLElement>(stage, '[data-credit]');
  const qlabel = $<HTMLElement>(stage, '[data-qlabel]');
  const qtext = $<HTMLElement>(stage, '[data-qtext]');
  const ctag = $<HTMLElement>(stage, '[data-ctag]');
  const boxtags = [...stage.querySelectorAll<HTMLElement>('[data-boxtag]')];
  const legend = $<HTMLElement>(stage, '[data-legend]');
  const ftags = [...stage.querySelectorAll<HTMLElement>('[data-ftag]')];
  const lives = [...story.querySelectorAll<HTMLElement>('[data-live]')];
  // Chapter I
  const pchips = $<HTMLElement>(story, '[data-pchips]');
  const panswer = $<HTMLElement>(story, '[data-panswer]');
  // Chapter III
  const chips = $<HTMLElement>(story, '[data-chips]');
  const views = $<HTMLElement>(story, '[data-views]');
  const maskInput = $<HTMLInputElement>(story, '[data-mask]');
  const answer = $<HTMLElement>(story, '[data-answer]');
  const provenance = $<HTMLElement>(story, '[data-provenance]');
  // Chapter IV
  const memMask = $<HTMLInputElement>(story, '[data-memmask]');
  const memAnswer = $<HTMLElement>(story, '[data-memanswer]');
  const memprov = $<HTMLElement>(story, '[data-memprov]');

  /** Position a stage label at a projected point, kept inside the stage. `cx` = label is centered on x. */
  const place = (el: HTMLElement, x: number, y: number, cx = false) => {
    const w = el.offsetWidth, W = stage.clientWidth, pad = 8;
    const left = cx ? Math.min(Math.max(x, w / 2 + pad), W - w / 2 - pad) : Math.min(Math.max(x, pad), W - w - pad);
    el.style.transform = `translate(${left}px, ${Math.max(y, pad)}px)`;
  };

  // ── State ──────────────────────────────────────────────────────────────
  let sceneIdx = 0, question = 0, masked = false, view = 0, memMasked = false, problemIdx = 0;
  let field: TokenField | undefined;
  let fstate: FieldState = { chapter: 0, query: null, pinned: false, yawDeg: 0, pitchDeg: 0, dragging: false };
  let stats: (EvidenceStats | null)[][] | null = null;
  let problemMaps: (Uint8Array | null)[] = problems.map(() => null);
  let simK = 12;
  // ── Chapter 0: the self-portrait ──
  // Pointing at the plant or the wall asks "whose left?" (use). Clicking my glasses, mouth, shirt print or the chair
  // runs the paper's grounding test on it, and the heads panel switches the 20 Visual Retrieval Heads — or 20 random
  // heads — off (verify). Answers: pipeline/mlx_vrh.py portrait.
  const bubble = $<HTMLElement>(stage, '[data-bubble]');
  const mtag = $<HTMLElement>(stage, '[data-mtag]');
  const vlines = $<SVGGElement>(stage, '[data-vlines]');
  const vgt = $<SVGPolygonElement>(stage, '[data-vgt]');
  const vpred = $<SVGPolygonElement>(stage, '[data-vpred]');
  const stageVrh = $<HTMLElement>(stage, '[data-vrh]');
  const vrhPanels = [...story.querySelectorAll<HTMLElement>('[data-vrh]')];
  const PG = 18; // portrait grid
  const PX = 28; // input pixels per token
  const probe = (fig1 as { portrait?: unknown }).portrait as PortraitProbe | undefined;
  /** Where things are on my 18×18 photo (measured on the token grid). */
  const whoseAt = (c: number, r: number) =>
    c <= 5 && (r <= 12 || c <= 2) ? 'wall' : (r <= 4 && c >= 7) || (c >= 13 && r <= 12) ? 'plant' : null;
  const whoseItems = (region: string) => {
    const control = probe?.whose.find((w) => w.region === region && w.kind === 'control');
    const perspective = probe?.whose.find((w) => w.region === region && w.kind === 'perspective');
    return control && perspective ? { control, perspective } : null;
  };
  /** The find target under a cell: the token's center lies in its hand-labeled box (with a quarter-token margin). */
  const findAt = (c: number, r: number) => {
    const x = (c + 0.5) * PX, y = (r + 0.5) * PX, m = PX / 4;
    return probe?.find.find(({ gt: [x0, y0, x1, y1] }) => x >= x0 - m && x <= x1 + m && y >= y0 - m && y <= y1 + m) ?? null;
  };
  let whose: string | null = null, whoseUntil = 0;
  let find: FindItem | null = null, cond: Condition = 'all';
  let hoverFind: FindItem | null = null;
  let ouchUntil = 0;
  /** The tokens a run's VRHs look at most, and how many of the heads still on look inside the hand-labeled box. */
  const looks = (f: FindItem) => (cond === 'random' ? f.random : cond === 'all' ? f.all : null);
  const N = 28;
  const vrhIdx = new Set((probe?.heads ?? []).map(([l, h]) => l * N + h));
  const rndIdx = new Set((probe?.randomHeads ?? []).map(([l, h]) => l * N + h));
  /** Heads switched off in a condition. The paper's random 20 can include a retrieval head (here: one). */
  const maskedIn = (c: Condition) => (c === 'vrh' ? vrhIdx : c === 'random' ? rndIdx : new Set<number>());
  const headOn = (k: number) => !maskedIn(cond).has(probe!.heads[k][0] * N + probe!.heads[k][1]);
  const onTarget = (f: FindItem, argmax: number[]) => argmax.filter((i, k) => {
    const [x0, y0, x1, y1] = f.gt, c = i % PG, r = Math.floor(i / PG);
    return headOn(k) && (c + 1) * PX > x0 && c * PX < x1 && (r + 1) * PX > y0 && r * PX < y1;
  }).length;
  const findCtx = () => {
    if (!find) return null;
    const l = looks(find);
    return { item: find, cond, onTarget: l ? onTarget(find, l.argmax) : 0, headsOn: probe!.heads.filter((_, k) => headOn(k)).length };
  };
  const quote = (t: string) => `“${esc(t).replace(/\.$/, '')}”`;
  const row = (label: string, value: string, mark = '') => `<span class="r"><i>${label}</i><span>${value}${mark}</span></span>`;
  const tick = (w: WhoseItem) => `<b class="${w.correct ? 'ok' : 'bad'}">${w.correct ? '✓' : '✗'}</b>`;
  const whoseHTML = (w: { control: WhoseItem; perspective: WhoseItem }) =>
    `<span class="t">${esc(probe!.model.split(',')[0].replace('-Instruct', ''))} · the ${w.control.region}</span>`
    + row('in the photo', quote(w.control.answer), tick(w.control))
    + row('from his side', quote(w.perspective.answer), tick(w.perspective));

  // Heads panel: the model's 784 heads as layers (rows) × heads (columns), its 20 retrieval heads lit.
  const CELL = 4, GAP = 1, SIDE = N * (CELL + GAP) - GAP;
  const gridSVG = () => {
    let s = `<svg class="grid" width="${SIDE}" height="${SIDE}" viewBox="0 0 ${SIDE} ${SIDE}" role="img" aria-label="784 attention heads, 28 layers by 28 heads; the 20 Visual Retrieval Heads are highlighted">`;
    for (let l = 0; l < N; l++)
      for (let h = 0; h < N; h++)
        s += `<rect data-h="${l * N + h}" x="${h * (CELL + GAP)}" y="${l * (CELL + GAP)}" width="${CELL}" height="${CELL}"/>`;
    return s + '</svg>';
  };
  const CONDS: [Condition, string][] = [['all', 'all heads'], ['vrh', '20 VRHs off'], ['random', '20 random off']];
  vrhPanels.forEach((p) => {
    p.innerHTML = gridSVG() + `<div class="side"><span class="t">${esc(probe?.model.split(',')[0].replace('-Instruct', '') ?? '')} · 784 heads (layer ↓ · head →)</span>`
      + `<span class="q" data-vq></span><div class="seg" role="group" aria-label="Which heads are switched off">`
      + CONDS.map(([c, label]) => `<button type="button" data-cond="${c}" aria-pressed="false">${label}</button>`).join('')
      + `</div><span class="res" data-vres aria-live="polite"></span>`
      + `<span class="key"><i class="kp"></i>model’s box &nbsp;<i class="kg"></i>hand-labeled</span></div>`;
    p.addEventListener('click', (e) => {
      const b = (e.target as HTMLElement).closest<HTMLButtonElement>('[data-cond]');
      if (!b) return;
      if (!find) find = probe?.find[0] ?? null; // nothing picked yet: start with my glasses
      cond = b.dataset.cond as Condition;
      applyFind();
    });
  });
  const stageCells = new Map([...stageVrh.querySelectorAll<SVGRectElement>('rect')].map((el) => [Number(el.dataset.h), el]));
  const renderPanels = () => {
    const r = find ? find[cond] : null;
    const v = r ? verdict(r) : 'none';
    const res = !find || !r ? '' : `<b class="${v === 'on' ? 'ok' : v === 'partly' ? 'part' : 'bad'}">${VERDICT[v]} ${esc(find.expr)}</b>${r.box ? ` · IoU ${r.iou.toFixed(2)}` : ''}`;
    const q = find ? `Q · Locate ${esc(find.expr)}.` : 'Click my glasses, mouth, shirt print or the chair.';
    vrhPanels.forEach((p) => {
      p.classList.toggle('on', !!find); // the phone copy only shows once something is being found
      p.querySelector('[data-vq]')!.innerHTML = q;
      p.querySelector('[data-vres]')!.innerHTML = res;
      p.querySelectorAll<HTMLButtonElement>('[data-cond]').forEach((b) => b.setAttribute('aria-pressed', String(!!find && b.dataset.cond === cond)));
      const masked = find ? maskedIn(cond) : new Set<number>();
      p.querySelectorAll<SVGRectElement>('rect').forEach((el) => {
        const i = Number(el.dataset.h);
        el.setAttribute('class', vrhIdx.has(i) ? (masked.has(i) ? 'v off' : 'v') : masked.has(i) ? 'r off' : '');
      });
    });
  };
  /** Show the current find on my photo: the VRHs' attention (while they're on) and redraw everything. */
  const applyFind = () => {
    const l = find ? looks(find) : null;
    field?.setPortraitAttention(l ? fromU8(Uint8Array.from(l.vrh)) : null);
    vlines.replaceChildren(...(l ? l.argmax.map(() => document.createElementNS('http://www.w3.org/2000/svg', 'path')) : []));
    whose = null; whoseUntil = 0;
    renderPanels(); renderLive(); runOverlays();
  };
  const clearFind = () => { if (!find) return; find = null; cond = 'all'; applyFind(); };

  // Overlays follow the photo while anything is on; the loop stops itself when nothing is.
  let overlayRAF = 0;
  const quad = (pts: { x: number; y: number }[] | null) => (pts ? pts.map((v) => `${v.x.toFixed(1)},${v.y.toFixed(1)}`).join(' ') : '');
  const renderOverlays = () => {
    const f = field;
    const now = performance.now();
    const on0 = fstate.chapter === 0 && !!f;
    if (whose && whoseUntil && now > whoseUntil) { whose = null; whoseUntil = 0; renderLive(); }
    const w = on0 && whose && !find ? whoseItems(whose) : null; // one readout at a time: a find wins
    const ouch = on0 && now < ouchUntil;
    const say = ouch ? 'hey — my glasses!' : w ? `It’s on my ${w.perspective.gt}.` : null;
    bubble.hidden = !say;
    if (say && f) {
      if (bubble.textContent !== say) bubble.textContent = say;
      const p = f.portraitScreen(10, 5, true);
      if (p) place(bubble, p.x, p.y, true);
    }
    mtag.hidden = !w;
    if (w && f) {
      const html = whoseHTML(w);
      if (mtag.dataset.key !== whose) { mtag.dataset.key = whose!; mtag.innerHTML = html; }
      const p = f.portraitScreen(whose === 'plant' ? 15 : 2, 8, true); // beside my head, below my speech bubble
      if (p) place(mtag, p.x, p.y, true);
    }
    // Heads panel under my photo (desktop), left-aligned with it.
    stageVrh.hidden = !on0;
    const frame = on0 && f ? f.portraitQuad(0, 0, PG, PG) : null;
    if (frame) place(stageVrh, frame[3].x, Math.max(frame[2].y, frame[3].y) + 14);
    // The hand-labeled box of what's found (or would be, on hover) and the model's box for the current condition.
    const shown = on0 ? (find ?? hoverFind) : null;
    const box = (b: number[] | null) => (b && f ? f.portraitQuad(b[0] / PX, b[1] / PX, b[2] / PX, b[3] / PX) : null);
    const gt = shown ? box(shown.gt) : null;
    vgt.classList.toggle('on', !!gt);
    if (gt) vgt.setAttribute('points', quad(gt));
    const pred = on0 && find ? box(find[cond].box) : null;
    vpred.classList.toggle('on', !!pred);
    if (pred) vpred.setAttribute('points', quad(pred));
    // Lines from each lit head in the panel to the token it looks at most (only while the heads are on).
    const l = on0 && find ? looks(find) : null;
    const paths = vlines.children;
    if (l && f && stageVrh.offsetParent) {
      const sr = stage.getBoundingClientRect();
      l.argmax.forEach((tok, k) => {
        const [lay, head] = probe!.heads[k];
        const cell = stageCells.get(lay * N + head)?.getBoundingClientRect();
        const p = f.portraitScreen(tok % PG, Math.floor(tok / PG));
        if (!paths[k]) return;
        if (!cell || !p || !headOn(k)) { paths[k].removeAttribute('d'); return; } // a switched-off head draws no line
        const x0 = cell.left + cell.width / 2 - sr.left, y0 = cell.top + cell.height / 2 - sr.top;
        paths[k].setAttribute('d', `M${x0.toFixed(1)},${y0.toFixed(1)} Q${x0.toFixed(1)},${((y0 + p.y) / 2).toFixed(1)} ${p.x.toFixed(1)},${p.y.toFixed(1)}`);
      });
    }
    vlines.style.display = l && stageVrh.offsetParent ? '' : 'none';
    return !!(on0 || say || w);
  };
  const runOverlays = () => {
    if (overlayRAF) return;
    const tick = () => { overlayRAF = renderOverlays() ? requestAnimationFrame(tick) : 0; };
    overlayRAF = requestAnimationFrame(tick);
  };
  const onPortraitHover = (cell: [number, number] | null) => {
    hoverFind = cell ? findAt(cell[0], cell[1]) : null;
    const region = cell && !hoverFind ? whoseAt(cell[0], cell[1]) : null;
    stage.classList.toggle('knockable', !!hoverFind);
    if (!whoseUntil && region !== whose) { whose = region; renderLive(); }
    runOverlays();
  };
  const onPortraitTap = (cell: [number, number], pointerType: string) => {
    const f = findAt(cell[0], cell[1]);
    if (f) {
      if (find !== f) { find = f; cond = 'all'; }
      applyFind();
      return true;
    }
    const region = whoseAt(cell[0], cell[1]);
    if (region && pointerType !== 'mouse') {
      // No hover on touch: a tap on the plant or the wall shows its answers for a few seconds.
      clearFind();
      whose = region; whoseUntil = performance.now() + 4000;
      renderLive(); runOverlays();
      return true;
    }
    clearFind(); // a plain poke elsewhere puts the photo back to just me
    return false;
  };
  renderPanels();
  const sceneIndexOf = (id: string) => Math.max(0, scenes.findIndex((s) => s.id === id));

  // ── Live readouts: one sentence per chapter saying what the current visual means ──
  const problemContext = () => {
    const item = problems[problemIdx];
    if (!item) return null;
    const s = scenes[sceneIndexOf(item.scene)];
    const w = unionCoverage(item.boxes, s.grid[0], s.grid[1]);
    const map = problemMaps[problemIdx];
    const other = problems.find((p) => p.scene === item.scene && p.kind === 'control' && p !== item);
    return { item, other, inBoxes: map ? inside(map, w) : null, area: w.reduce((a, b) => a + b, 0) / w.length };
  };
  const renderLive = () => {
    const ctx = {
      scene: scenes[sceneIdx], question, masked, view,
      viewLabels: fig1.vrh.views, k: fig1.vrh.k, totalHeads: 28 * 28, simLayer: fig1.vrh.simLayer, simK,
      stats: stats?.[sceneIdx]?.[question] ?? null,
      retained: field?.retainedCount(0) ?? 0,
      memMasked, video, papers, portraitTokens: PG * PG,
      memory: { t: playT, kept: keptCount(), end: T_END, asked: asked() },
      whose: fstate.chapter === 0 && whose && !find ? whoseItems(whose) : null,
      find: fstate.chapter === 0 ? findCtx() : null,
      problem: problemContext(),
    };
    // Each card narrates its own chapter; hover/drag only apply to the chapter on screen.
    lives.forEach((el) => {
      const ch = Number(el.dataset.live);
      const st = ch === fstate.chapter ? fstate : { ...fstate, query: null, yawDeg: 0, pitchDeg: 0, dragging: false };
      const html = narrate(ch, st, ctx);
      if (el.innerHTML !== html) el.innerHTML = html;
    });
  };

  // ── Chapter chrome: which scene, question label, picker and attention each chapter shows ──
  const syncChapter = () => {
    const ch = fstate.chapter;
    const item = problems[problemIdx];
    if (ch === 1 && item) {
      // Chapter I drives the scene itself and shows the probe's attention and boxes.
      field?.setScene(sceneIndexOf(item.scene));
      const s = scenes[sceneIndexOf(item.scene)];
      const map = problemMaps[problemIdx];
      field?.setOverride({ attn: map ? fromU8(map) : new Float32Array(s.grid[0] * s.grid[1]), boxes: item.boxes });
    } else {
      field?.setOverride(null);
      if (field && field.sceneIndex !== sceneIdx) { field.setScene(sceneIdx); field.setQuestion(question); }
    }
    const shown = ch === 1 && item ? scenes[sceneIndexOf(item.scene)] : scenes[sceneIdx];
    qtext.textContent = ch === 4 ? video.question : ch === 1 && item ? item.q : scenes[sceneIdx].questions[question].q;
    credit.textContent = ch === 0 ? `Chanho Park · ${PG}×${PG} tokens`
      : ch === 4 ? video.credit : `${shown.credit} · ${shown.grid[0]}×${shown.grid[1]} tokens`;
    picker.classList.toggle('off', ch === 0 || ch === 1 || ch === 4);
    stage.classList.toggle('c0', ch === 0);
    if (ch !== 0 && (whose || find || hoverFind)) {
      whose = null; whoseUntil = 0; hoverFind = null; stage.classList.remove('knockable');
      clearFind();
      renderLive();
    }
    if (ch === 0) runOverlays(); // the heads panel and its lines follow the photo while it's on screen
    // Chapter IV: fetch the clip on the way there, play it once on arrival, and pause it on leaving.
    if (ch >= 3) loadClip();
    if (ch === 4 && !autoplayed && !reduced && field) { autoplayed = true; play(); }
    if (ch !== 4 && playing) pause();
    if (legend.dataset.ch !== String(ch)) { legend.dataset.ch = String(ch); legend.innerHTML = legendHTML(LEGEND[ch]); }
  };

  // ── Chapter I: the question ────────────────────────────────────────────
  const choose = (i: number) => { problemIdx = i; renderProblem(); syncChapter(); renderLive(); };
  const renderProblem = () => {
    // One tab per photo, then that photo's questions (camera frame vs. another viewpoint).
    const groups = [...new Set(problems.map((p) => p.scene))];
    const curScene = problems[problemIdx]?.scene;
    const tabs = document.createElement('div');
    tabs.className = 'ptabs';
    tabs.replaceChildren(...groups.map((g) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = g.replace('-', ' ');
      b.setAttribute('aria-pressed', String(g === curScene));
      b.addEventListener('click', () => choose(problems.findIndex((p) => p.scene === g)));
      return b;
    }));
    pchips.replaceChildren(tabs, ...problems.flatMap((p, i) => {
      if (p.scene !== curScene) return [];
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = p.q;
      b.dataset.kind = p.kind;
      b.setAttribute('aria-pressed', String(i === problemIdx));
      b.addEventListener('click', () => choose(i));
      return [b];
    }));
    const p = problems[problemIdx];
    panswer.innerHTML = !p ? '' : p.answer === undefined
      ? `<span class="same">model answer pending</span>`
      : `${esc(p.answer)}<span class="${p.correct ? 'ok' : 'bad'}">${p.correct ? '✓ correct' : `✗ truth: ${esc(p.gt)}`}</span>`;
  };

  // ── Chapter III: verify ────────────────────────────────────────────────
  const renderProbe = () => {
    const s = scenes[sceneIdx];
    chips.replaceChildren(...s.questions.map((q, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = q.q;
      b.setAttribute('aria-pressed', String(i === question));
      b.addEventListener('click', () => { question = i; field?.setQuestion(i); renderProbe(); syncChapter(); });
      return b;
    }));
    const q = s.questions[question];
    answer.innerHTML = !masked
      ? `${esc(q.answer)}<span class="ok">all heads</span>`
      : q.changed === false
        ? `${esc(q.masked)}<span class="same">unchanged</span>`
        : `${esc(q.masked)}<span class="bad">${esc(q.answer)}</span>`;
    views.querySelectorAll('button').forEach((b, i) => {
      b.setAttribute('aria-pressed', String(i === view));
      b.disabled = masked;
    });
    renderLive();
  };
  views.replaceChildren(...fig1.vrh.views.map((label, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.addEventListener('click', () => { view = i; field?.setView(i); renderProbe(); });
    return b;
  }));
  maskInput.addEventListener('change', () => { masked = maskInput.checked; field?.setMasked(masked); renderProbe(); });

  // ── Chapter IV: over time ──────────────────────────────────────────────
  // The clip plays as the live plate of tokens; each of the five frames the model is given drops into memory when
  // the playhead reaches it. The question is asked at the last frame, so its attention and answer appear only then.
  const clip = $<HTMLVideoElement>(stage, '[data-clip]');
  const playBtn = $<HTMLButtonElement>(story, '[data-play]');
  const seek = $<HTMLInputElement>(story, '[data-seek]');
  const tnow = $<HTMLElement>(story, '[data-tnow]');
  const thumbs = [...story.querySelectorAll<HTMLButtonElement>('[data-thumb-frame]')];
  const T_END = video.frames[video.frames.length - 1].t;
  let playT = T_END, playing = false, autoplayed = false, clipLoading = false;
  const asked = () => playT >= T_END - 1e-3;
  const keptCount = () => video.frames.filter((f) => playT >= f.t - 1e-3).length;
  const renderMemory = () => {
    memAnswer.innerHTML = !asked()
      ? `<span class="same">the question comes at t = ${T_END.toFixed(1)} s</span>`
      : memMasked && video.masked
        ? `${esc(video.masked)}<span class="bad">${esc(video.answer)}</span>`
        : `${esc(video.answer)}<span class="ok">all heads</span>`;
    memMask.disabled = !asked();
  };
  memMask.addEventListener('change', () => { memMasked = memMask.checked; field?.setMemoryMasked(memMasked); renderMemory(); renderLive(); });
  const loadClip = () => { if (clipLoading) return; clipLoading = true; clip.preload = 'auto'; clip.load(); };
  const setT = (t: number) => {
    playT = Math.min(T_END, Math.max(0, t));
    field?.setPlayhead(playT);
    seek.value = String(playT);
    tnow.textContent = `${playT.toFixed(1)} s`;
    thumbs.forEach((b, i) => b.classList.toggle('kept', playT >= video.frames[i].t - 1e-3));
    renderMemory(); renderLive();
  };
  const setPlaying = (on: boolean) => {
    playing = on;
    playBtn.textContent = on ? '❚❚' : asked() ? '↺' : '▶';
    playBtn.setAttribute('aria-label', on ? 'Pause' : asked() ? 'Play the video again' : 'Play the video');
  };
  const follow = () => {
    if (!playing) return;
    if (clip.currentTime >= T_END - 0.02 || clip.ended) { clip.pause(); setT(T_END); setPlaying(false); return; }
    setT(clip.currentTime);
    requestAnimationFrame(follow);
  };
  const play = () => {
    loadClip();
    if (asked()) { clip.currentTime = 0; setT(0); } // from the start again
    clip.play().then(() => { setPlaying(true); requestAnimationFrame(follow); }).catch(() => setPlaying(false));
  };
  const pause = () => { clip.pause(); setPlaying(false); };
  const seekTo = (t: number) => {
    loadClip();
    if (playing) pause();
    clip.currentTime = Math.min(t, T_END);
    setT(t);
    setPlaying(false);
  };
  playBtn.addEventListener('click', () => (playing ? pause() : play()));
  seek.addEventListener('input', () => seekTo(Number(seek.value)));
  thumbs.forEach((b, i) => b.addEventListener('click', () => seekTo(video.frames[i].t)));
  // It builds up from the start when you arrive; with reduced motion it just shows the end (all frames, question asked).
  setT(reduced ? T_END : 0); setPlaying(false);

  // ── Picker (Chapters II–III) ───────────────────────────────────────────
  const setScene = (i: number) => {
    sceneIdx = i; question = 0;
    pickerBtns.forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
    field?.setScene(i);
    renderProbe();
    syncChapter();
  };
  pickerBtns.forEach((b, i) => b.addEventListener('click', () => setScene(i)));

  renderProblem(); renderProbe(); renderMemory(); syncChapter();

  if (!webglAvailable()) { stage.classList.add('no-webgl'); return; }

  // ── 3D field ───────────────────────────────────────────────────────────
  const [{ TokenField: Field, makeTexture }, imgs, frameImgs] = await Promise.all([
    import('./field'),
    Promise.all(scenes.map((s) => loadImage(s.image))),
    Promise.all(video.frames.map((f) => loadImage(f.image))),
  ]);
  const portraitScene = scenes.find((s) => s.id === 'portrait')!;
  const portraitImg = imgs[scenes.indexOf(portraitScene)];
  const canvas = $<HTMLCanvasElement>(stage, '[data-canvas]');
  const f3d = new Field(canvas, scenes, imgs.map((img) => ({ tex: makeTexture(img), img })), {
    hover: (h) => {
      if (!h) { ctag.hidden = true; return; }
      ctag.hidden = false;
      ctag.textContent = `query tok (${h.row}, ${h.col})`;
    },
    query: (p) => {
      qlabel.hidden = !p;
      if (p) place(qlabel, p.x, Math.max(p.y, qlabel.offsetHeight + 26), true); // label sits above its anchor
    },
    state: (st) => {
      const changed = st.chapter !== fstate.chapter;
      fstate = st;
      if (changed) syncChapter();
      renderLive();
    },
    boxes: (list) => {
      boxtags.forEach((el, i) => {
        const b = list.find((x) => x.i === i);
        el.hidden = !b;
        if (!b) return;
        const item = problems[problemIdx];
        el.textContent = fstate.chapter === 1 && item ? item.labels[i] ?? '' : 'evidence region · hand-labeled';
        place(el, b.x, b.y);
      });
    },
    frames: (list) => {
      ftags.forEach((el, i) => {
        const f = list[i];
        el.hidden = !f;
        if (!f) return;
        // The live video while it plays; then each kept frame by number and time ("now" = the last one, once reached).
        const now = f.frame === -1 || (f.frame === video.frames.length - 1 && asked());
        const narrow = innerWidth < 900; // phones: short labels, the stack runs to the screen's edge
        el.textContent = f.frame === -1 ? `live · ${playT.toFixed(1)} s`
          : `${narrow ? '' : 'Frame '}${f.frame + 1} · ${video.frames[f.frame].t.toFixed(1)} s${now ? ' · now' : ''}`;
        el.classList.toggle('now', now);
        el.style.transform = `translate(${f.x}px, ${f.y}px)`;
      });
    },
    ouch: () => { ouchUntil = performance.now() + 1600; runOverlays(); },
    portraitHover: onPortraitHover,
    tap: onPortraitTap,
  }, reduced,
  { data: video, textures: frameImgs.map(makeTexture), clip },
  {
    data: portraitScene, texture: makeTexture(portraitImg),
    // Measured on the 18×18 portrait (see pipeline notes): head ellipse, eye centers, glasses tokens.
    head: [9.6, 8.0, 3.3, 3.8], eyes: [0.492, 0.458, 0.606, 0.457], eyeR: 0.024,
    glasses: [8 * PG + 8, 8 * PG + 9, 8 * PG + 10, 8 * PG + 11],
  });
  field = f3d;
  if (import.meta.env.DEV) (window as unknown as { __tokenField: unknown }).__tokenField = f3d;
  f3d.setScene(sceneIdx); f3d.setQuestion(question); f3d.setMasked(masked); f3d.setView(view); f3d.setPlayhead(playT);
  syncChapter();

  loadAttention('/fig1/attention.json')
    .then(({ scenes: real, video: vmaps, problem: pmaps }) => {
      f3d.setAttentionData(real);
      if (vmaps) {
        f3d.setVideoAttention(vmaps);
        memprov.textContent = `Real attention of the top-${fig1.vrh.k} retrieval heads from the answer to each frame, minus a neutral-prompt baseline; answers are model outputs.`;
      }
      if (pmaps) problemMaps = problems.map((_, i) => pmaps[i] ?? null);
      stats = evidenceStats(scenes, real);
      simK = Object.values(real)[0]?.simK ?? simK;
      const v = fig1.vrh;
      provenance.textContent = `Real attention of the top-${v.k} Visual Retrieval Heads in ${v.model} (scored on ${v.source}). Answers and masked answers are model outputs; pointing shows layer-${v.simLayer} token similarity.`;
      syncChapter();
      renderLive();
    })
    .catch(() => { /* keep illustrative maps */ });

  canvas.addEventListener('pointermove', (e) => {
    const r = stage.getBoundingClientRect();
    place(ctag, e.clientX - r.left + 14, e.clientY - r.top + 14);
  });

  // Theme changes (toggle or OS) → re-read CSS tokens.
  new MutationObserver(() => f3d.readTheme()).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => f3d.readTheme());

  // Scroll → fractional chapter index, interpolating between chapter centers crossing mid-viewport.
  const onScroll = () => {
    const mid = innerHeight * 0.5;
    const centers = chapters.map((el) => { const r = el.getBoundingClientRect(); return r.top + r.height / 2; });
    let f = 0;
    for (let i = 0; i < centers.length; i++) {
      if (centers[i] > mid) break;
      const next = centers[i + 1];
      f = next === undefined ? i : i + Math.min(1, (mid - centers[i]) / (next - centers[i]));
    }
    f3d.setChapter(f);
    const active = Math.round(f);
    rail.forEach((li, i) => li.classList.toggle('on', i === active));
  };
  addEventListener('scroll', onScroll, { passive: true });
  addEventListener('resize', onScroll);
  onScroll();

  // Render only while the stage is on screen and the tab is visible.
  let onScreen = true;
  new IntersectionObserver(([e]) => {
    onScreen = e.isIntersecting;
    onScreen && !document.hidden ? f3d.start() : f3d.stop();
  }).observe(story);
  document.addEventListener('visibilitychange', () => (onScreen && !document.hidden ? f3d.start() : f3d.stop()));
  f3d.start();

  // Smooth scrolling for the story page.
  if (!reduced) {
    const { default: Lenis } = await import('lenis');
    new Lenis({ autoRaf: true, lerp: 0.12 });
  }
}
