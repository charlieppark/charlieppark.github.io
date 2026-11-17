// Turns the field's interaction state into one sentence per chapter that says what the
// visual means. Every number shown is computed from the data on the page.
import type { Box, SceneAttention, SceneData, VideoData } from './attention';
export { coverage, inside };
import type { FieldState } from './field';

const pct = (v: number) => `${Math.round(v * 100)}%`;
const esc = (t: string) => t.replace(/[<>&]/g, '');
/** Answer text for quoting inside a sentence (no trailing period). */
const bare = (t: string) => esc(t).replace(/\.$/, '');

/** Per-token coverage of a normalized box on the token grid. */
function coverage(box: Box, gw: number, gh: number) {
  const [x0, y0, x1, y1] = box;
  const w = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++)
    for (let x = 0; x < gw; x++) {
      const ox = Math.max(0, Math.min((x + 1) / gw, x1) - Math.max(x / gw, x0)) * gw;
      const oy = Math.max(0, Math.min((y + 1) / gh, y1) - Math.max(y / gh, y0)) * gh;
      w[y * gw + x] = ox * oy;
    }
  return w;
}

/** Share of a map's attention mass inside the box (max-normalization cancels out). */
function inside(map: Uint8Array, w: Float32Array) {
  let a = 0, t = 0;
  for (let i = 0; i < map.length; i++) { a += map[i] * w[i]; t += map[i]; }
  return t > 0 ? a / t : 0;
}

export interface EvidenceStats {
  area: number;      // fraction of the image covered by the evidence box
  views: number[];   // share inside the box per head view (top-k mean, individual heads)
  masked: number;    // share inside the box for the remaining heads, with retrieval heads masked
}

export function evidenceStats(scenes: SceneData[], real: Record<string, SceneAttention>) {
  return scenes.map((s) => s.questions.map((q, qi) => {
    const r = real[s.id]?.questions[qi];
    const w = coverage(q.box, s.grid[0], s.grid[1]);
    const area = w.reduce((a, b) => a + b, 0) / w.length;
    return r ? { area, views: r.vrh.map((m) => inside(m, w)), masked: inside(r.rest, w) } : null;
  }));
}

export interface PaperMeta { id: string; title: string; venue: string; year: number; thumb: string }

/** Chapter I probe: one question in the camera's frame, one from another viewpoint. */
export interface ProblemItem {
  scene: string;
  kind: 'control' | 'perspective';
  q: string;
  gt: string;
  boxes: Box[];
  labels: string[];
  answer?: string;
  correct?: boolean;
}

/** Chapter 0, "move": an object on my photo asked about in the camera's frame and from my side. */
export interface WhoseItem { region: string; kind: 'control' | 'perspective'; q: string; gt: string; answer: string; correct: boolean }
/** Chapter 0, "click": the paper's grounding test on my photo, with all heads / VRHs masked / random heads masked. */
export type Condition = 'all' | 'vrh' | 'random';
export interface FindRun { text: string; box: [number, number, number, number] | null; iou: number }
export interface FindItem {
  id: string; expr: string; gt: [number, number, number, number];   // hand-labeled box, input pixels (504 × 504)
  /** `vrh`: the 20 VRHs' mean attention while answering; `argmax`: the token each of them looks at most. */
  all: FindRun & { vrh: number[]; argmax: number[] }; vrh: FindRun; random: FindRun & { vrh: number[]; argmax: number[] };
}
export interface PortraitProbe {
  model: string; heads: [number, number][]; randomHeads: [number, number][]; prompt: string;
  whose: WhoseItem[]; find: FindItem[];
}
/** RefCOCO counts IoU ≥ 0.5 as found; a box that still overlaps the object less than that is "partly on". */
export const verdict = (r: FindRun) => (!r.box ? 'none' : r.iou >= 0.5 ? 'on' : r.iou >= 0.1 ? 'partly' : 'off');
export const VERDICT = { on: '✓ on', partly: '◐ partly on', off: '✗ off', none: '✗ no box for' } as const;
/** Overlap of two boxes (x0, y0, x1, y1). */
export const boxIoU = (a: number[] | null, b: number[] | null) => {
  if (!a || !b) return 0;
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const area = (r: number[]) => Math.max(0, r[2] - r[0]) * Math.max(0, r[3] - r[1]);
  return (ix * iy) / Math.max(1e-9, area(a) + area(b) - ix * iy);
};

export interface Context {
  scene: SceneData;
  question: number;
  masked: boolean;
  view: number;
  viewLabels: string[];
  k: number;
  totalHeads: number;
  simLayer: number;
  simK: number;
  stats: EvidenceStats | null;
  /** Tokens kept from the oldest video frame. */
  retained: number;
  video: VideoData;
  memMasked: boolean;
  /** Chapter IV playback: playhead (s), frames kept so far, the last frame's time, and whether the question is asked. */
  memory: { t: number; kept: number; end: number; asked: boolean };
  papers: PaperMeta[];
  portraitTokens: number;
  /** Chapter 0: the object pointed at, or the one the model was asked to find (which heads are off, how many look at it). */
  whose: { control: WhoseItem; perspective: WhoseItem } | null;
  find: { item: FindItem; cond: Condition; onTarget: number; headsOn: number } | null;
  /** Chapter I: the selected probe, its sibling, and the share of attention inside its boxes. */
  problem: { item: ProblemItem; other?: ProblemItem; inBoxes: number | null; area: number } | null;
}

function queryLine(st: FieldState, c: Context) {
  const [col, row] = st.query!;
  return `Pointing at token <b>(${row}, ${col})</b>. Outlined: the <b>${c.simK}</b> tokens whose layer-${c.simLayer} representations are closest to it — what the model treats as the same stuff. (Similarity, not attention.)${st.pinned ? ' <i>Pinned — tap it again to release.</i>' : ''}`;
}

const side = (d: number, pos: string, neg: string) => `${Math.abs(d)}° ${d >= 0 ? pos : neg}`;

export function narrate(chapter: number, st: FieldState, c: Context): string {
  const [gw, gh] = c.scene.grid;
  const n = gw * gh;
  switch (chapter) {
    case 0: {
      if (c.find) {
        const { item: f, cond, onTarget, headsOn } = c.find;
        const what = esc(f.expr);
        const box = (r: FindRun) => {
          const v = verdict(r);
          const mark = v === 'on' ? '✓ ' : v === 'off' ? '✗ ' : '';
          return v === 'none' ? 'no box at all' : `a box ${v === 'partly' ? 'partly on' : v} ${what} ${mark}(IoU ${r.iou.toFixed(2)})`;
        };
        const paper = 'In the paper, masking these 20 drops RefCOCO grounding from <b>89%</b> to <b>9.5%</b>.';
        if (cond === 'all')
          return `Asked to locate ${what}, the model draws <b>${box(f.all)}</b>. The lines run from its <b>20</b> Visual Retrieval Heads — 2.6% of 784 — to where each looks most: <b>${onTarget}</b> of 20 land on it. <i>Now switch them off.</i>`;
        if (cond === 'vrh') {
          const v = verdict(f.vrh), moves = v === 'off' || v === 'none' ? 'jumps away' : v === 'partly' ? 'drifts' : 'shifts';
          return boxIoU(f.vrh.box, f.all.box) < 0.5
            ? `With those 20 heads off, the model still answers in valid JSON — but its box ${moves}: <b>${box(f.vrh)}</b>. It can still generate; it no longer looks in the right place. ${paper}`
            : `With those 20 heads off, the box stays put this time: <b>${box(f.vrh)}</b>. ${paper}`;
        }
        const stay = boxIoU(f.random.box, f.all.box);
        const shared = 20 - headsOn;
        const still = `${onTarget} of the ${headsOn} retrieval heads still on look at ${what}${shared ? ` (${shared === 1 ? 'one' : shared} of the random 20 ${shared === 1 ? 'is' : 'are'} itself a retrieval head)` : ''}`;
        return stay >= 0.5
          ? `Twenty random heads off instead: the box stays where it was (${stay.toFixed(2)} overlap with the all-heads box), and ${still}. Random heads move RefCOCO accuracy by less than a point — it’s these 20 that do the looking.`
          : `Twenty random heads off instead: here the box shifts too (${stay.toFixed(2)} overlap with the all-heads box). On RefCOCO, random heads move accuracy by less than a point on average.`;
      }
      if (c.whose) {
        const { control: a, perspective: b } = c.whose;
        return `In the photo, the model puts the ${b.region} on the <b>${bare(a.answer)}</b> ${a.correct ? '✓' : '✗'}. Asked from my side, it says <b>“${bare(b.answer)}”</b> ${b.correct ? '✓' : `✗ — it’s on my ${b.gt}`}. It sees the ${b.region}, but can’t use it from my point of view.`;
      }
      return `Hi — that’s me, in <b>${c.portraitTokens}</b> tokens, the way a vision-language model takes me in. Point at the plant or the wall to ask whose left it is; click my glasses, mouth, shirt print or the chair to have the model find it — then switch off the heads that do the looking.`;
    }

    case 1: {
      if (st.query) return queryLine(st, c);
      const pr = c.problem;
      if (!pr) return '';
      const { item, other, inBoxes, area } = pr;
      if (item.answer === undefined) return `Two questions about the same photo — one in the camera’s frame, one from another point of view. (Model answers pending.)`;
      const who = `the ${item.labels.join(' and the ')}`;
      const lift = inBoxes !== null && area > 0 ? inBoxes / area : null;
      const attn = inBoxes === null || lift === null ? '' :
        ` Retrieval-head attention on ${who}: <b>${pct(inBoxes)}</b> (they cover ${pct(area)} of the image${lift < 1.5 ? ' — no more than chance' : `, ${lift.toFixed(1)}× chance`}).`;
      if (item.kind === 'control')
        return item.correct
          ? `In the camera’s frame the model gets it: <b>“${bare(item.answer)}”</b> ✓ — the layout is in the image and the model can read it.${attn} <i>Now ask from another point of view.</i>`
          : `Even in the camera’s frame the model answers <b>“${bare(item.answer)}”</b> ✗ (truth: ${item.gt}).${attn}`;
      if (item.correct) return `From another point of view the model still gets it: <b>“${bare(item.answer)}”</b> ✓.${attn}`;
      const evidence = other?.correct ? ` Asked in the camera’s frame it was right (“${bare(other.answer ?? '')}” ✓), so the information is there —` : ' The information is in the photo —';
      return `Same photo, other viewpoint: <b>“${bare(item.answer)}”</b> ✗ (truth: <b>${item.gt}</b>) — answered as if from the camera.${evidence} it just can’t use it from another point of view.${attn}`;
    }

    case 2: {
      if (st.query) return queryLine(st, c);
      return `Seen from <b>${side(st.yawDeg, 'right', 'left')}</b>, <b>${side(st.pitchDeg, 'above', 'below')}</b> the photo’s camera. Every patch is an original token, pixels unchanged — it only moved to where its depth puts it and turned to face you. That is <i>token warping</i>: a nearby view built from the same tokens. The gaps are what the original camera never saw. <i>Drag to move the viewpoint.</i>`;
    }

    case 3: {
      if (st.query) return queryLine(st, c);
      const q = c.scene.questions[c.question];
      const s = c.stats;
      if (!s) return `The question attends to image tokens through the retrieval heads.`;
      const share = (100 * c.k) / c.totalHeads;
      if (c.masked) {
        const result = q.changed === false
          ? `the answer holds (“${bare(q.masked)}”) — this question survives without them.`
          : `the answer breaks: <b>${bare(q.answer)} → ${bare(q.masked)}</b>.`;
        return `Masked <b>${c.k}</b> of ${c.totalHeads} heads (<b>${share.toFixed(1)}%</b>). The remaining heads put only <b>${pct(s.masked)}</b> of their attention in the evidence region, and ${result}`;
      }
      const whose = c.view === 0 ? `the top-${c.k} retrieval heads’` : `head ${c.viewLabels[c.view]}’s`;
      const lift = s.area > 0 ? s.views[c.view] / s.area : 0;
      return `<b>${pct(s.views[c.view])}</b> of ${whose} attention lands in the evidence region, which covers <b>${pct(s.area)}</b> of the image (<b>${lift.toFixed(0)}×</b> chance).${c.view === 0 ? ' Now mask them.' : ''}`;
    }

    case 4: {
      const v = c.video;
      const [vw, vh] = v.grid;
      const mem = c.memory;
      const span = (mem.end - v.frames[0].t).toFixed(1);
      if (!mem.asked)
        return `<b>t = ${mem.t.toFixed(1)} s</b>. The video streams in as tokens, and every 1.5 s a frame is kept: <b>${mem.kept}</b> of ${v.frames.length} so far — ${mem.kept * vw * vh} tokens in memory, each older than the last. At t = ${mem.end.toFixed(1)} s the question asks about Frame 1.`;
      if (c.memMasked && v.masked)
        return `With the ${c.k} retrieval heads masked, the model answers <b>“${bare(v.masked)}”</b> — where the ball is <i>now</i>, not where it was. A stale state passed off as memory. (The lifted tokens drop to show what those heads carried.)`;
      const m = v.mass;
      const first = m ? `<b>${pct(m[0])}</b> of the question’s attention reaches back to it, ${span} s into the past` : `The question reaches back to it, ${span} s into the past`;
      return `Asked at t = ${mem.end.toFixed(1)} s about Frame 1, ${first}${m ? ` (an even split would be ${pct(1 / m.length)})` : ''}, and its <b>${c.retained}</b> strongest of ${vw * vh} tokens are lifted out. The rest of each past frame is stale and drifts away. What to keep — and when to revise it — is the question I’m working on now.`;
    }
  }
  void n;
  return '';
}
