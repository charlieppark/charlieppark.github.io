// Attention maps for Fig. 1, one value per visual token in [0, 1].
//
// These are ILLUSTRATIVE stand-ins built from hand-labeled evidence boxes and
// token colors. They share the shape of the real data (one map per token grid),
// so the extracted retrieval-head attention can replace them without UI changes.

export type Box = [number, number, number, number]; // x0, y0, x1, y1 (normalized)

export interface Question {
  q: string;
  answer: string;
  masked: string;
  /** Whether masking the retrieval heads changed the model's answer. */
  changed?: boolean;
  box: Box;
}

export interface SceneData {
  id: string;
  image: string;
  grid: [number, number];
  credit: string;
  depth: number[];
  questions: Question[];
}

/** Deterministic per-token noise so maps don't flicker between renders. */
const hash = (i: number, seed: number) => {
  const x = Math.sin(i * 127.1 + seed * 311.7) * 43758.5453;
  return x - Math.floor(x);
};

const normalize = (a: Float32Array) => {
  let max = 0;
  for (const v of a) max = Math.max(max, v);
  if (max > 0) for (let i = 0; i < a.length; i++) a[i] /= max;
  return a;
};

/** One 3x3 box-blur pass; softens hard box edges into attention-like falloff. */
function blur(a: Float32Array, gw: number, gh: number) {
  const out = new Float32Array(a.length);
  for (let y = 0; y < gh; y++)
    for (let x = 0; x < gw; x++) {
      let s = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= gw || yy >= gh) continue;
          const w = dx === 0 && dy === 0 ? 2 : 1;
          s += a[yy * gw + xx] * w; n += w;
        }
      out[y * gw + x] = s / n;
    }
  return out;
}

/** Retrieval-head attention for a question: concentrated on the evidence box. */
export function evidenceAttention(scene: SceneData, qi: number): Float32Array {
  const [gw, gh] = scene.grid;
  const [x0, y0, x1, y1] = scene.questions[qi].box;
  let a = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++)
    for (let x = 0; x < gw; x++) {
      // Fraction of the token cell covered by the box.
      const ox = Math.max(0, Math.min((x + 1) / gw, x1) - Math.max(x / gw, x0)) * gw;
      const oy = Math.max(0, Math.min((y + 1) / gh, y1) - Math.max(y / gh, y0)) * gh;
      a[y * gw + x] = ox * oy;
    }
  a = blur(a, gw, gh);
  for (let i = 0; i < a.length; i++) a[i] = a[i] * (0.75 + 0.25 * hash(i, qi + 1)) + 0.035 * hash(i, 7);
  return normalize(a);
}

/** With retrieval heads masked: evidence is lost; mass falls to a sink token and scattered noise. */
export function maskedAttention(scene: SceneData, qi: number): Float32Array {
  const [gw, gh] = scene.grid;
  const a = new Float32Array(gw * gh);
  for (let i = 0; i < a.length; i++) {
    const h = hash(i, 31 + qi);
    a[i] = 0.08 * h + (h > 0.965 ? 0.45 : 0);
  }
  a[0] = 1; // attention sink at the first visual token
  return a;
}

/** Per-token mean colors (sRGB 0..1), sampled once from the scene texture. */
export function tokenColors(img: HTMLImageElement, gw: number, gh: number): Float32Array {
  const c = document.createElement('canvas');
  c.width = gw; c.height = gh;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, gw, gh);
  const px = ctx.getImageData(0, 0, gw, gh).data;
  const out = new Float32Array(gw * gh * 3);
  for (let i = 0; i < gw * gh; i++) for (let k = 0; k < 3; k++) out[i * 3 + k] = px[i * 4 + k] / 255;
  return out;
}

/** Hover query: a patch attends to tokens that look alike and sit nearby. */
export function patchAttention(colors: Float32Array, gw: number, gh: number, qx: number, qy: number): Float32Array {
  const a = new Float32Array(gw * gh);
  const q = (qy * gw + qx) * 3;
  for (let y = 0; y < gh; y++)
    for (let x = 0; x < gw; x++) {
      const i = y * gw + x;
      let d2 = 0;
      for (let k = 0; k < 3; k++) d2 += (colors[i * 3 + k] - colors[q + k]) ** 2;
      const sim = Math.exp(-d2 / 0.02);
      const near = Math.exp(-((x - qx) ** 2 + (y - qy) ** 2) / (2 * 7 ** 2));
      a[i] = sim * (0.35 + 0.65 * near);
    }
  return normalize(a);
}

export function topK(a: Float32Array, k: number): number[] {
  return Array.from(a.keys())
    .sort((i, j) => a[j] - a[i])
    .slice(0, k)
    .filter((i) => a[i] > 0.2);
}

// ── Real data (extracted with pipeline/vrh_fig1.py) ─────────────────────────

export interface SceneAttention {
  /** Per question: [top-k mean, head 1, head 2, …] maps and the masked "other heads" map. */
  questions: { vrh: Uint8Array[]; rest: Uint8Array }[];
  simIdx: Uint16Array;
  simVal: Uint8Array;
  simK: number;
}

interface RawSceneAttention {
  questions: { vrh: string[]; rest: string }[];
  simIdx: string;
  simVal: string;
  simK: number;
}

const bytes = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));

export async function loadAttention(url: string): Promise<{
  scenes: Record<string, SceneAttention>; video: Uint8Array[] | null; problem: Uint8Array[] | null;
}> {
  const raw = (await (await fetch(url)).json()) as { scenes: Record<string, RawSceneAttention>; video?: string[]; problem?: string[] };
  const scenes = Object.fromEntries(
    Object.entries(raw.scenes).map(([id, s]) => [id, {
      questions: s.questions.map((q) => ({ vrh: q.vrh.map(bytes), rest: bytes(q.rest) })),
      simIdx: new Uint16Array(bytes(s.simIdx).buffer),
      simVal: bytes(s.simVal),
      simK: s.simK,
    }]),
  );
  return { scenes, video: raw.video ? raw.video.map(bytes) : null, problem: raw.problem ? raw.problem.map(bytes) : null };
}

/** u8 map (normalized by max) → display values; gamma lifts mid-range evidence. */
export const fromU8 = (m: Uint8Array, gamma = 0.75) => Float32Array.from(m, (v) => Math.pow(v / 255, gamma));

/** Hover query from stored top-k hidden-state similarities of one token. */
export function similarityAttention(s: SceneAttention, n: number, token: number): Float32Array {
  const a = new Float32Array(n);
  for (let k = 0; k < s.simK; k++) a[s.simIdx[token * s.simK + k]] = s.simVal[token * s.simK + k] / 255;
  a[token] = 1;
  return a;
}

// ── Chapter III video ───────────────────────────────────────────────────────

export interface VideoFrame { image: string; t: number; depth: number[]; box: Box }
export interface VideoData {
  id: string;
  credit: string;
  url: string;
  question: string;
  answer: string;
  /** Answer with the retrieval heads masked (model output). */
  masked?: string;
  /** Share of question-specific attention per frame (oldest first). */
  mass?: number[];
  real?: boolean;
  grid: [number, number];
  frames: VideoFrame[];
}

/** Frame as a SceneData so it can reuse TokenMesh. */
export const frameScene = (v: VideoData, i: number): SceneData => ({
  id: `${v.id}-f${i}`, image: v.frames[i].image, grid: v.grid, credit: v.credit, depth: v.frames[i].depth,
  questions: [{ q: v.question, answer: v.answer, masked: v.answer, box: v.frames[i].box }],
});

/**
 * ILLUSTRATIVE memory attention for the question asked at the last frame ("where was the ball
 * at the start?"): concentrated on the ball in the earliest frames, fading toward the present.
 * Replaced by attention extracted from the model when available.
 */
export function memoryAttention(v: VideoData, i: number): Float32Array {
  const weight = [1, 0.75, 0.45, 0.3, 0.55][i] ?? 0.3;
  return evidenceAttention(frameScene(v, i), 0).map((a) => a * weight);
}
