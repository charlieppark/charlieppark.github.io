// The Token Field: renderer, camera rig, scroll-driven chapter states, attention
// arcs and pointer interaction for the home-page story.
import * as THREE from 'three';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import {
  evidenceAttention, frameScene, fromU8, maskedAttention, memoryAttention, patchAttention, similarityAttention,
  tokenColors, topK, type Box, type SceneAttention, type SceneData, type VideoData,
} from './attention';
import { TokenMesh } from './tokens';

/** Visual state for each story chapter; the field eases between neighbours while scrolling. */
interface Keyframe {
  relief: number; focus: number; yaw: number; pitch: number; dist: number; stack: number;
  mobileTop: number; warp: number; mosaic: number;
}
const CHAPTERS: Keyframe[] = [
  // mobileTop: fraction of the viewport (from the top) left free above that chapter's text card.
  { relief: 0.12, focus: 0, yaw: -0.3, pitch: 0.16, dist: 1.0, stack: 0, mobileTop: 0.44, warp: 0, mosaic: 1 },  // 0 · self-portrait
  { relief: 0.15, focus: 1, yaw: -0.14, pitch: 0.2, dist: 1.02, stack: 0, mobileTop: 0.36, warp: 0, mosaic: 0 }, // I · the problem
  { relief: 1.0, focus: 0, yaw: 0.32, pitch: 0.22, dist: 1.05, stack: 0, mobileTop: 0.62, warp: 1, mosaic: 0 },  // II · use (warping)
  { relief: 0.2, focus: 1, yaw: -0.14, pitch: 0.2, dist: 1.02, stack: 0, mobileTop: 0.44, warp: 0, mosaic: 0 },  // III · verify
  { relief: 0.14, focus: 0, yaw: 0.88, pitch: 0.26, dist: 1.6, stack: 1, mobileTop: 0.42, warp: 0, mosaic: 0 },  // IV · over time
];
export const CHAPTER_COUNT = CHAPTERS.length;
const ARC_K = 9;
const ARC_SEG = 28;
const FRAME_GAP = 8; // world units between video frames in the memory stack
const MAX_BOXES = 2;

export interface FieldEvents {
  /** Query token (hovered or pinned) with its depth. */
  hover?: (info: { row: number; col: number; depth: number } | null) => void;
  /** Screen position of the floating query token, or null when hidden. */
  query?: (pos: { x: number; y: number } | null) => void;
  /** Interaction state that the page narrates (changes only). */
  state?: (st: FieldState) => void;
  /** Screen anchors (top-left corners) of the visible boxes. */
  boxes?: (list: { i: number; x: number; y: number }[]) => void;
  /** Screen anchors for plate labels in the memory stack: `frame` is the kept frame's index, −1 the live video. */
  frames?: (list: { frame: number; x: number; y: number }[]) => void;
  /** Chapter 0: my glasses were knocked off. */
  ouch?: () => void;
  /** Chapter 0: the portrait token under the pointer, or null. */
  portraitHover?: (cell: [number, number] | null) => void;
  /** Chapter 0: a click or tap on the portrait. Return true when handled (no default burst). */
  tap?: (cell: [number, number], pointerType: string) => boolean;
}

export interface FieldState {
  chapter: number;
  /** Token acting as the query (hovered, or pinned by a click), as [col, row]. */
  query: [number, number] | null;
  pinned: boolean;
  /** Camera angles relative to the photo's original (frontal) camera, in degrees. */
  yawDeg: number;
  pitchDeg: number;
  dragging: boolean;
}

/** Chapter 0: my photo as tokens; head ellipse and eye centers drive the look-at behaviour. */
export interface Portrait {
  data: SceneData;
  texture: THREE.Texture;
  head: [number, number, number, number]; // center col,row; radii cols,rows
  eyes: [number, number, number, number]; // uv of left and right eye centers (y down)
  eyeR: number;
  glasses: number[];                      // token indices of the glasses (for the "hey!")
}

const hexVec = (hex: string) => {
  const h = hex.trim().replace('#', '');
  return new THREE.Vector3(parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255);
};
/** Scene texture sampled as raw display values (the shader skips color management). */
export function makeTexture(img: HTMLImageElement) {
  const tex = new THREE.Texture(img);
  tex.colorSpace = THREE.NoColorSpace;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  return tex;
}

const damp = (a: number, b: number, k: number, dt: number) => a + (b - a) * (1 - Math.exp(-k * dt));
const smooth = (t: number) => t * t * (3 - 2 * t);

export class TokenField {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(30, 1, 0.1, 500);
  private meshes: TokenMesh[] = [];
  private colors: Float32Array[] = [];
  private frames: TokenMesh[] = [];
  // Chapter IV playback: the live video as a plate of tokens, the playhead (s), each kept frame's time,
  // and how far the question at the end has been asked (0 → 1).
  private live: TokenMesh | null = null;
  private clip: HTMLVideoElement | null = null;
  private frameTimes: number[] = [];
  private playT = 0;
  private asked = 1;
  private mosaicMesh: TokenMesh | null = null; // the portrait token mesh (Chapter 0)
  private portrait?: Portrait;
  // Chapter 0 physics (per token): velocity, angular velocity, seconds since last hit
  private vel = new Float32Array(0);
  private angVel = new Float32Array(0);
  private hitT = new Float32Array(0);
  private portraitFocus = 0; // 1 while my photo shows the retrieval heads' attention
  private lastPortraitKey = '';
  private poke ={ on: false, x: 0, y: 0, last: new THREE.Vector3(), lastT: 0, moved: false };
  private look = new THREE.Vector2();
  private gaze = new THREE.Vector2();
  private blink = { t: 9, next: 3, lastGaze: 0 };
  private ouchCool = 0;
  private heat = Array.from({ length: 5 }, () => new THREE.Vector3());
  private bg = new THREE.Vector3();
  private ink = new THREE.Vector3();
  private arcs: Line2[] = [];
  private arcMat: LineMaterial;
  private queryToken: THREE.Mesh;
  private boxLines: Line2[] = [];
  private boxMat: LineMaterial;
  private boxZ = new Array(MAX_BOXES).fill(0);
  private boxAnchors = Array.from({ length: MAX_BOXES }, () => new THREE.Vector3());
  private raycaster = new THREE.Raycaster();
  private lastT = 0;
  private raf = 0;
  private running = false;
  private w = 1;
  private h = 1;

  // State
  private current = 0;
  private question = 0;
  private masked = false;
  private view = 0;
  private memoryMasked = false;
  private real: Record<string, SceneAttention> | null = null;
  /** Attention shown instead of the question's (Chapter I), with its boxes. */
  private override: { attn: Float32Array; boxes: Box[] } | null = null;
  private chapter = 0;
  private hoverCell: [number, number] | null = null;
  private pinned: [number, number] | null = null;
  /** Effective query token: live hover first, else the pinned token (none while the memory stack is out). */
  private qcell: [number, number] | null = null;
  private pointer = new THREE.Vector2();
  private pointerIn = false;
  private drag = { on: false, x: 0, y: 0, yaw: 0, pitch: 0, yaw0: 0, pitch0: 0 };
  private intro = 0;
  private rippleT = 100;
  private arcProgress = 0;
  private arcKey = '';
  private viewYaw = 0;
  private viewPitch = 0;

  // Eased values
  private s = {
    relief: 0, focus: 0, yaw: -0.3, pitch: 0.16, dist: 1, stack: 0, hover: 0, px: 0, py: 0,
    mobileTop: 0.44, warp: 0, mosaic: 1,
  };

  constructor(
    private canvas: HTMLCanvasElement,
    private data: SceneData[],
    textures: { tex: THREE.Texture; img: HTMLImageElement }[],
    private events: FieldEvents = {},
    private reduced = false,
    video?: { data: VideoData; textures: THREE.Texture[]; clip?: HTMLVideoElement },
    portrait?: Portrait,
  ) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace; // shader works in display (sRGB) values directly
    this.readTheme();

    data.forEach((d, i) => {
      const m = new TokenMesh(d, textures[i].tex, this.heat, this.bg, this.ink);
      m.setRetain(evidenceAttention(d, 0).map((v) => (v > 0.3 ? 1 : 0.15 * v)));
      m.mesh.visible = false;
      m.mesh.renderOrder = i === 0 ? 10 : 0;
      this.meshes.push(m);
      this.colors.push(tokenColors(textures[i].img, d.grid[0], d.grid[1]));
      this.scene.add(m.mesh);
    });

    // Chapter 0: my photo as tokens. It looks at the cursor, blinks, and can be poked apart.
    if (portrait) {
      this.portrait = portrait;
      const mm = new TokenMesh({ ...portrait.data, id: 'portrait-ch0' }, portrait.texture, this.heat, this.bg, this.ink);
      mm.uniforms.uVisible.value = 1;
      (mm.uniforms.uHead.value as THREE.Vector4).set(...portrait.head);
      (mm.uniforms.uEyes.value as THREE.Vector4).set(...portrait.eyes);
      mm.uniforms.uEyeR.value = portrait.eyeR;
      const n = mm.gw * mm.gh;
      this.vel = new Float32Array(n * 3);
      this.angVel = new Float32Array(n * 3);
      this.hitT = new Float32Array(n).fill(99);
      this.mosaicMesh = mm;
      this.scene.add(mm.mesh);
    } else this.s.mosaic = 0;

    // Chapter IV: video frames stacked in time; the last one is "now".
    if (video) {
      video.data.frames.forEach((_, i) => {
        const m = new TokenMesh(frameScene(video.data, i), video.textures[i], this.heat, this.bg, this.ink);
        const a = memoryAttention(video.data, i);
        m.target.set(a); m.attn.set(a);
        m.setRetain(a.map((v) => (v > 0.3 ? 1 : 0)));
        m.mesh.visible = false;
        this.frames.push(m);
        this.scene.add(m.mesh);
      });
      this.frameTimes = video.data.frames.map((f) => f.t);
      this.playT = this.frameTimes[this.frameTimes.length - 1]; // until played: the full stack, question asked
      // The clip itself, tokenized live: the front plate while it plays (flat; kept frames carry the depth).
      if (video.clip) {
        this.clip = video.clip;
        const tex = new THREE.VideoTexture(video.clip);
        tex.colorSpace = THREE.NoColorSpace;
        const sd = frameScene(video.data, video.data.frames.length - 1);
        this.live = new TokenMesh({ ...sd, id: `${video.data.id}-live`, depth: sd.depth.map(() => 0) }, tex, this.heat, this.bg, this.ink);
        this.live.mesh.visible = false;
        this.scene.add(this.live.mesh);
      }
    }

    this.arcMat = new LineMaterial({ color: 0xffffff, linewidth: 1.6, transparent: true, opacity: 0, worldUnits: false });
    for (let k = 0; k < ARC_K; k++) {
      const g = new LineGeometry();
      g.setPositions(new Array(ARC_SEG * 3).fill(0));
      const line = new Line2(g, this.arcMat);
      line.frustumCulled = false;
      line.renderOrder = 20;
      this.arcs.push(line);
      this.scene.add(line);
    }
    this.queryToken = new THREE.Mesh(
      new THREE.BoxGeometry(1.1, 1.1, 0.4),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }),
    );
    this.queryToken.renderOrder = 21;
    this.scene.add(this.queryToken);
    this.boxMat = new LineMaterial({ color: 0x000000, linewidth: 1.2, transparent: true, opacity: 0, worldUnits: false, depthTest: false });
    for (let i = 0; i < MAX_BOXES; i++) {
      const g = new LineGeometry();
      g.setPositions(new Array(15).fill(0));
      const line = new Line2(g, this.boxMat);
      line.frustumCulled = false;
      line.renderOrder = 22;
      this.boxLines.push(line);
      this.scene.add(line);
    }
    this.applyThemeColors();

    if (reduced) this.intro = 2;
    this.refreshAttention();
    this.bindInput();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  // ── Public API ──────────────────────────────────────────────────────────
  get sceneIndex() { return this.current; }

  setScene(i: number) {
    if (i === this.current) return;
    this.meshes[this.current].mesh.renderOrder = 0;
    this.current = i;
    const m = this.meshes[i];
    m.mesh.renderOrder = 10;
    m.uniforms.uVisible.value = 0;
    m.uniforms.uRelief.value = this.s.relief;
    this.hoverCell = this.pinned = this.qcell = null;
    this.question = 0;
    this.refreshAttention();
  }

  setQuestion(qi: number) { this.question = qi; this.pinned = null; this.refreshAttention(); }
  /** 0 = top-k mean, 1… = individual retrieval heads. */
  setView(v: number) { this.view = v; this.refreshAttention(); }
  setMasked(b: boolean) { this.masked = b; this.refreshAttention(); }
  /** Chapter IV: with retrieval heads masked, the past frames' kept tokens fall back. */
  setMemoryMasked(b: boolean) { this.memoryMasked = b; }
  /** Chapter IV: video time (s). Frames at or before it are in memory; the question is asked at the last one. */
  setPlayhead(t: number) { this.playT = t; }
  get debugMemory() {
    return { t: this.playT, kept: this.frameTimes.filter((ft) => this.playT >= ft - 1e-3).length, live: !!this.live?.mesh.visible, asked: this.asked };
  }
  /** Chapter I: show this attention (and boxes) on the current scene instead of the question's. */
  setOverride(o: { attn: Float32Array; boxes: Box[] } | null) { this.override = o; this.pinned = null; this.refreshAttention(); }
  /** Scroll position in chapters (0 … CHAPTER_COUNT − 1, fractional while between). */
  setChapter(f: number) { this.chapter = Math.max(0, Math.min(CHAPTERS.length - 1, f)); }

  /** Real attention from the question at t to each video frame (jointly normalized, oldest first). */
  setVideoAttention(maps: Uint8Array[]) {
    this.frames.forEach((m, i) => {
      if (!maps[i]) return;
      const a = fromU8(maps[i]);
      m.target.set(a); m.attn.set(a);
      (m.mesh.geometry.getAttribute('aAttn') as THREE.InstancedBufferAttribute).needsUpdate = true;
      m.setRetain(a.map((v) => (v > 0.3 ? 1 : 0)));
    });
  }

  /** Swap illustrative maps for attention extracted from the model. */
  setAttentionData(real: Record<string, SceneAttention>) {
    this.real = real;
    this.refreshAttention();
  }

  readTheme() {
    const cs = getComputedStyle(document.documentElement);
    for (let i = 0; i < 5; i++) this.heat[i].copy(hexVec(cs.getPropertyValue(`--heat-${i}`)));
    this.bg.copy(hexVec(cs.getPropertyValue('--bg')));
    this.ink.copy(hexVec(cs.getPropertyValue('--ink')));
    if (this.arcMat) this.applyThemeColors();
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.lastT = performance.now();
    const loop = () => { this.frame(); if (this.running) this.raf = requestAnimationFrame(loop); };
    this.raf = requestAnimationFrame(loop);
  }

  stop() { this.running = false; cancelAnimationFrame(this.raf); }

  /** Tokens kept in memory for a video frame (retain = 1); 0 = oldest. */
  retainedCount(frame: number) {
    const m = this.frames[frame];
    if (!m) return 0;
    const a = m.mesh.geometry.getAttribute('aRetain').array as Float32Array;
    let n = 0;
    for (const v of a) if (v >= 1) n++;
    return n;
  }

  /** Dev/test helpers. */
  debugTokenScreen(i: number) {
    const m = this.active();
    const p = m.tokenPosition(i, new THREE.Vector3()).project(this.camera);
    return { x: (p.x * 0.5 + 0.5) * this.w, y: (-p.y * 0.5 + 0.5) * this.h, grid: [m.gw, m.gh] };
  }
  /** Screen-space direction of the field's facing normal (+z); x>0 = facing right, y>0 = facing down. */
  debugFacing() {
    const a = new THREE.Vector3(0, 0, 0).project(this.camera), b = new THREE.Vector3(0, 0, 6).project(this.camera);
    return { x: ((b.x - a.x) * this.w) / 2, y: (-(b.y - a.y) * this.h) / 2 };
  }
  /** Screen position of a portrait token (Chapter 0), e.g. to anchor a speech bubble. `home` ignores pokes. */
  portraitScreen(col: number, row: number, home = false) {
    const m = this.mosaicMesh;
    if (!m) return null;
    const i = row * m.gw + col, p = m.tokenPosition(i, new THREE.Vector3());
    if (home) p.sub(this.tmpA.set(m.off[i * 3], m.off[i * 3 + 1], m.off[i * 3 + 2]));
    return this.project(p);
  }
  /** Screen corners (clockwise from top-left) of a rectangle on my photo, in token units (0..18), on its surface. */
  portraitQuad(x0: number, y0: number, x1: number, y1: number) {
    const m = this.mosaicMesh;
    if (!m) return null;
    const c = Math.min(m.gw - 1, Math.max(0, Math.floor(x0))), r = Math.min(m.gh - 1, Math.max(0, Math.floor(y0)));
    const i = r * m.gw + c, z = m.tokenPosition(i, this.tmpB).z - m.off[i * 3 + 2];
    return ([[x0, y0], [x1, y0], [x1, y1], [x0, y1]] as const)
      .map(([x, y]) => this.project(this.tmpA.set(x - m.gw / 2, -(y - m.gh / 2), z)));
  }
  /** Chapter 0: show this attention on my photo (heat color + lift), or none. */
  setPortraitAttention(a: Float32Array | null) {
    const m = this.mosaicMesh;
    if (!m) return;
    m.target.set(a ?? new Float32Array(m.target.length));
    this.portraitFocus = a ? 1 : 0;
  }
  get debugHover() { return this.hoverCell; }
  /** Largest physics displacement on the portrait (0 when every token is home). */
  get debugPoked() {
    const m = this.mosaicMesh;
    if (!m) return 0;
    let mx = 0;
    for (let i = 0; i < m.off.length; i += 3) mx = Math.max(mx, Math.hypot(m.off[i], m.off[i + 1], m.off[i + 2]));
    return mx;
  }
  get debugPinned() { return this.pinned; }
  get debugRipple() { return (this.active().uniforms.uRipple.value as THREE.Vector3).toArray(); }

  // ── Internals ───────────────────────────────────────────────────────────
  /** The mesh the pointer works on: the self-portrait in Chapter 0, otherwise the current scene. */
  private active(): TokenMesh {
    return this.mosaicMesh && this.s.mosaic > 0.5 ? this.mosaicMesh : this.meshes[this.current];
  }

  private applyThemeColors() {
    const cs = getComputedStyle(document.documentElement);
    const accent = hexVec(cs.getPropertyValue('--accent'));
    const ink = hexVec(cs.getPropertyValue('--ink'));
    this.boxMat?.color.setRGB(ink.x, ink.y, ink.z);
    this.arcMat.color.setRGB(accent.x, accent.y, accent.z);
    (this.queryToken.material as THREE.MeshBasicMaterial).color.setRGB(accent.x, accent.y, accent.z);
  }

  private refreshAttention() {
    if (this.mosaicMesh && this.s.mosaic > 0.5) return; // Chapter 0's portrait shows no attention
    const m = this.meshes[this.current];
    const d = this.data[this.current];
    const r = this.real?.[d.id];
    let a: Float32Array;
    if (this.qcell) {
      const tok = this.qcell[1] * m.gw + this.qcell[0];
      a = r ? similarityAttention(r, m.gw * m.gh, tok) : patchAttention(this.colors[this.current], m.gw, m.gh, this.qcell[0], this.qcell[1]);
    } else if (this.override) a = this.override.attn;
    else if (r) {
      const q = r.questions[this.question];
      a = fromU8(this.masked ? q.rest : q.vrh[Math.min(this.view, q.vrh.length - 1)]);
    } else a = this.masked ? maskedAttention(d, this.question) : evidenceAttention(d, this.question);
    m.target.set(a);
    const key = `${this.current}:${this.question}:${this.masked}:${this.view}:${!!this.override}:${this.qcell?.join(',') ?? ''}`;
    if (key !== this.arcKey) { this.arcKey = key; this.arcProgress = 0; }
  }

  private resize() {
    const r = this.canvas.getBoundingClientRect();
    this.w = Math.max(1, r.width); this.h = Math.max(1, r.height);
    this.renderer.setSize(this.w, this.h, false);
    this.camera.aspect = this.w / this.h;
    this.applyViewOffset();
    this.arcMat.resolution.set(this.w, this.h);
    this.boxMat.resolution.set(this.w, this.h);
  }

  /** Desktop: push the field right of the text column. Mobile: center it in the space above the card. */
  private applyViewOffset() {
    const wide = this.w >= 900;
    // Desktop: lift clear of the legend. Mobile: slightly below the middle of the free space (room for labels).
    const y = wide ? this.h * 0.05 : this.h * (0.5 - this.s.mobileTop * 0.58);
    this.camera.setViewOffset(this.w, this.h, wide ? -this.w * 0.2 : 0, y, this.w, this.h);
    this.camera.updateProjectionMatrix();
  }

  private fitDistance(gw: number, gh: number) {
    const t = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const wide = this.w >= 900;
    const byW = gw / ((wide ? 0.42 : 0.9) * 2 * t * this.camera.aspect);
    const byH = gh / ((wide ? 0.52 : this.s.mobileTop * 0.62) * 2 * t);
    return Math.max(byW, byH);
  }

  /** Token under the pointer: nearest tile hit by the camera ray (tiles sit at different heights). */
  private cellFromPointer(): [number, number] | null {
    const m = this.active();
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const i = this.s.warp > 0.5 ? this.pickFacing(m) : m.pick(this.raycaster.ray);
    return i < 0 ? null : [i % m.gw, Math.floor(i / m.gw)];
  }

  /** Picking for camera-facing patches: nearest tile whose on-screen square contains the pointer. */
  private pickFacing(m: TokenMesh) {
    const t = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const p = new THREE.Vector3(), q = new THREE.Vector3();
    let best = -1, bestD = Infinity;
    for (let i = 0; i < m.gw * m.gh; i++) {
      m.tokenPosition(i, p);
      const d = q.copy(p).applyMatrix4(this.camera.matrixWorldInverse).z; // negative in front
      if (d >= 0) continue;
      p.project(this.camera);
      const hy = 0.5 / (-d * t), hx = hy / this.camera.aspect; // half tile in NDC
      if (Math.abs(this.pointer.x - p.x) <= hx && Math.abs(this.pointer.y - p.y) <= hy && -d < bestD) { bestD = -d; best = i; }
    }
    return best;
  }

  private inPortrait() { return !!this.mosaicMesh && this.s.mosaic > 0.5; }

  /** Pointer on the portrait plane (z = 0), in world units. */
  private pointerWorld() {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = new THREE.Vector3();
    return this.raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), hit) ? hit : null;
  }

  /** Portrait cell under the pointer at its home place (knocked-out tokens floating in front don't count). */
  private portraitCell(): [number, number] | null {
    const m = this.mosaicMesh, w = this.pointerWorld();
    if (!m || !w) return null;
    const c = Math.floor(w.x + m.gw / 2), r = Math.floor(m.gh / 2 - w.y);
    return c >= 0 && r >= 0 && c < m.gw && r < m.gh ? [c, r] : null;
  }

  /** Kick tokens near `at` with velocity `v` (world units / s). */
  private kick(at: THREE.Vector3, v: THREE.Vector3, radius: number, radial: number) {
    const m = this.mosaicMesh!;
    const p = new THREE.Vector3();
    for (let i = 0; i < m.gw * m.gh; i++) {
      m.tokenPosition(i, p);
      const dx = p.x - at.x, dy = p.y - at.y, d = Math.hypot(dx, dy);
      if (d > radius) continue;
      const f = 1 - d / radius;
      const r = Math.random;
      this.vel[i * 3] += v.x * f + (dx / (d + 0.3)) * radial * f + (r() - 0.5) * 2;
      this.vel[i * 3 + 1] += v.y * f + (dy / (d + 0.3)) * radial * f + (r() - 0.5) * 2;
      this.vel[i * 3 + 2] += (6 + r() * 6) * f + Math.hypot(v.x, v.y) * 0.25 * f;
      for (let k = 0; k < 3; k++) this.angVel[i * 3 + k] += (r() - 0.5) * 18 * f;
      this.hitT[i] = 0;
    }
  }

  private burst() {
    const w = this.pointerWorld();
    if (w) this.kick(w, new THREE.Vector3(), 3.2, 9);
  }

  private pokeMove(e: PointerEvent) {
    const w = this.pointerWorld();
    if (!w) return;
    const now = performance.now(), dt = Math.max(8, now - this.poke.lastT) / 1000;
    const v = w.clone().sub(this.poke.last).divideScalar(dt);
    if (Math.hypot(e.clientX - this.poke.x, e.clientY - this.poke.y) > 5 && v.length() > 4) this.kick(w, v.multiplyScalar(0.55), 1.8, 1);
    this.poke.last.copy(w); this.poke.lastT = now;
  }

  /** Integrate poked tokens: fly, fall, then spring home. */
  private stepPhysics(dt: number) {
    const m = this.mosaicMesh!;
    const n = m.gw * m.gh;
    let active = false, glassesOff = 0;
    for (let i = 0; i < n; i++) {
      this.hitT[i] += dt;
      const home = smooth(Math.min(1, Math.max(0, (this.hitT[i] - 0.7) / 0.9)));
      for (let k = 0; k < 3; k++) {
        const j = i * 3 + k;
        if (Math.abs(m.off[j]) < 1e-4 && Math.abs(this.vel[j]) < 1e-4 && Math.abs(m.spin[j]) < 1e-4 && Math.abs(this.angVel[j]) < 1e-4) continue;
        active = true;
        const g = k === 1 ? -18 * (1 - home) : 0;                          // gravity while airborne
        this.vel[j] += (g - m.off[j] * 40 * home - this.vel[j] * (1.2 + 8 * home)) * dt;
        m.off[j] += this.vel[j] * dt;
        this.angVel[j] += (-m.spin[j] * 40 * home - this.angVel[j] * (0.8 + 8 * home)) * dt;
        m.spin[j] += this.angVel[j] * dt;
        if (home >= 1 && Math.abs(m.off[j]) < 0.002 && Math.abs(this.vel[j]) < 0.01) { m.off[j] = 0; this.vel[j] = 0; }
        if (home >= 1 && Math.abs(m.spin[j]) < 0.002 && Math.abs(this.angVel[j]) < 0.01) { m.spin[j] = 0; this.angVel[j] = 0; }
      }
    }
    for (const i of this.portrait!.glasses) if (Math.hypot(m.off[i * 3], m.off[i * 3 + 1], m.off[i * 3 + 2]) > 1.2) glassesOff++;
    this.ouchCool -= dt;
    if (glassesOff >= 2 && this.ouchCool <= 0) { this.ouchCool = 3; this.events.ouch?.(); }
    if (active) m.physicsDirty();
  }

  /** Head turns and eyes follow the pointer; idle sway and blinks when it's away. */
  private stepLook(dt: number, t: number) {
    const m = this.mosaicMesh!, P = this.portrait!;
    const u = m.uniforms;
    let ty = 0, tp = 0, gx = 0, gy = 0;
    if (this.pointerIn) {
      // Pointer relative to the head center on screen, in units of the portrait's half-size.
      const c = this.tmpA.set(P.head[0] - m.gw / 2 + 0.5, -(P.head[1] - m.gh / 2 + 0.5), 0).project(this.camera);
      const e = this.tmpB.set(m.gw / 2, 0, 0).project(this.camera);
      const scale = Math.max(0.05, Math.abs(e.x - this.tmpC.set(0, 0, 0).project(this.camera).x));
      const dx = THREE.MathUtils.clamp((this.pointer.x - c.x) / scale, -1.6, 1.6);
      const dy = THREE.MathUtils.clamp((this.pointer.y - c.y) / scale, -1.6, 1.6);
      ty = dx * 0.14; tp = -dy * 0.1;
      // Gaze: mostly sideways, barely vertical (a vertical shift reads as squinting), capped.
      gx = THREE.MathUtils.clamp(dx, -1, 1) * 0.0045; gy = -THREE.MathUtils.clamp(dy, -1, 1) * 0.0012;
    } else {
      ty = Math.sin(t * 0.35) * 0.06; tp = Math.sin(t * 0.23) * 0.03;
    }
    this.look.set(damp(this.look.x, ty, 5, dt), damp(this.look.y, tp, 5, dt));
    this.gaze.set(damp(this.gaze.x, gx, 9, dt), damp(this.gaze.y, gy, 9, dt));
    (u.uLook.value as THREE.Vector2).copy(this.look);
    (u.uGaze.value as THREE.Vector2).copy(this.gaze);
    // Blinks: every 2–6 s, sometimes a quick double; also often when the gaze jumps (as people do).
    const b = this.blink;
    b.t += dt; b.next -= dt;
    const jump = Math.abs(gx - b.lastGaze) > 0.003;
    b.lastGaze = gx;
    if (b.t > 0.4 && (b.next <= 0 || (jump && b.t > 1.2 && Math.random() < 0.35))) {
      b.t = 0;
      b.next = Math.random() < 0.15 ? 0.45 : 2 + Math.random() * 4;
    }
    // Fast close (70 ms, ease-in), brief hold, slower open (180 ms, ease-out).
    const bt = b.t;
    u.uBlink.value = bt < 0.07 ? (bt / 0.07) ** 2 : bt < 0.1 ? 1 : bt < 0.28 ? 1 - smooth((bt - 0.1) / 0.18) : 0;
  }

  private bindInput() {
    const el = this.canvas;
    const toNdc = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    };
    el.addEventListener('pointermove', (e) => {
      toNdc(e);
      this.pointerIn = true;
      if (this.poke.on) this.pokeMove(e);
      const d = this.drag;
      if (d.on && !this.inPortrait()) {
        // Grab-and-turn: dragging right turns the field's front toward the right (camera orbits left),
        // dragging down tips its top toward the viewer (camera rises).
        // Clamped to nearby viewpoints (±25° / ±18°), the range token warping is meant for.
        d.yaw = Math.max(-0.44, Math.min(0.44, d.yaw0 - (e.clientX - d.x) * 0.006));
        d.pitch = Math.max(-0.31, Math.min(0.31, d.pitch0 + (e.clientY - d.y) * 0.004));
      }
    });
    el.addEventListener('pointerleave', () => { this.pointerIn = false; this.setHover(null); });
    el.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      if (this.inPortrait()) {
        toNdc(e);
        const w = this.pointerWorld();
        Object.assign(this.poke, { on: true, x: e.clientX, y: e.clientY, lastT: performance.now(), moved: false });
        if (w) this.poke.last.copy(w);
      }
      const d = this.drag;
      Object.assign(d, { on: true, x: e.clientX, y: e.clientY, yaw0: d.yaw, pitch0: d.pitch });
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointerup', (e) => {
      if (!this.drag.on) return;
      const moved = Math.hypot(e.clientX - this.drag.x, e.clientY - this.drag.y);
      this.drag.on = false;
      const poked = this.poke.on;
      this.poke.on = false;
      toNdc(e);
      this.camera.updateMatrixWorld();
      if (this.inPortrait()) {
        if (poked && moved < 5) {
          const c = this.portraitCell();
          if (!c || !this.events.tap?.(c, e.pointerType)) this.burst();
        }
        return;
      }
      if (moved >= 5) return;
      const c = this.cellFromPointer();
      if (!c) { this.pinned = null; return; }
      const m = this.active();
      // Otherwise a click pins this token as the query (works without hover, e.g. touch); again to release.
      const same = this.pinned && this.pinned[0] === c[0] && this.pinned[1] === c[1];
      this.pinned = same ? null : c;
      (m.uniforms.uRipple.value as THREE.Vector3).set(c[0], c[1], 0);
      this.rippleT = 0;
    });
    // Touch scrolls (and other interruptions) cancel the gesture without a pointerup.
    el.addEventListener('pointercancel', () => { this.drag.on = false; this.poke.on = false; this.pointerIn = false; this.setHover(null); });
  }

  private setHover(c: [number, number] | null) {
    const same = c && this.hoverCell && c[0] === this.hoverCell[0] && c[1] === this.hoverCell[1];
    if (same || (!c && !this.hoverCell)) return;
    this.hoverCell = c;
  }

  private frame() {
    const now = performance.now();
    const dt = Math.min((now - this.lastT) / 1000, 0.05);
    this.lastT = now;
    const s = this.s;
    this.intro = Math.min(2, this.intro + dt * 0.75);
    this.rippleT += dt;

    // Interpolate chapter keyframes; each chapter holds through the middle of its scroll slot.
    const i0 = Math.floor(this.chapter), i1 = Math.min(CHAPTERS.length - 1, i0 + 1);
    const t = smooth(Math.min(1, Math.max(0, (this.chapter - i0 - 0.2) / 0.6)));
    const k = Object.fromEntries(
      (Object.keys(CHAPTERS[0]) as (keyof Keyframe)[]).map((key) => [key, CHAPTERS[i0][key] + (CHAPTERS[i1][key] - CHAPTERS[i0][key]) * t]),
    ) as unknown as Keyframe;
    if (!this.mosaicMesh) k.mosaic = 0;

    const wasMosaic = s.mosaic > 0.5;
    s.relief = damp(s.relief, k.relief, 4, dt);
    s.focus = damp(s.focus, k.focus, 4, dt);
    s.stack = damp(s.stack, k.stack, 3, dt);
    s.warp = damp(s.warp, k.warp, 4, dt);
    s.mosaic = damp(s.mosaic, k.mosaic, 4, dt);
    s.hover = damp(s.hover, this.qcell ? 1 : 0, 8, dt);
    if (!this.drag.on) { this.drag.yaw = damp(this.drag.yaw, 0, 2.5, dt); this.drag.pitch = damp(this.drag.pitch, 0, 2.5, dt); }
    s.px = damp(s.px, this.pointerIn && !this.reduced ? this.pointer.x : 0, 2, dt);
    s.py = damp(s.py, this.pointerIn && !this.reduced ? this.pointer.y : 0, 2, dt);
    s.yaw = damp(s.yaw, k.yaw, 3, dt);
    s.pitch = damp(s.pitch, k.pitch, 3, dt);
    s.dist = damp(s.dist, k.dist, 3, dt);
    if (this.w < 900 && Math.abs(s.mobileTop - k.mobileTop) > 1e-4) {
      s.mobileTop = damp(s.mobileTop, k.mobileTop, 3, dt);
      this.applyViewOffset();
    }
    if (wasMosaic !== s.mosaic > 0.5) { this.hoverCell = this.pinned = this.qcell = null; this.refreshAttention(); }

    // Camera orbit around the field.
    const act = this.active();
    // Parallax leans the field toward the cursor, like a gentle drag.
    const yaw = s.yaw + this.drag.yaw - s.px * 0.12;
    const pitch = Math.max(-0.2, Math.min(1.1, s.pitch + this.drag.pitch - s.py * 0.08));
    this.viewYaw = yaw; this.viewPitch = pitch;
    const R = this.fitDistance(act.gw, act.gh) * s.dist;
    this.camera.position.set(R * Math.sin(yaw) * Math.cos(pitch), R * Math.sin(pitch), R * Math.cos(yaw) * Math.cos(pitch));
    this.camera.lookAt(0, 0, s.stack * -FRAME_GAP * 2);
    this.camera.updateMatrixWorld();

    // Per-scene uniforms. The still image appears after the self-portrait and gives way to the video.
    const focus = Math.max(s.focus, s.hover * 0.9);
    const sceneVis = (1 - smooth(Math.min(1, k.mosaic * 1.2))) * (1 - smooth(Math.min(1, s.stack * 1.5)));
    this.meshes.forEach((m, idx) => {
      const u = m.uniforms;
      const isCur = idx === this.current;
      u.uVisible.value = damp(u.uVisible.value, isCur ? sceneVis : 0, isCur ? 2.2 : 3, dt);
      m.mesh.visible = u.uVisible.value > 0.01;
      u.uIntro.value = this.intro;
      u.uRelief.value = isCur ? s.relief : damp(u.uRelief.value, 0.1, 3, dt);
      u.uFocus.value = isCur ? focus : damp(u.uFocus.value, 0, 6, dt);
      // Lift tiles by question attention only. Hover similarity outlines tiles but never moves them,
      // so the tile under the cursor can't be displaced by its own query (no pick feedback loop).
      u.uLift.value = isCur ? 3 * s.focus * (1 - s.hover) : damp(u.uLift.value, 0, 6, dt);
      u.uWarp.value = isCur ? s.warp : damp(u.uWarp.value, 0, 4, dt);
      u.uSim.value = isCur ? s.hover : 0; // hover = similarity query → outlines, not heat
      u.uZ.value = 0;
      u.uStale.value = 0;
      u.uAlpha.value = 1;
      const q = isCur && act === m ? this.qcell : null;
      (u.uHover.value as THREE.Vector2).set(q ? q[0] : -100, q ? q[1] : -100);
      if (act === m) (u.uRipple.value as THREE.Vector3).z = this.rippleT;
    });
    if (this.mosaicMesh) {
      const u = this.mosaicMesh.uniforms;
      u.uVisible.value = damp(u.uVisible.value, smooth(Math.min(1, k.mosaic * 1.2)), 2.2, dt);
      this.mosaicMesh.mesh.visible = u.uVisible.value > 0.01;
      u.uIntro.value = this.intro;
      u.uRelief.value = 0.12;
      u.uFocus.value = damp(u.uFocus.value, this.portraitFocus, 5, dt);
      u.uLift.value = damp(u.uLift.value, 2.2 * this.portraitFocus, 5, dt);
      u.uSim.value = 0;
      (u.uHover.value as THREE.Vector2).set(-100, -100);
      if (act === this.mosaicMesh) (u.uRipple.value as THREE.Vector3).z = this.rippleT;
      this.mosaicMesh.stepAttention(1 - Math.exp(-6 * dt));
      if (this.mosaicMesh.mesh.visible) { this.stepPhysics(dt); this.stepLook(dt, now / 1000); }
    }
    this.meshes[this.current].stepAttention(1 - Math.exp(-7 * dt));
    this.updateFrames(dt);

    // Pointer hover → token query, picked against this frame's geometry and camera
    // (not while dragging, during the intro, or while the memory stack is out).
    const hoverOK = this.pointerIn && !this.drag.on && s.stack < 0.5 && this.intro > 1.2 && !this.inPortrait();
    this.setHover(hoverOK ? this.cellFromPointer() : null);
    // Chapter 0: which part of my photo the pointer is on (the page decides what that means).
    const pc = this.pointerIn && !this.drag.on && this.intro > 1.2 && this.inPortrait() ? this.portraitCell() : null;
    const pk = pc ? pc.join() : '';
    if (pk !== this.lastPortraitKey) { this.lastPortraitKey = pk; this.events.portraitHover?.(pc); }
    const q = s.stack < 0.5 && this.intro > 1.2 ? (this.hoverCell ?? this.pinned) : null;
    if (q?.[0] !== this.qcell?.[0] || q?.[1] !== this.qcell?.[1]) { this.qcell = q; this.refreshAttention(); }

    const cur = this.meshes[this.current];
    this.updateBoxes(cur, dt);
    this.updateArcs(cur, dt);
    this.renderer.render(this.scene, this.camera);
    this.emitEvents(act);
  }

  /**
   * Memory stack: frame t (now) in front, t−1 … t−4 behind it. Tokens the question attends to
   * are kept and lifted out of their frame; the rest of each past frame fades and drifts.
   */
  private updateFrames(dt: number) {
    const s = this.s, n = this.frames.length;
    if (!n) return;
    const times = this.frameTimes, tEnd = times[n - 1], step = n > 1 ? (tEnd - times[0]) / (n - 1) : 1;
    const t = this.playT, end = t >= tEnd - 1e-3;
    // The question is asked at the last frame: only then do its tokens lift and the rest go stale.
    this.asked = damp(this.asked, end ? 1 : 0, end ? 3 : 8, dt);
    const q = this.asked;
    this.frames.forEach((m, i) => {
      const u = m.uniforms;
      const kept = t >= times[i] - 1e-3;            // a frame enters memory when the video reaches it
      const age = Math.max(0, (t - times[i]) / step); // in frame steps behind now, continuous while playing
      const vis = kept ? smooth(Math.min(1, Math.max(0, s.stack * 1.6 - age * 0.12))) : 0;
      u.uVisible.value = damp(u.uVisible.value, vis, kept ? 6 : 12, dt);
      m.mesh.visible = u.uVisible.value > 0.01;
      if (!m.mesh.visible) return;
      u.uIntro.value = 2;
      u.uRelief.value = 0.12 * Math.min(1, age / 0.3 + q); // a just-kept frame peels off the flat live plate
      u.uZ.value = -age * FRAME_GAP * s.stack;
      const reach = this.memoryMasked && age > 0.5 ? 0 : 1; // masked: the lifted tokens drop back
      u.uFocus.value = damp(u.uFocus.value, 0.85 * s.stack * (age > 0.5 ? reach : 0.5) * q, 5, dt);
      u.uLift.value = damp(u.uLift.value, 2.2 * s.stack * reach * q, 5, dt);
      u.uStale.value = age < 0.5 ? 0 : (0.25 + age * 0.22) * s.stack * q;
      u.uAlpha.value = 1 - age * 0.12;
    });
    if (this.live) {
      const u = this.live.uniforms;
      const ready = (this.clip?.readyState ?? 0) >= 2; // no frame decoded yet: no plate (it would be blank)
      u.uVisible.value = damp(u.uVisible.value, end || !ready ? 0 : smooth(Math.min(1, s.stack * 1.6)), end ? 10 : 6, dt);
      this.live.mesh.visible = u.uVisible.value > 0.01;
      u.uIntro.value = 2;
      u.uRelief.value = 0;
      u.uZ.value = 0.3 * s.stack; // just in front, so a frame being kept slides out from behind it
      u.uFocus.value = 0; u.uLift.value = 0; u.uStale.value = 0; u.uAlpha.value = 1;
    }
  }

  /** Outlines of the hand-labeled regions, floating just above the tallest tile inside each. */
  private updateBoxes(m: TokenMesh, dt: number) {
    const s = this.s;
    const show = smooth(Math.max(0, (s.focus - 0.5) * 2)) * (1 - s.hover) * (1 - s.stack) * (1 - s.mosaic);
    this.boxMat.opacity = show * 0.85;
    const boxes = this.override?.boxes ?? [this.data[this.current].questions[this.question].box];
    this.boxLines.forEach((line, bi) => {
      const box = boxes[bi];
      line.visible = show > 0.01 && !!box;
      if (!line.visible || !box) return;
      const [x0, y0, x1, y1] = box;
      const c0 = Math.floor(x0 * m.gw), c1 = Math.ceil(x1 * m.gw), r0 = Math.floor(y0 * m.gh), r1 = Math.ceil(y1 * m.gh);
      let top = 0;
      for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) top = Math.max(top, m.tokenPosition(r * m.gw + c, this.tmpB).z);
      this.boxZ[bi] = damp(this.boxZ[bi], top + 0.25, 6, dt);
      const X0 = c0 - m.gw / 2 - 0.05, X1 = c1 - m.gw / 2 + 0.05, Y0 = m.gh / 2 - r0 + 0.05, Y1 = m.gh / 2 - r1 - 0.05, z = this.boxZ[bi];
      (line.geometry as LineGeometry).setPositions([X0, Y0, z, X1, Y0, z, X1, Y1, z, X0, Y1, z, X0, Y0, z]);
      this.boxAnchors[bi].set(X0, Y0, z);
    });
  }

  private tmpA = new THREE.Vector3();
  private tmpB = new THREE.Vector3();
  private tmpC = new THREE.Vector3();

  private updateArcs(m: TokenMesh, dt: number) {
    const s = this.s;
    // Lines mean attention only; a similarity query (hover) hides them.
    const show = smooth(Math.max(0, (s.focus - 0.5) * 2)) * (1 - s.hover) * (1 - s.stack) * (1 - s.mosaic)
      * Math.min(1, Math.max(0, this.intro - 1));
    this.arcProgress = Math.min(1, this.arcProgress + dt * 1.6);
    this.arcMat.opacity = show * 0.9;

    const src = this.tmpA.set(0, m.gh / 2 + 2.2, 5.5);
    const qm = this.queryToken.material as THREE.MeshBasicMaterial;
    qm.opacity = damp(qm.opacity, show, 8, dt);
    this.queryToken.position.copy(src);
    this.queryToken.rotation.set(0.3, performance.now() * 0.0006, 0);

    // In the memory stack the query reaches back into past frames instead.
    const memory = this.frames.length > 0 && s.stack > 0.5;
    if (memory) {
      const front = this.frames[this.frames.length - 1];
      src.set(front.gw * 0.28, front.gh / 2 + 2.6, 4);
      this.queryToken.position.copy(src);
      qm.opacity = damp(qm.opacity, smooth((s.stack - 0.5) * 2) * this.asked, 8, dt);
      this.arcMat.opacity = smooth((s.stack - 0.5) * 2) * 0.9 * (this.memoryMasked ? 0.12 : 1) * this.asked;
    }
    const pairs: [TokenMesh, number][] = memory
      ? this.frames.slice(0, -1).flatMap((fm) => topK(fm.target, 3).map((i) => [fm, i] as [TokenMesh, number])).slice(0, ARC_K)
      : topK(m.target, ARC_K).map((i) => [m, i]);
    this.arcs.forEach((line, k) => {
      const [tm, ti] = pairs[k] ?? [m, undefined];
      line.visible = (memory || show > 0.01) && ti !== undefined;
      if (!line.visible || ti === undefined) return;
      const dst = tm.tokenPosition(ti, this.tmpB);
      const ctl = this.tmpC.copy(src).add(dst).multiplyScalar(0.5);
      ctl.z += 4 + src.distanceTo(dst) * 0.25;
      const pts: number[] = [];
      const end = smooth(Math.min(1, this.arcProgress * 1.4 - k * 0.04));
      for (let j = 0; j < ARC_SEG; j++) {
        const u = (j / (ARC_SEG - 1)) * Math.max(0.001, end);
        const a = (1 - u) * (1 - u), b = 2 * (1 - u) * u, c = u * u;
        pts.push(a * src.x + b * ctl.x + c * dst.x, a * src.y + b * ctl.y + c * dst.y, a * src.z + b * ctl.z + c * dst.z);
      }
      (line.geometry as LineGeometry).setPositions(pts);
    });
  }

  private lastHoverKey = '';
  private lastStateKey = '';
  private project(v: THREE.Vector3) {
    const p = this.tmpC.copy(v).project(this.camera);
    return { x: (p.x * 0.5 + 0.5) * this.w, y: (-p.y * 0.5 + 0.5) * this.h };
  }

  private emitEvents(act: TokenMesh) {
    if (this.events.state) {
      const st: FieldState = {
        chapter: Math.round(this.chapter),
        query: this.qcell,
        pinned: !!this.pinned && !this.hoverCell,
        yawDeg: Math.round((this.viewYaw * 180) / Math.PI),
        pitchDeg: Math.round((this.viewPitch * 180) / Math.PI),
        dragging: this.drag.on,
      };
      const key = JSON.stringify(st);
      if (key !== this.lastStateKey) { this.lastStateKey = key; this.events.state(st); }
    }
    if (this.events.boxes)
      this.events.boxes(this.boxMat.opacity > 0.3
        ? this.boxLines.flatMap((l, i) => (l.visible ? [{ i, ...this.project(this.boxAnchors[i]) }] : []))
        : []);
    if (this.events.frames) {
      const list: { frame: number; x: number; y: number }[] = [];
      const at = (fm: TokenMesh) => this.project(this.tmpA.set(fm.gw / 2 - 1, fm.gh / 2 + 0.9, fm.uniforms.uZ.value));
      if (this.s.stack > 0.6) {
        const live = !!this.live && this.live.uniforms.uVisible.value >= 0.5;
        this.frames.forEach((fm, i) => {
          // A frame still sliding out from behind the live plate would share its label spot: label it once it's clear.
          const behind = -fm.uniforms.uZ.value / (FRAME_GAP * this.s.stack);
          if (fm.uniforms.uVisible.value >= 0.5 && !(live && behind < 0.6)) list.push({ frame: i, ...at(fm) });
        });
        if (live) list.push({ frame: -1, ...at(this.live!) });
      }
      this.events.frames(list);
    }
    if (this.events.hover) {
      const c = this.qcell;
      const key = c ? `${act.data.id}:${c[0]},${c[1]}` : '';
      if (key !== this.lastHoverKey) {
        this.lastHoverKey = key;
        const i = c ? c[1] * act.gw + c[0] : 0;
        this.events.hover(c ? { col: c[0], row: c[1], depth: act.data.depth[i] } : null);
      }
    }
    if (this.events.query) {
      const op = (this.queryToken.material as THREE.MeshBasicMaterial).opacity;
      this.events.query(op < 0.05 ? null : this.project(this.queryToken.position));
    }
  }
}
