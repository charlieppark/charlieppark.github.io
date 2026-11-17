// One scene = one InstancedMesh of visual-token tiles. All motion (intro assembly,
// split-flap scene flips, depth relief, attention lift, ripples, memory decay, head turn)
// happens in the vertex shader; the CPU eases a handful of uniforms, the per-token
// attention attribute, and (Chapter 0 only) per-token physics offsets.
import * as THREE from 'three';
import type { SceneData } from './attention';

const TILE = 0.94;
const THICK = 0.32;

const vert = /* glsl */ `
  attribute vec2 aCell;
  attribute float aDepth;
  attribute float aAttn;
  attribute float aRand;
  attribute float aRetain;
  attribute vec3 aOff;   // physics displacement (poked tokens)
  attribute vec3 aSpin;  // physics rotation (euler)

  uniform vec2 uGrid;
  uniform float uIntro;
  uniform float uVisible;
  uniform float uRelief;
  uniform float uFocus;
  uniform float uLift;
  uniform float uZ;
  uniform float uStale;
  uniform vec3 uRipple;
  uniform vec2 uHover;
  uniform float uWarp;   // 1 = token warping view: thin patches that turn to face the camera
  uniform vec2 uLook;    // head turn (yaw, pitch) in radians, applied to tokens inside uHead
  uniform vec4 uHead;    // head ellipse: center (col, row), radii (cols, rows)
  uniform float uHeadZ;  // pivot depth behind the face

  varying vec2 vUv;
  varying vec2 vLocal;
  varying vec2 vUvC;
  varying vec3 vNormal;
  varying float vFront;
  varying float vAttn;
  varying float vFade;

  mat3 rotX(float a) { float c = cos(a), s = sin(a); return mat3(1., 0., 0., 0., c, s, 0., -s, c); }
  mat3 rotY(float a) { float c = cos(a), s = sin(a); return mat3(c, 0., -s, 0., 1., 0., s, 0., c); }
  mat3 rotZ(float a) { float c = cos(a), s = sin(a); return mat3(c, s, 0., -s, c, 0., 0., 0., 1.); }
  float easeIO(float t) { return t < .5 ? 4. * t * t * t : 1. - pow(-2. * t + 2., 3.) / 2.; }

  void main() {
    vec2 c = aCell;
    vec3 rnd = vec3(fract(aRand * 13.17), fract(aRand * 7.31), fract(aRand * 3.77)) - .5;

    // Intro: tiles fly in from a scattered cloud, staggered per tile.
    float ti = easeIO(clamp((uIntro - aRand * .6) / .9, 0., 1.));
    float sc = 1. - ti;

    // Split-flap flip on scene change, staggered left → right.
    float stagger = (c.x / uGrid.x) * .55 + aRand * .12;
    float vi = easeIO(clamp(uVisible * 1.67 - stagger, 0., 1.));

    mat3 R = rotY((1. - vi) * 1.5708) * rotX(sc * 6. * rnd.x) * rotY(sc * 6. * rnd.y);
    vec3 p = position;
    // Blocks extrude with nearness; in the warping view they are thin patches instead.
    p.z *= mix(1. + aDepth * uRelief * 3., .12, uWarp);
    vec3 n = normal;
    // Token warping: each token keeps its own pixels and only turns to face the viewer;
    // where it lands is set by its depth. Camera basis comes from the view matrix rows.
    mat3 face = mat3(
      vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]),
      vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]),
      vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]));
    p = mix(p, face * p, uWarp);
    n = normalize(mix(n, face * n, uWarp));
    // Physics spin, then the head turn for tokens inside the head ellipse.
    mat3 S = rotZ(aSpin.z) * rotY(aSpin.y) * rotX(aSpin.x);
    float hw = uHead.z > 0. ? 1. - smoothstep(.35, 1.5, length((c - uHead.xy) / uHead.zw)) : 0.; // soft edge: no seams
    mat3 T = rotY(uLook.x * hw) * rotX(uLook.y * hw);
    p = T * S * R * (p * vi);                    // fully hidden tiles collapse to nothing
    vNormal = normalize(T * S * R * n);

    vec3 world = vec3(c.x - uGrid.x * .5 + .5, -(c.y - uGrid.y * .5 + .5), 0.);
    world += rnd * vec3(70., 50., 60.) * sc;

    // Memory: stale (non-retained) tokens of older frames drift away and fade.
    float stale = uStale * (1. - aRetain);
    world += rnd * vec3(6., 6., 9.) * stale;

    float z = aDepth * uRelief * 7. + aAttn * uLift + uZ;
    float d = distance(c, uRipple.xy);
    float front = uRipple.z * 10.;
    z += sin(d * .9 - uRipple.z * 9.) * exp(-uRipple.z * 1.4) * exp(-pow(d - front, 2.) * .04) * 1.8;
    vec2 hv = c - uHover;
    z += exp(-dot(hv, hv) * .9) * .9;
    world.z += z;
    if (hw > 0.) {
      vec3 piv = vec3(uHead.x - uGrid.x * .5 + .5, -(uHead.y - uGrid.y * .5 + .5), -uHeadZ);
      world = piv + T * (world - piv);
    }
    world += aOff;

    vFront = step(.5, normal.z);
    vUv = (c + vec2(position.x / ${TILE.toFixed(2)} + .5, .5 - position.y / ${TILE.toFixed(2)})) / uGrid;
    vUvC = (c + .5) / uGrid;
    vLocal = vec2(position.x / ${TILE.toFixed(2)} + .5, .5 - position.y / ${TILE.toFixed(2)});
    vAttn = aAttn;
    vFade = clamp(1. - stale * .85, 0., 1.);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(world + p, 1.);
  }
`;

const frag = /* glsl */ `
  uniform sampler2D uTex;
  uniform vec3 uHeat[5];
  uniform vec3 uBg;
  uniform float uFocus;
  uniform float uAlpha;
  uniform float uSim;    // 1 = similarity query: outline tokens instead of heat-coloring them
  uniform vec3 uInk;
  uniform vec4 uEyes;    // eye centers in uv (left x,y, right x,y); zero = no eyes
  uniform float uEyeR;   // eye radius in uv
  uniform vec2 uGaze;    // pupil shift in uv
  uniform float uBlink;  // 0 open … 1 closed

  varying vec2 vUv;
  varying vec2 vLocal;
  varying vec2 vUvC;
  varying vec3 vNormal;
  varying float vFront;
  varying float vAttn;
  varying float vFade;

  vec3 heat(float t) {
    t = clamp(t, 0., 1.) * 4.;
    if (t < 1.) return mix(uHeat[0], uHeat[1], t);
    if (t < 2.) return mix(uHeat[1], uHeat[2], t - 1.);
    if (t < 3.) return mix(uHeat[2], uHeat[3], t - 2.);
    return mix(uHeat[3], uHeat[4], t - 3.);
  }

  void main() {
    vec2 uv = mix(vUvC, clamp(vUv, 0., 1.), vFront);
    float lash = 0.;
    if (uEyeR > 0. && vFront > .5) {
      // Eyes follow the cursor. The warp lives inside an ellipse shaped like the eye opening:
      // the center (the iris) moves rigidly, only the white around it stretches, and the lids,
      // skin and glasses outside the ellipse never move.
      // Blink: the upper lid slides down over the opening. Covered pixels show lid skin, sampled
      // from just above the eye and stretched downward, with a soft lash line at the lid's edge.
      vec2 uv0 = uv;
      for (int k = 0; k < 2; k++) {
        vec2 e = k == 0 ? uEyes.xy : uEyes.zw;
        vec2 d = uv0 - e;
        float bh = uEyeR * .33;                                   // half-height of the opening
        vec2 n = d / vec2(uEyeR, bh);
        float r = length(n);
        float f = 1. - smoothstep(.3, 1., r);
        uv -= uGaze * f;
        if (uBlink > .001) {
          // Lid edge (in units of bh): starts above the opening, ends as a soft downward curve.
          float edgeN = -1.15 + 2.1 * uBlink + .3 * n.x * n.x;
          float m = 1. - smoothstep(.85, 1.08, r);                // only the eye opening
          float cov = smoothstep(-.12, .12, edgeN - n.y) * m;
          // Lid skin: one row of skin just above the eye, so there is no vertical streaking.
          vec2 skin = vec2(uv0.x - uGaze.x * f, e.y - bh * 2.2);
          uv = mix(uv, skin, cov);
          lash = max(lash, (1. - smoothstep(0., .22, abs(edgeN - n.y))) * m * smoothstep(.2, .6, uBlink));
        }
      }
    }
    vec2 tuv = vec2(uv.x, 1. - uv.y);
    vec3 col = texture2D(uTex, tuv).rgb;
    col *= 1. - lash * .35;
    float light = .6 + .4 * max(dot(normalize(vNormal), normalize(vec3(-.35, .55, .9))), 0.);
    col *= mix(.72, 1., vFront) * light;

    float keep = smoothstep(.1, .65, vAttn);
    vec3 dim = mix(vec3(dot(col, vec3(.299, .587, .114))), uBg, .6);
    col = mix(col, mix(dim, col, keep), uFocus);
    // Attention → heat color. Similarity → ink outline (a different quantity gets a different mark).
    col = mix(col, heat(vAttn), uFocus * smoothstep(.15, 1., vAttn) * .6 * (1. - uSim));
    float edge = max(abs(vLocal.x - .5), abs(vLocal.y - .5)) * 2.;
    float ring = step(.74, edge) * vFront * step(.15, vAttn);
    col = mix(col, uInk, ring * uSim);

    gl_FragColor = vec4(col, uAlpha * vFade);
  }
`;

export class TokenMesh {
  readonly mesh: THREE.InstancedMesh;
  readonly uniforms: Record<string, THREE.IUniform>;
  readonly attn: Float32Array;
  readonly target: Float32Array;
  readonly gw: number;
  readonly gh: number;
  private attnAttr: THREE.InstancedBufferAttribute;
  /** Physics displacement / rotation per token (Chapter 0). Call physicsDirty() after writing. */
  readonly off: Float32Array;
  readonly spin: Float32Array;
  private offAttr: THREE.InstancedBufferAttribute;
  private spinAttr: THREE.InstancedBufferAttribute;
  physicsDirty() { this.offAttr.needsUpdate = true; this.spinAttr.needsUpdate = true; }

  constructor(readonly data: SceneData, tex: THREE.Texture, heat: THREE.Vector3[], bg: THREE.Vector3, ink: THREE.Vector3) {
    [this.gw, this.gh] = data.grid;
    const n = this.gw * this.gh;
    const geo = new THREE.BoxGeometry(TILE, TILE, THICK);
    const cell = new Float32Array(n * 2), depth = new Float32Array(n), rand = new Float32Array(n), retain = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      cell[i * 2] = i % this.gw;
      cell[i * 2 + 1] = Math.floor(i / this.gw);
      depth[i] = data.depth[i];
      rand[i] = (Math.sin(i * 91.7 + this.gw * 3.1) * 43758.5453) % 1;
      rand[i] = Math.abs(rand[i]);
    }
    this.attn = new Float32Array(n);
    this.target = new Float32Array(n);
    this.attnAttr = new THREE.InstancedBufferAttribute(this.attn, 1);
    this.attnAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aCell', new THREE.InstancedBufferAttribute(cell, 2));
    geo.setAttribute('aDepth', new THREE.InstancedBufferAttribute(depth, 1));
    geo.setAttribute('aRand', new THREE.InstancedBufferAttribute(rand, 1));
    geo.setAttribute('aRetain', new THREE.InstancedBufferAttribute(retain, 1));
    geo.setAttribute('aAttn', this.attnAttr);
    this.off = new Float32Array(n * 3);
    this.spin = new Float32Array(n * 3);
    this.offAttr = new THREE.InstancedBufferAttribute(this.off, 3).setUsage(THREE.DynamicDrawUsage);
    this.spinAttr = new THREE.InstancedBufferAttribute(this.spin, 3).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aOff', this.offAttr);
    geo.setAttribute('aSpin', this.spinAttr);

    this.uniforms = {
      uTex: { value: tex },
      uGrid: { value: new THREE.Vector2(this.gw, this.gh) },
      uIntro: { value: 0 },
      uVisible: { value: 0 },
      uRelief: { value: 0 },
      uFocus: { value: 0 },
      uLift: { value: 0 }, // height added per unit attention (question attention only, never hover)
      uZ: { value: 0 },
      uStale: { value: 0 },
      uAlpha: { value: 1 },
      uRipple: { value: new THREE.Vector3(0, 0, 100) },
      uHover: { value: new THREE.Vector2(-100, -100) },
      uHeat: { value: heat },
      uBg: { value: bg },
      uWarp: { value: 0 },
      uLook: { value: new THREE.Vector2() },
      uHead: { value: new THREE.Vector4() },
      uHeadZ: { value: 1.5 },
      uEyes: { value: new THREE.Vector4() },
      uEyeR: { value: 0 },
      uGaze: { value: new THREE.Vector2() },
      uBlink: { value: 0 },
      uSim: { value: 0 },
      uInk: { value: ink },
    };
    const mat = new THREE.ShaderMaterial({
      vertexShader: vert,
      fragmentShader: frag,
      uniforms: this.uniforms,
      transparent: true,
    });
    this.mesh = new THREE.InstancedMesh(geo, mat, n);
    this.mesh.frustumCulled = false;
  }

  setRetain(values: Float32Array) {
    const a = this.mesh.geometry.getAttribute('aRetain') as THREE.InstancedBufferAttribute;
    (a.array as Float32Array).set(values);
    a.needsUpdate = true;
  }

  /** Ease attention toward its target; returns true while still moving. */
  stepAttention(k: number) {
    let moving = false;
    for (let i = 0; i < this.attn.length; i++) {
      const d = this.target[i] - this.attn[i];
      if (Math.abs(d) > 1e-3) { this.attn[i] += d * k; moving = true; }
    }
    if (moving) this.attnAttr.needsUpdate = true;
    return moving;
  }

  /** World-space position of a token's front face (mirrors the vertex shader, minus transient effects). */
  tokenPosition(i: number, out: THREE.Vector3) {
    const u = this.uniforms;
    const x = i % this.gw, y = Math.floor(i / this.gw);
    const z = this.data.depth[i] * u.uRelief.value * 7 + this.attn[i] * u.uLift.value + u.uZ.value;
    const half = this.thickness(i) * 0.5;
    return out.set(x - this.gw / 2 + 0.5 + this.off[i * 3], -(y - this.gh / 2 + 0.5) + this.off[i * 3 + 1], z + half + this.off[i * 3 + 2]);
  }

  /** Tile thickness along z (mirrors the vertex shader). */
  thickness(i: number) {
    const u = this.uniforms, w = u.uWarp.value as number;
    const block = 1 + this.data.depth[i] * u.uRelief.value * 3;
    return THICK * (block + (0.12 - block) * w);
  }

  private box = new THREE.Box3();
  private hit = new THREE.Vector3();

  /** Index of the nearest tile whose box the ray hits, or -1. Mirrors tokenPosition (no transient effects). */
  pick(ray: THREE.Ray) {
    const u = this.uniforms;
    const p = new THREE.Vector3();
    let best = -1, bestD = Infinity;
    for (let i = 0; i < this.attn.length; i++) {
      this.tokenPosition(i, p); // front-face center
      const depth = this.thickness(i);
      this.box.min.set(p.x - 0.5, p.y - 0.5, p.z - depth);
      this.box.max.set(p.x + 0.5, p.y + 0.5, p.z);
      if (ray.intersectBox(this.box, this.hit)) {
        const d = this.hit.distanceToSquared(ray.origin);
        if (d < bestD) { bestD = d; best = i; }
      }
    }
    return best;
  }

  dispose() {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}
