// Interaction correctness tests for the home-page Token Field (needs `npm run dev` running).
// Usage: node tests/interaction.mjs [width height]   e.g. node tests/interaction.mjs 390 844

import puppeteer from 'puppeteer-core';
import { readFileSync } from 'node:fs';
const fig1 = JSON.parse(readFileSync(new URL('../src/data/fig1.json', import.meta.url)));
const W = +(process.argv[2] || 1440), H = +(process.argv[3] || 900);
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new', args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist'] });
const page = await browser.newPage();
const errs = [];
page.on('pageerror', (e) => errs.push(e.message));
page.on('console', (m) => m.type() === 'error' && errs.push(m.text()));
await page.setViewport({ width: W, height: H, hasTouch: false });
await page.goto(process.env.SITE_URL ?? 'http://localhost:4321/', { waitUntil: 'networkidle0' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await sleep(3200);
const tf = (fn, ...a) => page.evaluate(fn, ...a);
const results = [];
const ok = (name, pass, detail) => { results.push({ name, pass, detail }); };

const gotoChapter = async (i) => {
  // Chapter 0: top of the page, where visitors land. Desktop: chapter centered.
  // Mobile: the instrument card sits at the bottom, field above it.
  const y = await tf((i) => {
    if (i === 0) return 0;
    const e = document.querySelector(`[data-chapter="${i}"]`);
    const inst = e.querySelector('.inst');
    if (innerWidth < 900 && inst) return inst.getBoundingClientRect().bottom + scrollY - innerHeight + 24;
    const r = e.getBoundingClientRect(); return r.top + scrollY + r.height / 2 - innerHeight / 2;
  }, i);
  await tf((y) => scrollTo(0, y), y); await sleep(2200);
};
const stageRect = () => tf(() => { const r = document.querySelector('[data-canvas]').getBoundingClientRect(); return { left: r.left, top: r.top }; });
const tokenScreen = (i) => tf((i) => window.__tokenField.debugTokenScreen(i), i);
const hovered = () => tf(() => window.__tokenField.debugHover);
const elAt = (x, y) => tf((x, y) => { const e = document.elementFromPoint(x, y); return e?.matches('[data-canvas]') ? 'canvas' : e?.tagName; }, x, y);

// ── Chapter 0: poke, "whose left?" on hover, and the grounding test under head masking (no orbit there by design).
await gotoChapter(0);
{
  const portraitAt = (c, r) => tf((c, r) => window.__tokenField.portraitScreen(c, r, true), c, r);
  const sr = await stageRect();
  const at = async (c, r) => { const p = await portraitAt(c, r); return [sr.left + p.x, sr.top + p.y]; };
  const shown = (sel) => page.$eval(sel, (e) => (e.hidden ? '' : e.textContent));
  const poked = () => tf(() => window.__tokenField.debugPoked);
  await page.mouse.click(...(await at(9, 12))); await sleep(250); // my neck: no labeled region → plain poke
  const p0 = await poked();
  ok('ch0 click knocks tokens loose', p0 > 0.5, `max displacement ${p0.toFixed(2)}`);
  await sleep(4000);
  const home = await poked();
  ok('ch0 knocked tokens return home', home < 0.05, `after 4s: ${home.toFixed(3)}`);

  // Move = viewpoint: on the plant, the model's answers (✓ in the photo, ✗ from my side) and my own.
  await page.mouse.move(...(await at(15, 3))); await sleep(500);
  const mt = await shown('[data-mtag]'), say = await shown('[data-bubble]');
  ok('ch0 plant: ✓ in the photo, ✗ from my side', /✓/.test(mt) && /✗/.test(mt) && /my left/.test(say), `${mt} | ${say}`);

  // Click = verify: clicking my glasses runs the paper's grounding test; the heads panel switches heads off.
  // Each condition must show exactly the model's recorded result (fig1.json → portrait.find).
  const glasses = fig1.portrait.find.find((f) => f.id === 'glasses');
  const mark = (r) => (!r.box ? '✗ no box for' : r.iou >= 0.5 ? '✓ on' : r.iou >= 0.1 ? '◐ partly on' : '✗ off');
  const panel = () => tf(() => {
    // The visible copy of the heads panel (phones hide theirs until something is found).
    const p = [...document.querySelectorAll('[data-vrh]')].find((e) => e.offsetParent) ?? document.querySelector('.stage-vrh');
    return p && { res: p.querySelector('[data-vres]').textContent, vOff: p.querySelectorAll('rect.v.off').length,
      rOff: p.querySelectorAll('rect.r.off').length, lit: p.querySelectorAll('rect.v:not(.off)').length,
      pred: document.querySelector('[data-vpred]').classList.contains('on'),
      lines: getComputedStyle(document.querySelector('[data-vlines]')).display !== 'none' ? document.querySelectorAll('[data-vlines] path[d]').length : 0 };
  });
  const condBtn = (c) => tf((c) => [...document.querySelectorAll('[data-vrh]')].find((e) => e.offsetParent).querySelector(`[data-cond="${c}"]`).click(), c);
  await page.mouse.click(...(await at(9, 8))); await sleep(700);
  let s = await panel();
  ok('ch0 glasses, all heads: model box shown as recorded', s?.pred && s.res.includes(mark(glasses.all)) && s.lit === 20 && (W < 900 || s.lines === 20), JSON.stringify(s));
  await condBtn('vrh'); await sleep(500);
  s = await panel();
  ok('ch0 glasses, 20 VRHs off: heads dark, box as recorded', s.vOff === 20 && s.res.includes(mark(glasses.vrh)) && s.lines === 0, JSON.stringify(s));
  await condBtn('random'); await sleep(500);
  s = await panel();
  // The paper's random 20 (random.Random(42)) includes one retrieval head; it must show as switched off too.
  const shared = fig1.portrait.randomHeads.filter(([l, h]) => fig1.portrait.heads.some(([L, H]) => L === l && H === h)).length;
  ok('ch0 glasses, 20 random off: exactly those 20 dark, box as recorded',
    s.rOff + s.vOff === 20 && s.vOff === shared && s.lit === 20 - shared && s.res.includes(mark(glasses.random)) && (W < 900 || s.lines === 20 - shared),
    JSON.stringify({ ...s, shared }));
  await page.mouse.click(...(await at(9, 12))); await sleep(500); // a plain poke clears it
  s = await panel();
  ok('ch0 plain poke clears the find', !s.pred && s.lit === 20 && s.vOff === 0, JSON.stringify(s));
}

for (const ch of [1, 2]) {
  await gotoChapter(ch);
  // ── Picking accuracy: hover each sampled token's projected front-face center.
  const { grid } = await tokenScreen(0);
  const n = grid[0] * grid[1];
  let hits = 0, tried = 0, near = 0; const misses = [];
  for (let k = 0; k < 40; k++) {
    const i = Math.floor(((k * 0.618034) % 1) * n);
    const sr = await stageRect();
    const p = await tokenScreen(i);
    const x = sr.left + p.x, y = sr.top + p.y;
    if (x < 5 || y < 5 || x > W - 5 || y > H - 5 || (await elAt(x, y)) !== 'canvas') continue;
    await page.mouse.move(x, y); await sleep(140);
    const h = await hovered(); tried++;
    if (h && h[1] * grid[0] + h[0] === i) hits++;
    else if (h && Math.abs(h[0] - (i % grid[0])) <= 1 && Math.abs(h[1] - Math.floor(i / grid[0])) <= 1) { near++; misses.push([i, h]); }
    else misses.push([i, h]);
  }
  ok(`ch${ch} hover picks token under cursor`, tried > 10 && hits / tried >= 0.9, `${hits}/${tried} exact, ${near} adjacent (occlusion by taller neighbour), misses: ${JSON.stringify(misses.slice(0, 4))}`);

  // ── Drag direction: grab the field, drag right → front faces move right; drag down → move down.
  const c = Math.floor(n / 2) + Math.floor(grid[0] / 2);
  const sr = await stageRect();
  let p0 = await tokenScreen(c);
  const gx = sr.left + p0.x, gy = sr.top + p0.y;
  if ((await elAt(gx, gy)) === 'canvas') {
    await page.mouse.move(gx, gy); await sleep(500);
    p0 = await tokenScreen(c);
    const f0 = await tf(() => window.__tokenField.debugFacing());
    await page.mouse.down();
    for (let s = 1; s <= 10; s++) { await page.mouse.move(gx + s * 18, gy); await sleep(16); }
    await sleep(500);
    const p1 = await tokenScreen(c);
    const f1 = await tf(() => window.__tokenField.debugFacing());
    await page.mouse.up(); await sleep(1500);
    const pR = await tokenScreen(c);
    await page.mouse.move(gx, gy); await sleep(400);
    const g0 = await tf(() => window.__tokenField.debugFacing());
    await page.mouse.down();
    for (let s = 1; s <= 10; s++) { await page.mouse.move(gx, gy + s * 15); await sleep(16); }
    await sleep(500);
    const g1 = await tf(() => window.__tokenField.debugFacing());
    await page.mouse.up();
    ok(`ch${ch} drag right turns field to face right`, f1.x > f0.x + 2, `facing.x ${f0.x.toFixed(1)} → ${f1.x.toFixed(1)}`);
    ok(`ch${ch} drag down tips top toward viewer`, g1.y > g0.y + 2, `facing.y ${g0.y.toFixed(1)} → ${g1.y.toFixed(1)}`);
    ok(`ch${ch} release springs back`, (Math.abs(pR.x - p0.x) < 3 || Math.abs(pR.x - p0.x) < Math.abs(p1.x - p0.x) * 0.5), `after 1.5s: ${(pR.x - p0.x).toFixed(1)}px off`);
    await sleep(1500);
  } else ok(`ch${ch} drag`, false, 'field center covered by ' + (await elAt(gx, gy)));

  // ── Click: Ch. 0 opens the paper the tile came from; elsewhere a ripple starts at the clicked token.
  {
    const i = Math.floor(n * 0.37) + 3;
    const sr2 = await stageRect(); const p = await tokenScreen(i);
    const x = sr2.left + p.x, y = sr2.top + p.y;
    if ((await elAt(x, y)) === 'canvas') {
      await page.mouse.move(x, y); await sleep(200);
      const h = await hovered();
      {
        await page.mouse.click(x, y); await sleep(50);
        const r = await tf(() => window.__tokenField.debugRipple);
        ok(`ch${ch} click ripple at hovered token`, h && r[0] === h[0] && r[1] === h[1], `hover=${JSON.stringify(h)} ripple=${r.slice(0, 2)}`);
      }
    }
  }
  await page.mouse.move(2, 2); await sleep(300);
}

// ── Scene switch while hovering a token outside the new grid (24×18 → 18×18). Picker lives in Ch. III.
await gotoChapter(3);
{
  const sr = await stageRect(); const p = await tokenScreen(23); // top-right token of a 24-wide grid
  await page.mouse.move(sr.left + p.x, sr.top + p.y); await sleep(300);
  await page.click('[data-scene="4"]'); await sleep(800);
  const h = await hovered();
  ok('scene switch clears/keeps valid hover', !h || (h[0] < 18 && h[1] < 18), JSON.stringify(h));
}
// ── Chapter IV: the clip plays on arrival; each frame enters memory when the playhead reaches its time, and the
// question (attention, answer, mask switch) only comes at the last frame. Scrubbing goes back in time.
await gotoChapter(4);
{
  const mem = () => tf(() => window.__tokenField.debugMemory);
  const times = fig1.video.frames.map((f) => f.t), tEnd = times[times.length - 1];
  const expectKept = (t) => times.filter((ft) => t >= ft - 1e-3).length;
  await sleep(1800);
  const m1 = await mem();
  const pending = await tf(() => [document.querySelector('[data-memmask]').disabled, document.querySelector('[data-memanswer]').textContent]);
  ok('ch4 plays on arrival: frames kept so far match the playhead, question not yet asked',
    m1.t > 0.3 && m1.t < tEnd && m1.kept === expectKept(m1.t) && m1.live && m1.asked < 0.1 && pending[0] && /comes at/.test(pending[1]),
    JSON.stringify({ ...m1, pending }));
  await sleep((tEnd - m1.t) * 1000 + 1500);
  const m2 = await mem();
  const done = await tf(() => [document.querySelector('[data-memmask]').disabled, document.querySelector('[data-memanswer]').textContent,
    document.querySelectorAll('[data-thumb-frame].kept').length]);
  ok('ch4 at the end: all frames kept, question asked, answer shown', m2.t === tEnd && m2.kept === times.length && !m2.live && m2.asked > 0.9
    && !done[0] && done[1].includes(fig1.video.answer.replace(/\.$/, '')) && done[2] === times.length, JSON.stringify({ ...m2, done }));
  await page.click('[data-thumb-frame="1"]'); await sleep(600);
  const m3 = await mem();
  ok('ch4 frame 2 thumbnail scrubs back to its time', Math.abs(m3.t - times[1]) < 0.01 && m3.kept === 2 && m3.asked < 0.5, JSON.stringify(m3));
}

// ── No horizontal overflow.
const ov = await tf(() => [document.documentElement.scrollWidth, innerWidth]);
ok('no horizontal overflow', ov[0] <= ov[1], `${ov[0]} vs ${ov[1]}`);
ok('no page errors', errs.length === 0, errs.slice(0, 3).join(' | '));

for (const r of results) console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}  — ${r.detail}`);
await browser.close();
