// What each visual mark means, per chapter. Rendered next to the field (desktop) and inside
// each chapter's instrument card (mobile). One mark = one meaning across the whole story:
// heat color/lift and lines always mean attention; outlines always mean similarity.

export type Swatch = 'tile' | 'outline' | 'heat' | 'line' | 'box' | 'gap' | 'depth' | 'stack' | 'fade';
export interface LegendItem { swatch: Swatch; text: string }

export const LEGEND: LegendItem[][] = [
  [
    { swatch: 'tile', text: 'tile = one visual token of my photo (28 × 28 px)' },
  ],
  [
    { swatch: 'heat', text: 'color · lift = attention from the answer (top-20 retrieval heads)' },
    { swatch: 'line', text: 'line = the 9 most-attended tokens' },
    { swatch: 'box', text: 'box = the people/objects the question is about (hand-labeled)' },
    { swatch: 'outline', text: 'outline = tokens represented like the one you point at (layer 16)' },
  ],
  [
    { swatch: 'tile', text: 'patch = an original token, pixels unchanged' },
    { swatch: 'depth', text: 'how far it sits out = estimated nearness' },
    { swatch: 'gap', text: 'gap = never seen from the original camera' },
  ],
  [
    { swatch: 'heat', text: 'color · lift = attention from the answer (top-20 retrieval heads)' },
    { swatch: 'line', text: 'line = the 9 most-attended tokens' },
    { swatch: 'box', text: 'box = the object asked about (hand-labeled)' },
  ],
  [
    { swatch: 'stack', text: 'front plate = the live video; behind = frames kept in memory, one every 1.5 s' },
    { swatch: 'heat', text: 'color · lift = attention from the question, asked at the last frame' },
    { swatch: 'line', text: 'line = attention reaching back in time' },
    { swatch: 'fade', text: 'faded = little attention (not needed for the answer)' },
  ],
];

export const legendHTML = (items: LegendItem[]) =>
  items.map((it) => `<li><i class="sw sw-${it.swatch}" aria-hidden="true"></i><span>${it.text}</span></li>`).join('');
