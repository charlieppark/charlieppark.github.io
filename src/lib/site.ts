import { load as loadYaml } from 'js-yaml';
import cvRaw from '../data/cv.yaml?raw';

export const SITE = {
  name: 'Chanho Park',
  alias: 'Charlie Park',
  url: 'https://charlieppark.kr',
  description:
    'Chanho Park is a Master’s student at KAIST Visual AI Group studying Spatial Intelligence: how vision-language models perceive, represent, and reason about the physical world.',
  email: 'charlieppark@kaist.ac.kr',
  links: {
    scholar: 'https://scholar.google.com/citations?user=iMG8GOgAAAAJ',
    github: 'https://github.com/charlieppark',
    linkedin: 'https://www.linkedin.com/in/charlieppark/',
    x: 'https://x.com/_charlieppark',
  },
};

export const NAV = [
  { href: '/papers/', label: 'papers' },
  // Unpublished until finished (pages renamed to src/pages/_graph.astro and src/pages/_notes/):
  // { href: '/graph/', label: 'graph' },
  // { href: '/notes/', label: 'notes' },
  { href: '/cv/', label: 'cv' },
];

type Entry = { org?: string; orgUrl?: string; role: string; detail?: string; place?: string; period?: string; items?: string[] };
export type CV = {
  interests: string[];
  education: Entry[];
  research: Entry[];
  work: Entry[];
  service: Entry[];
  leadership: Entry[];
};
export const cv = loadYaml(cvRaw) as CV;

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
/** "2026-09" → "Sep 2026" */
export const fmtMonth = (ym: string) => {
  const [y, m] = ym.split('-').map(Number);
  return `${MONTHS[m - 1]} ${y}`;
};
