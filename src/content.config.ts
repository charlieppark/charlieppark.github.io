import { defineCollection } from 'astro:content';
import { file, glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { load as loadYaml } from 'js-yaml';

const publications = defineCollection({
  loader: file('src/data/publications.yaml'),
  schema: z.object({
    title: z.string(),
    authors: z.array(z.string()),
    me: z.number(),
    equal: z.array(z.number()).optional(),
    venue: z.string(),
    venueFull: z.string(),
    year: z.number(),
    date: z.string(), // YYYY-MM of the venue, for ordering
    note: z.string().optional(),
    thumb: z.string(),
    topics: z.array(z.string()),
    links: z.object({
      paper: z.url().optional(),
      page: z.url().optional(),
      code: z.url().optional(),
      bibtex: z.url().optional(),
    }),
  }),
});

const news = defineCollection({
  // Entries carry no id in the YAML; index them so order is preserved.
  loader: file('src/data/news.yaml', {
    parser: (text) =>
      (loadYaml(text) as Record<string, unknown>[]).map((n, i) => ({ id: String(i).padStart(3, '0'), ...n })),
  }),
  schema: z.object({
    date: z.string(),
    text: z.string(),
    paper: z.string().optional(),
  }),
});

const notes = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: 'src/content/notes' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { publications, news, notes };
