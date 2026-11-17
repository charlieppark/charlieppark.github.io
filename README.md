# charlieppark.kr

Source of [charlieppark.kr](https://charlieppark.kr): an [Astro](https://astro.build) site with a Three.js token field
on the home page. Pushing to `main` builds and deploys it with GitHub Actions (`.github/workflows/deploy.yml`).

```sh
npm install
npm run dev        # http://localhost:4321
npm run build      # static site in dist/
npm run test:interaction -- 1440 900   # interaction checks against a running dev server (SITE_URL to override)
```

- Content: `src/data/` (publications, news, CV, model outputs for the home-page figures).
- Model outputs (Qwen2.5-VL-7B attention, answers, head masking) are produced offline and stored as data;
  nothing runs a model in the browser.
- The previous Jekyll site is kept on the `jekyll-site` branch.
