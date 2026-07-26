# Deploy: kernels.rudrite.com

Production hosting is Cloudflare Pages, same pattern as the other rudrite wings.

In the Cloudflare dashboard → Pages → connect `rudrite/kernels`:

| setting | value |
|---|---|
| Production branch | `main` |
| Root directory | `site` |
| Build command | `npm run build` |
| Build output directory | `dist` |
| Node version | from `.nvmrc` (22) |

Then add the custom domain `kernels.rudrite.com` (Cloudflare auto-manages the CNAME). Every push to `main` triggers a build; the build's firewall gate runs inside it, so the served `dist/` is verified clean before going live. GitHub Actions (`.github/workflows/ci.yml`) runs the same test + build + firewall on every push and PR as the pre-merge check.
