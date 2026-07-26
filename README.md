# kernels

Rudrite's TPU kernel wing: a public, lab-driven curriculum for learning to write fast TPU kernels, from the machine's first principles to distributed Pallas, built while learning it, in the open; checkpoints only count when a real chip measured them.

Site: **live** at [kernels.rudrite.com](https://kernels.rudrite.com) · Cloudflare Pages, deploys on every push to main

- `CURRICULUM.md`: the 14-week track. Six stages, each gated by a measurable checkpoint.
- `labs/`: runnable notebooks, one per lab. Open directly in Colab from GitHub (each notebook carries an Open in Colab badge); save a copy to your own account to keep your work.
- `bench/`: every number the site shows, with provenance (chip generation, dtype, shapes, date, commit).
- `site/`: kernels.rudrite.com, the presentation layer. The repo stands alone without it.

Status: Stage 0 in progress. All 13 labs are authored, runnable, and Colab-linked (every notebook executes end to end off-chip: interpret mode for kernels, simulated devices for the distributed labs). Bench holds roofline predictions and off-chip verification records; hardware rows replace them as gates are attempted.
