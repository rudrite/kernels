// The sitemap, generated from the same registries the pages render from, so
// a chapter or lesson cannot ship without being listed here. Static surfaces
// are named once below; everything else derives.
import type { APIRoute } from 'astro'
import { PATH } from '../lib/path'
import { JAX_PATH } from '../data/jax/track'
import { XLA_PATH } from '../data/xla/track'
import { PT_PATH } from '../data/pytorch/track'
import { ALL_UNIT_LESSONS, lessonHref } from '../data/lessons'

const STATIC = [
  '/',
  '/routes',
  '/workshop',
  '/bench',
  '/specimen',
  '/gym',
  '/gym/kernels',
  '/gym/jax',
  '/gym/xla',
  '/gym/pytorch',
  '/mistakes',
  '/mistakes/kernels',
  '/mistakes/jax',
  '/mistakes/xla',
  '/mistakes/pytorch',
  '/jax',
  '/xla',
  '/pytorch',
]

export const GET: APIRoute = ({ site }) => {
  const chapters = [...PATH, ...JAX_PATH, ...XLA_PATH, ...PT_PATH].map((c) => c.href)
  const lessons = ALL_UNIT_LESSONS.flatMap((u) => u.lessons.map((l) => lessonHref(u.unit, l.id)))
  const urls = [...new Set([...STATIC, ...chapters, ...lessons])]
  const body =
    '<?xml version="1.0" encoding="UTF-8"?>\n' +
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' +
    urls.map((u) => `  <url><loc>${new URL(u, site).href}</loc></url>`).join('\n') +
    '\n</urlset>\n'
  return new Response(body, { headers: { 'Content-Type': 'application/xml' } })
}
