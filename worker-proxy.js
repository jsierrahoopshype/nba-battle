export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, OPTIONS',
          'Access-Control-Allow-Headers': '*',
          'Access-Control-Max-Age': '86400',
        },
      });
    }

    // Match /headshot/{slug}.png — proxy from GitHub Pages with CORS
    const match = url.pathname.match(/^\/headshot\/(.+)$/);
    if (match) {
      const slug = match[1];
      const upstream = `https://jsierrahoopshype.github.io/nba-headshots/players/headshots/face/${slug}`;

      try {
        const resp = await fetch(upstream, { cf: { cacheTtl: 86400 } });
        if (!resp.ok) {
          return new Response('Not found', { status: 404 });
        }

        const headers = new Headers(resp.headers);
        headers.set('Access-Control-Allow-Origin', '*');
        headers.set('Cache-Control', 'public, max-age=86400');

        return new Response(resp.body, {
          status: resp.status,
          headers,
        });
      } catch (e) {
        return new Response('Proxy error: ' + e.message, { status: 502 });
      }
    }

    // All other requests — serve static assets
    const asset = await env.ASSETS.fetch(request);
    return asset;
  },
};
