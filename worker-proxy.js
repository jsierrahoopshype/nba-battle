export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Proxy NBA CDN headshots with CORS headers
    if (url.pathname.startsWith('/nba-headshot/')) {
      const id = url.pathname.replace('/nba-headshot/', '');
      const imageUrl = `https://cdn.nba.com/headshots/nba/latest/1040x760/${id}.png`;

      try {
        const response = await fetch(imageUrl);
        const headers = new Headers(response.headers);
        headers.set('Access-Control-Allow-Origin', '*');
        headers.set('Cache-Control', 'public, max-age=86400');
        return new Response(response.body, {
          status: response.status,
          headers
        });
      } catch (e) {
        return new Response('Image not found', { status: 404 });
      }
    }

    // Proxy headshot images with CORS headers (legacy route)
    if (url.pathname.startsWith('/headshot/')) {
      const slug = url.pathname.replace('/headshot/', '');
      const imageUrl = `https://jsierrahoopshype.github.io/nba-headshots/players/headshots/face/${slug}`;

      try {
        const response = await fetch(imageUrl);
        const headers = new Headers(response.headers);
        headers.set('Access-Control-Allow-Origin', '*');
        headers.set('Cache-Control', 'public, max-age=86400');
        return new Response(response.body, {
          status: response.status,
          headers
        });
      } catch (e) {
        return new Response('Image not found', { status: 404 });
      }
    }

    // Serve static assets for everything else
    return env.ASSETS.fetch(request);
  }
};
