export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Proxy headshot images with CORS headers
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
