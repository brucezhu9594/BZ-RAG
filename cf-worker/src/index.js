/**
 * BZ-RAG 金丝雀路由 Worker
 *
 * 按 KV 中 canary_weight (0-100) 把流量分给 stable / canary 两个 Railway service。
 * 概念上对齐 Mira 的 cf-lb-weights.sh —— 只是把 Cloudflare Load Balancer 的 pool
 * 权重换成了 Workers + KV 自实现，因为我们没有付费 LB 套餐。
 */

export default {
  async fetch(request, env) {
    const expected = env.EDGE_AUTH_TOKEN;
    const provided = request.headers.get("x-edge-auth");
    if (!expected || provided !== expected) {
      return new Response("Unauthorized", { status: 401 });
    }

    let weight = 0;
    try {
      const raw = await env.BZ_RAG_CANARY.get("canary_weight");
      const n = parseInt(raw, 10);
      if (!Number.isNaN(n) && n >= 0 && n <= 100) weight = n;
    } catch (_) {
      weight = 0;
    }

    const isCanary = Math.random() * 100 < weight;
    const backend = isCanary ? env.CANARY_URL : env.STABLE_URL;
    const target = isCanary ? "canary" : "stable";

    const url = new URL(request.url);
    const upstreamUrl = backend.replace(/\/$/, "") + url.pathname + url.search;

    const headers = new Headers(request.headers);
    headers.delete("x-edge-auth");
    headers.set("host", new URL(backend).host);

    const upstreamResp = await fetch(upstreamUrl, {
      method: request.method,
      headers,
      body: request.body,
      redirect: "manual",
    });

    const respHeaders = new Headers(upstreamResp.headers);
    respHeaders.set("x-bz-backend", target);
    respHeaders.set("x-bz-canary-weight", String(weight));

    return new Response(upstreamResp.body, {
      status: upstreamResp.status,
      statusText: upstreamResp.statusText,
      headers: respHeaders,
    });
  },
};
