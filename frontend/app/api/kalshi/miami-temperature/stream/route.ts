export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function GET(request: Request) {
  const upstreamUrl = new URL(
    "/api/kalshi/miami-temperature/stream",
    backendUrl,
  );

  try {
    const upstream = await fetch(upstreamUrl, {
      headers: {
        Accept: "text/event-stream",
        "Cache-Control": "no-cache",
      },
      cache: "no-store",
      signal: request.signal,
    });

    if (!upstream.ok || !upstream.body) {
      return Response.json(
        { error: `Backend stream returned ${upstream.status}.` },
        { status: 502 },
      );
    }

    return new Response(upstream.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Backend stream unavailable.";
    return Response.json({ error: message }, { status: 502 });
  }
}
