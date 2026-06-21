// Maps raw provider/database errors to a correct HTTP status + a clear, graceful
// user-facing message. The goal: a 429 rate-limit must NOT show up as "couldn't
// read your PDF" or "no information found" — the user should be told what really
// happened and whether retrying will help.
//
// It also avoids leaking internals (stack traces, API keys) to the client.

function blob(error) {
  const parts = [
    error?.message,
    error?.name,
    error?.code,
    error?.status,
    error?.response?.status,
    error?.response?.data && JSON.stringify(error.response.data),
  ].filter(Boolean);
  return parts.join(" | ").toLowerCase();
}

export function classifyError(error) {
  const s = blob(error);
  const httpStatus = error?.status || error?.response?.status;

  // --- Rate limit / quota exhausted ---------------------------------------
  if (
    httpStatus === 429 ||
    s.includes("429") ||
    s.includes("rate limit") ||
    s.includes("ratelimit") ||
    s.includes("too many requests") ||
    s.includes("quota") ||
    s.includes("resource_exhausted") ||
    s.includes("resource exhausted") ||
    s.includes("insufficient_quota")
  ) {
    return {
      status: 429,
      kind: "rate_limit",
      message:
        "The AI service is currently rate-limited or out of quota. Please wait a few seconds and try again.",
    };
  }

  // --- Auth / credentials (server-side misconfig — don't leak details) -----
  if (
    httpStatus === 401 ||
    httpStatus === 403 ||
    s.includes("401") ||
    s.includes("403") ||
    s.includes("api key") ||
    s.includes("api_key") ||
    s.includes("unauthenticated") ||
    s.includes("permission denied") ||
    s.includes("invalid authentication") ||
    s.includes("incorrect api key")
  ) {
    return {
      status: 503,
      kind: "auth",
      message:
        "The server's AI credentials are missing or invalid. Please contact the administrator.",
    };
  }

  // --- Timeouts / network --------------------------------------------------
  if (
    s.includes("timeout") ||
    s.includes("timed out") ||
    s.includes("etimedout") ||
    s.includes("econnreset") ||
    s.includes("econnrefused") ||
    s.includes("enotfound") ||
    s.includes("fetch failed") ||
    s.includes("network") ||
    httpStatus === 504
  ) {
    return {
      status: 504,
      kind: "network",
      message:
        "The AI service is temporarily unreachable or timed out. Please try again in a moment.",
    };
  }

  // --- Model overloaded / service unavailable ------------------------------
  if (
    httpStatus === 500 ||
    httpStatus === 502 ||
    httpStatus === 503 ||
    s.includes("overloaded") ||
    s.includes("unavailable") ||
    s.includes("service is currently")
  ) {
    return {
      status: 503,
      kind: "overloaded",
      message:
        "The AI service is temporarily overloaded. Please try again shortly.",
    };
  }

  // --- Fallback ------------------------------------------------------------
  return {
    status: 500,
    kind: "unknown",
    message: "Something went wrong while processing your request. Please try again.",
  };
}
