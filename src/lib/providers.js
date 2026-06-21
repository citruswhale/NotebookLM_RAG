// Provider abstraction: Google Gemini is the PRIMARY provider; Groq and OpenAI
// are chat BACKUPS (tried in that order if Gemini fails).
// All API keys are read from environment variables (process.env) — never hardcoded.
//   - GOOGLE_API_KEY   (primary; chat + embeddings)
//   - GROQ_API_KEY     (chat backup 1 — LLM inference only, no embeddings)
//   - OPENAI_API_KEY   (chat backup 2; also embeddings backup if no Google key)
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { ChatGroq } from "@langchain/groq";

const hasGoogle = () => Boolean(process.env.GOOGLE_API_KEY);
const hasGroq = () => Boolean(process.env.GROQ_API_KEY);
const hasOpenAI = () => Boolean(process.env.OPENAI_API_KEY);

// ---------------------------------------------------------------------------
// EMBEDDINGS
// ---------------------------------------------------------------------------
// IMPORTANT: embeddings must be STABLE across indexing (/api/upload) and
// querying (/api/chat). If the query were embedded by a different model than the
// stored documents, the two vectors would live in incompatible spaces and the
// similarity search would be meaningless (and Qdrant dimensions would mismatch).
//
// So we DELIBERATELY do NOT do try/catch fallback for embeddings. We pick ONE
// provider deterministically from the environment and use it everywhere. The
// OpenAI "backup" only takes over for embeddings if Google is not configured at
// all — and in that case the whole collection is also indexed with OpenAI, so it
// stays consistent.
export function getEmbeddings() {
  if (hasGoogle()) {
    return new GoogleGenerativeAIEmbeddings({
      model: process.env.GOOGLE_EMBEDDING_MODEL || "gemini-embedding-2",
    });
  }
  if (hasOpenAI()) {
    return new OpenAIEmbeddings({
      model: process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small",
    });
  }
  throw new Error(
    "No embedding provider configured. Set GOOGLE_API_KEY (primary) or OPENAI_API_KEY (backup)."
  );
}

export function getEmbeddingProviderName() {
  if (hasGoogle()) return "google";
  if (hasOpenAI()) return "openai";
  return "none";
}

// ---------------------------------------------------------------------------
// CHAT / REASONING MODELS
// ---------------------------------------------------------------------------
// "fast: true" selects the small/cheap model (SLM) used for the auxiliary
// reasoning steps (query rewriting, sub-query generation, LLM-as-judge).
// "fast: false" selects the main generation model.
function googleChat({ temperature = 0, fast = false } = {}) {
  return new ChatGoogleGenerativeAI({
    model: fast
      ? process.env.GOOGLE_SLM_MODEL || "gemini-2.5-flash-lite"
      : process.env.GOOGLE_LLM_MODEL || "gemini-2.5-flash-lite",
    temperature,
    maxRetries: 2,
  });
}

function groqChat({ temperature = 0, fast = false } = {}) {
  return new ChatGroq({
    model: fast
      ? process.env.GROQ_SLM_MODEL || "llama-3.1-8b-instant"
      : process.env.GROQ_LLM_MODEL || "llama-3.3-70b-versatile",
    temperature,
    maxRetries: 2,
  });
}

function openaiChat({ temperature = 0, fast = false } = {}) {
  return new ChatOpenAI({
    model: fast
      ? process.env.OPENAI_SLM_MODEL || "gpt-4.1-nano"
      : process.env.OPENAI_LLM_MODEL || "gpt-4.1-mini",
    temperature,
    maxRetries: 2,
  });
}

// Unlike embeddings, chat/reasoning calls are self-contained, so it IS safe to
// fall back between providers per-call. Primary = Google (Gemini); on any error
// (quota, outage, missing key) we transparently retry the SAME prompt on OpenAI.
export async function chatComplete({ system, user, temperature = 0, fast = false }) {
  const messages = [];
  if (system) messages.push(["system", system]);
  messages.push(["user", user]);

  // Order = fallback order: primary first, then backups. Groq is placed ahead of
  // OpenAI as the first backup (fast + free-tier friendly).
  const providers = [];
  if (hasGoogle()) providers.push({ name: "google", make: googleChat });
  if (hasGroq()) providers.push({ name: "groq", make: groqChat });
  if (hasOpenAI()) providers.push({ name: "openai", make: openaiChat });
  if (providers.length === 0) {
    throw new Error(
      "No chat provider configured. Set GOOGLE_API_KEY (primary), or GROQ_API_KEY / OPENAI_API_KEY (backup)."
    );
  }

  let primaryError;
  for (const p of providers) {
    try {
      const model = p.make({ temperature, fast });
      const res = await model.invoke(messages);
      const text = typeof res.content === "string" ? res.content : String(res.content);
      return { text, provider: p.name };
    } catch (err) {
      // Remember the PRIMARY (first) provider's error. When every provider
      // fails, that's the more actionable signal to surface (e.g. "Gemini is
      // rate-limited"), rather than the backup's failure (e.g. a bad backup key).
      if (!primaryError) primaryError = err;
      console.error(`[providers] ${p.name} chat failed; falling back if possible:`, err.message);
    }
  }
  throw primaryError;
}
