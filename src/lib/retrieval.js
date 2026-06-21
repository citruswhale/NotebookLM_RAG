// Advanced / Corrective RAG retrieval pipeline.
//
// This module turns the BASE retrieval (embed query -> single vector search ->
// top-k) into an ADVANCED pipeline taught in the "RAG 2 - Advanced Retrieval
// Strategies" class:
//
//   1. Query rewriting / translation  (SLM fixes typos + adds context)
//   2. Sub-query / multi-query expansion (more angles on the same question)
//   3. (optional) HyDE - Hypothetical Document Embeddings (for public-data RAG)
//   4. Retrieve for every query, then RE-RANK via Reciprocal Rank Fusion
//      (the principled version of the "chunk frequency across sub-queries" idea)
//   5. LLM-as-a-judge grades each candidate chunk against the ORIGINAL query
//   6. Corrective loop: if nothing relevant survives, rewrite + retry (bounded)
//
// Every stage is individually toggleable via environment variables so the
// pipeline can be tuned (and the speed/accuracy trade-off dialled) without code
// changes.
import { chatComplete } from "@/lib/providers";

// ---- small helpers --------------------------------------------------------
function parseJsonLoose(text, fallback) {
  if (!text) return fallback;
  let t = text.trim().replace(/^```(?:json)?/i, "").replace(/```$/, "").trim();
  const start = t.search(/[[{]/);
  if (start === -1) return fallback;
  t = t.slice(start);
  try {
    return JSON.parse(t);
  } catch {
    return fallback;
  }
}

const flag = (name, def) => {
  const v = process.env[name];
  if (v === undefined) return def;
  return v === "true" || v === "1";
};
const num = (name, def) => {
  const v = parseInt(process.env[name], 10);
  return Number.isFinite(v) ? v : def;
};

// ---- 1. Query rewriting / translation -------------------------------------
// Fixes typos and adds light context. `feedback` is supplied on a corrective
// retry so the model knows the previous attempt missed.
export async function rewriteQuery(userQuery, feedback) {
  const system = `You are a query-rewriting assistant for a document retrieval system.
Your job:
- Fix any spelling mistakes / typos in the user's query.
- Expand abbreviations and add a little helpful context so the query retrieves better.
- Preserve the user's original intent. Do NOT answer the question.
Return ONLY the rewritten query as plain text — no quotes, no preamble.`;
  const user = feedback
    ? `Original query: "${userQuery}"\n\nThe previous retrieval was not relevant because: ${feedback}\nRewrite the query to retrieve more relevant document chunks.`
    : `Rewrite this query for better retrieval: "${userQuery}"`;
  try {
    const { text } = await chatComplete({ system, user, temperature: 0, fast: true });
    const out = (text || "").trim();
    return out || userQuery;
  } catch {
    return userQuery; // never block retrieval just because rewriting failed
  }
}

// ---- 2. Sub-query / multi-query expansion ----------------------------------
export async function generateSubQueries(userQuery, n) {
  const system = `You generate alternative search queries to improve document retrieval.
Given a user question, produce ${n} diverse rephrasings / sub-questions that capture
different aspects or wordings of the same information need.
Return ONLY a JSON array of strings. No other text.`;
  try {
    const { text } = await chatComplete({
      system,
      user: `User question: "${userQuery}"`,
      temperature: 0.3, // a little diversity so the variants aren't identical
      fast: true,
    });
    const arr = parseJsonLoose(text, []);
    const cleaned = Array.isArray(arr)
      ? arr.filter((s) => typeof s === "string" && s.trim()).map((s) => s.trim())
      : [];
    return cleaned.slice(0, n);
  } catch {
    return [];
  }
}

// ---- 3. HyDE: Hypothetical Document Embeddings -----------------------------
// Write a hypothetical answer passage and embed THAT (richer than a short
// question). Best for public-data RAG; off by default for private documents.
export async function generateHydeDocument(userQuery) {
  const system = `You are helping a retrieval system using the HyDE technique
(Hypothetical Document Embeddings). Write a short, factual hypothetical passage
(3-5 sentences) that would plausibly answer the user's question, as if it were an
excerpt from the source document. This passage is ONLY used to embed and search;
it is never shown to the user. Do not add disclaimers.`;
  try {
    const { text } = await chatComplete({ system, user: userQuery, temperature: 0, fast: false });
    return (text || "").trim();
  } catch {
    return "";
  }
}

// ---- 4. Re-ranking: Reciprocal Rank Fusion ---------------------------------
// The principled version of the class's "count how often a chunk shows up across
// sub-query results" idea: a chunk ranking highly across several result lists
// accumulates the most score and bubbles to the top.
export function reciprocalRankFusion(resultLists, k, rrfK = 60) {
  const scores = new Map(); // pageContent -> { score, doc }
  for (const list of resultLists) {
    list.forEach((doc, rank) => {
      const key = doc.pageContent;
      const entry = scores.get(key) || { score: 0, doc };
      entry.score += 1 / (rrfK + rank + 1);
      scores.set(key, entry);
    });
  }
  return [...scores.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((e) => e.doc);
}

// ---- 5. LLM-as-a-judge -----------------------------------------------------
// Grades every candidate chunk against the ORIGINAL user query (not the rewritten
// one) and keeps only the relevant ones. Batched into a single call for cost.
export async function judgeChunks(userQuery, chunks) {
  if (chunks.length === 0) return [];
  const numbered = chunks
    .map((c, i) => `[${i}] ${c.pageContent.slice(0, 800)}`)
    .join("\n\n");
  const system = `You are a relevance grader (LLM-as-a-judge) for a RAG system.
For each numbered chunk, decide if it is relevant to answering the user's query.
Return ONLY a JSON array of objects: [{"index": <number>, "relevant": <true|false>}].`;
  const user = `User query: "${userQuery}"\n\nChunks:\n${numbered}`;
  try {
    const { text } = await chatComplete({ system, user, temperature: 0, fast: true });
    const verdicts = parseJsonLoose(text, []);
    const relevant = new Set(
      (Array.isArray(verdicts) ? verdicts : [])
        .filter((v) => v && v.relevant === true && Number.isInteger(v.index))
        .map((v) => v.index)
    );
    // If the judge returned nothing parseable, don't throw away all context —
    // fall back to keeping the candidates (fail-open, not fail-closed).
    if (relevant.size === 0) return chunks;
    return chunks.filter((_, i) => relevant.has(i));
  } catch {
    return chunks;
  }
}

// ---- Orchestrator: Corrective RAG -----------------------------------------
export async function correctiveRetrieve(vectorStore, userQuery) {
  const enableRewrite = flag("ENABLE_QUERY_REWRITE", true);
  const enableSub = flag("ENABLE_SUBQUERIES", true);
  const enableHyde = flag("ENABLE_HYDE", false);
  const enableJudge = flag("ENABLE_LLM_JUDGE", true);
  const k = num("RETRIEVAL_K", 4);
  const fetchK = num("RETRIEVAL_FETCH_K", 6);
  const subCount = num("SUBQUERY_COUNT", 3);
  const maxIter = num("CORRECTIVE_MAX_ITERATIONS", 1);

  const trace = { steps: [] };
  let feedback = null;
  let finalChunks = [];

  for (let iter = 0; iter <= maxIter; iter++) {
    // 1. Query rewriting / translation
    const rewritten = enableRewrite ? await rewriteQuery(userQuery, feedback) : userQuery;
    trace.steps.push({ iter, stage: "rewrite", query: rewritten });

    // 2. Build the query set (rewritten + sub-queries + optional HyDE passage)
    const queries = [rewritten];
    if (enableSub) {
      const subs = await generateSubQueries(userQuery, subCount);
      queries.push(...subs);
      trace.steps.push({ iter, stage: "subqueries", queries: subs });
    }
    if (enableHyde) {
      const hyde = await generateHydeDocument(rewritten);
      if (hyde) {
        queries.push(hyde);
        trace.steps.push({ iter, stage: "hyde", preview: hyde.slice(0, 160) });
      }
    }

    // 3. Retrieve for every query in parallel.
    // We tolerate SOME queries failing, but if EVERY search fails (e.g. the
    // embedding API is rate-limited or Qdrant is down) we must NOT pretend we
    // simply found nothing — we re-throw so the route returns the real error
    // (e.g. a 429), not a misleading "no information found" message.
    const settled = await Promise.allSettled(
      queries.map((q) => vectorStore.similaritySearch(q, fetchK))
    );
    const resultLists = settled
      .filter((r) => r.status === "fulfilled")
      .map((r) => r.value);
    if (resultLists.length === 0) {
      throw settled.find((r) => r.status === "rejected")?.reason ||
        new Error("Retrieval failed for all queries.");
    }

    // 4. Re-rank via Reciprocal Rank Fusion
    const candidates = reciprocalRankFusion(resultLists, Math.max(k, fetchK));
    trace.steps.push({ iter, stage: "rerank", candidateCount: candidates.length });

    // 5. LLM-as-a-judge filters to chunks relevant to the ORIGINAL query
    const graded = enableJudge ? await judgeChunks(userQuery, candidates) : candidates;
    trace.steps.push({ iter, stage: "judge", keptCount: graded.length });

    finalChunks = graded.slice(0, k);

    // 6. Corrective decision: enough relevant chunks, or out of retries?
    if (finalChunks.length >= 1 || iter === maxIter) break;
    feedback = "the previously retrieved chunks were judged not relevant to the query";
    trace.steps.push({ iter, stage: "corrective", action: "retry-with-rewrite" });
  }

  return { chunks: finalChunks, trace };
}
