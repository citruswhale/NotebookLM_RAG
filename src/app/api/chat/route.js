import { NextResponse } from "next/server";
import { QdrantVectorStore } from "@langchain/qdrant";
import { getEmbeddings, chatComplete } from "@/lib/providers";
import { correctiveRetrieve } from "@/lib/retrieval";
import { classifyError } from "@/lib/errors";

// A failed connection to an existing collection usually means "no document
// uploaded yet" (collection missing) rather than a service outage.
function looksLikeMissingCollection(error) {
  const s = `${error?.message || ""} ${error?.status || ""} ${error?.response?.status || ""}`.toLowerCase();
  return (
    s.includes("not found") ||
    s.includes("404") ||
    s.includes("doesn't exist") ||
    s.includes("does not exist") ||
    s.includes("no collection") ||
    s.includes("collection") && s.includes("exist")
  );
}

export async function POST(req) {
  try {
    const { message, history } = await req.json();

    if (!message) {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }

    // Same embedding provider as ingestion (see src/lib/providers.js).
    const embeddings = getEmbeddings();

    const url = process.env.QDRANT_URL || "http://localhost:6333";
    const apiKey = process.env.QDRANT_API_KEY;
    const collectionName = process.env.QDRANT_COLLECTION_NAME || "notebooklm-rag";

    const qdrantConfig = {
      url,
      collectionName,
    };

    if (apiKey) {
        qdrantConfig.apiKey = apiKey;
    }

    // Initialize Vector Store for Retrieval
    let vectorStore;
    try {
        vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, qdrantConfig);
    } catch (e) {
        console.error("Vector store connection error:", e);
        // Only show "upload a document first" when the collection is genuinely
        // missing. A rate-limit / auth / network failure must surface as itself.
        if (looksLikeMissingCollection(e)) {
            return NextResponse.json(
              { error: "No document found. Please upload a document before asking questions." },
              { status: 400 }
            );
        }
        const { status, message } = classifyError(e);
        return NextResponse.json({ error: message }, { status });
    }

    // ADVANCED / CORRECTIVE RAG retrieval pipeline (query rewriting -> sub-queries
    // -> RRF re-ranking -> LLM-as-judge -> corrective retry). Replaces the BASE
    // single-shot top-k vector search.
    const { chunks: searchedChunks, trace } = await correctiveRetrieve(vectorStore, message);

    if (!searchedChunks || searchedChunks.length === 0) {
       return NextResponse.json({
           response: "I couldn't find any information related to your query in the uploaded document. Please ask something covered in the document.",
           sources: [],
           trace,
       });
    }

    const contextText = searchedChunks.map(doc => doc.pageContent).join("\n\n");
    const systemPrompt = `You are a helpful AI Assistant, designed to answer user queries strictly based on the provided context.

Context from uploaded document:
${contextText}

Rules:
- You must ONLY answer based on the provided context.
- If the answer is not contained in the context, you must reply: "I cannot answer this based on the provided document."
- Do not use your general knowledge to answer.
- Provide a clear, professional, and grounded response.`;

    // Final generation. chatComplete uses Gemini (primary) and transparently
    // falls back to OpenAI (backup) on failure. temperature 0 = grounded/faithful.
    const { text: responseText, provider } = await chatComplete({
      system: systemPrompt,
      user: message,
      temperature: 0,
      fast: false,
    });

    // Extract unique sources for the frontend to display
    const sources = searchedChunks.map(doc => doc.metadata?.sourceFileName || "Unknown Source");
    const uniqueSources = [...new Set(sources)];

    return NextResponse.json({
        response: responseText,
        sources: uniqueSources,
        chunks: searchedChunks.map(c => c.pageContent), // optional: to display citations
        provider, // which LLM provider answered (google primary / openai backup)
        trace,    // retrieval trace (rewrite/subqueries/rerank/judge/corrective)
    });

  } catch (error) {
    console.error("Chat error:", error);
    const { status, message } = classifyError(error);
    return NextResponse.json({ error: message }, { status });
  }
}
