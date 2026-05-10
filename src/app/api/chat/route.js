import { NextResponse } from "next/server";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";


export async function POST(req) {
  try {
    const { message, history } = await req.json();

    if (!message) {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }

    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: "gemini-embedding-2", 
    });

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
        console.error("Vector store error. Did you upload a document first?", e);
        return NextResponse.json({ error: "Could not connect to database. Please ensure you upload a document first." }, { status: 400 });
    }

    const retriever = vectorStore.asRetriever({
      k: 4, // Retrieve top 4 most relevant chunks
    });

    const searchedChunks = await retriever.invoke(message);

    if (!searchedChunks || searchedChunks.length === 0) {
       return NextResponse.json({ 
           response: "I couldn't find any information related to your query in the uploaded document. Please ask something covered in the document.",
           sources: [] 
       });
    }

    // Prepare LLM Model
    const llm = new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash-lite",
      temperature: 0,
      maxRetries: 2,
    });

    const contextText = searchedChunks.map(doc => doc.pageContent).join("\n\n");
    const systemPrompt = `You are a helpful AI Assistant, designed to answer user queries strictly based on the provided context.

Context from uploaded document:
${contextText}

Rules:
- You must ONLY answer based on the provided context.
- If the answer is not contained in the context, you must reply: "I cannot answer this based on the provided document."
- Do not use your general knowledge to answer.
- Provide a clear, professional, and grounded response.`;

    const llmResponse = await llm.invoke([
        ["system", systemPrompt],
        ["user", message]
    ]);

    const responseText = llmResponse.content;

    // Extract unique sources for the frontend to display
    const sources = searchedChunks.map(doc => doc.metadata?.sourceFileName || "Unknown Source");
    const uniqueSources = [...new Set(sources)];

    return NextResponse.json({ 
        response: responseText,
        sources: uniqueSources,
        chunks: searchedChunks.map(c => c.pageContent) // optional: to display citations
    });

  } catch (error) {
    console.error("Chat error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
