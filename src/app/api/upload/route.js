import { NextResponse } from "next/server";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import { extractText, getDocumentProxy } from "unpdf";

export const maxDuration = 60; // Prevent Vercel 504 timeouts (especially for larger files)

export async function POST(req) {
  try {
    const formData = await req.formData();
    const file = formData.get("file");

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    // 1. Safe filename fallback (Vercel sometimes parses as a Blob, leaving file.name undefined)
    let fileName = file.name;
    if (!fileName) {
      const mimeType = file.type || "";
      if (mimeType.includes("pdf")) {
        fileName = "uploaded_document.pdf";
      } else if (mimeType.includes("csv")) {
        fileName = "uploaded_document.csv";
      } else {
        fileName = "uploaded_document.txt";
      }
    }

    // 2. Safe arrayBuffer extraction (Vercel sometimes lacks the native method on polyfilled files)
    const arrayBuffer = typeof file.arrayBuffer === "function" 
      ? await file.arrayBuffer() 
      : await new Response(file).arrayBuffer();
        
    const buffer = Buffer.from(arrayBuffer);

    let docs = [];

    // 3. 100% In-Memory Parsing (No disk-writes or fs-I/O whatsoever)
    if (fileName.endsWith(".pdf")) {
      // In-memory pure-JS PDF text extraction using unpdf (Vercel-safe)
      const pdf = await getDocumentProxy(new Uint8Array(buffer));
      const { text } = await extractText(pdf, { mergePages: true });
      docs = [{ pageContent: text, metadata: { source: fileName } }];
    } else if (fileName.endsWith(".txt") || fileName.endsWith(".csv")) {
      // In-memory text/csv decoding
      const text = buffer.toString("utf-8");
      docs = [{ pageContent: text, metadata: { source: fileName } }];
    } else {
      return NextResponse.json({ error: "Unsupported file type. Please upload a PDF, TXT, or CSV." }, { status: 400 });
    }

    // 4. In-Memory Semantic Chunking
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    
    const splitDocs = await textSplitter.splitDocuments(docs);

    // Tag chunks with source filename
    const docsWithMetadata = splitDocs.map(doc => {
      return {
        ...doc,
        metadata: {
          ...doc.metadata,
          sourceFileName: fileName
        }
      };
    });

    // 5. Generate Embeddings using Gemini Gen 2 model
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

    // 6. Clear Previous Context (Explicitly drop & recreate the collection)
    const client = new QdrantClient({ url, apiKey });
    try {
      await client.deleteCollection(collectionName);
      console.log(`Cleared previous context by deleting collection: ${collectionName}`);
    } catch (e) {
      console.log("No existing collection to delete or error deleting:", e.message);
    }

    // 7. Store in VectorDB
    await QdrantVectorStore.fromDocuments(docsWithMetadata, embeddings, qdrantConfig);

    return NextResponse.json({ 
        success: true, 
        message: "Document successfully ingested, chunked, and stored.",
        chunksCount: splitDocs.length
    });

  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json({ error: error.message, stack: error.stack }, { status: 500 });
  }
}
