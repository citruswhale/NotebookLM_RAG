import { NextResponse } from "next/server";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import { extractText, getDocumentProxy } from "unpdf";
import { getEmbeddings } from "@/lib/providers";
import { classifyError } from "@/lib/errors";

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

    // 3. 100% In-Memory Parsing (No disk-writes or fs-I/O whatsoever).
    // Parsing failures are a FILE problem (corrupt/locked PDF), distinct from a
    // later AI-service problem — so we report them with their own clear message.
    try {
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
    } catch (parseErr) {
      console.error("File parse error:", parseErr);
      return NextResponse.json(
        { error: "We couldn't read that file. It may be corrupted, password-protected, or not a valid PDF/TXT/CSV." },
        { status: 422 }
      );
    }

    // Guard against empty / image-only documents (no extractable text).
    if (!docs.length || !docs[0].pageContent || !docs[0].pageContent.trim()) {
      return NextResponse.json(
        { error: "No readable text was found in this file. If it's a scanned PDF, it may contain only images." },
        { status: 422 }
      );
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

    // 5. Generate Embeddings (Gemini primary / OpenAI backup — see src/lib/providers.js).
    // MUST be the same provider used at query time so the vector spaces match.
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
    // Reaches here mainly for embedding/vector-store failures (e.g. a 429 rate
    // limit while embedding chunks). Classify so the user sees the REAL cause —
    // not a generic "upload failed" — and never leak stack traces to the client.
    console.error("Upload error:", error);
    const { status, message } = classifyError(error);
    return NextResponse.json({ error: message }, { status });
  }
}
