import { NextResponse } from "next/server";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import * as fs from "fs/promises";
import * as path from "path";
import * as os from "os";

export async function POST(req) {
  try {
    const formData = await req.formData();
    const file = formData.get("file");

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // Save temporary file
    const tempDir = os.tmpdir();
    const filePath = path.join(tempDir, file.name);
    await fs.writeFile(filePath, buffer);

    let docs = [];

    if (file.name.endsWith(".pdf")) {
      const loader = new PDFLoader(filePath);
      docs = await loader.load();
    } else if (file.name.endsWith(".txt") || file.name.endsWith(".csv")) {
      const text = await fs.readFile(filePath, "utf-8");
      docs = [{ pageContent: text, metadata: { source: file.name } }];
    } else {
      return NextResponse.json({ error: "Unsupported file type. Please upload a PDF, TXT, or CSV." }, { status: 400 });
    }

    // Chunking: Implementation of a chunking strategy
    // We use RecursiveCharacterTextSplitter to split the document into smaller, semantically meaningful chunks.
    // chunkOverlap ensures context isn't lost between consecutive chunks.
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    
    const splitDocs = await textSplitter.splitDocuments(docs);

    // Add some metadata to the chunks
    const docsWithMetadata = splitDocs.map(doc => {
      return {
        ...doc,
        metadata: {
          ...doc.metadata,
          sourceFileName: file.name
        }
      }
    });

    // Embeddings
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

    // Clear previous context
    const client = new QdrantClient({ url, apiKey });
    try {
      await client.deleteCollection(collectionName);
      console.log(`Cleared previous context by deleting collection: ${collectionName}`);
    } catch (e) {
      console.log("No existing collection to delete or error deleting:", e.message);
    }

    // Store in VectorDB (this automatically recreates the collection)
    await QdrantVectorStore.fromDocuments(docsWithMetadata, embeddings, qdrantConfig);

    // Cleanup temp file
    await fs.unlink(filePath).catch(console.error);

    return NextResponse.json({ 
        success: true, 
        message: "Document successfully ingested, chunked, and stored.",
        chunksCount: splitDocs.length
    });

  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
