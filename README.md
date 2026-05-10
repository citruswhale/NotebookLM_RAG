# Google NotebookLM RAG Clone

A full-stack web application that allows users to upload documents (PDF, TXT, CSV) and interactively chat with them. It implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline using modern AI frameworks.

## Features

- **Document Ingestion:** Drag-and-drop support for PDF, TXT, and CSV formats.
- **Intelligent Chunking:** Uses `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200) to ensure context remains unbroken across boundaries.
- **Vector Database:** Integrates natively with Qdrant for storing and retrieving high-dimensional embeddings.
- **Advanced Embeddings:** Utilizes Google's Generative AI `gemini-embedding-2` for creating deep semantic embeddings.
- **Conversational UI:** A highly polished, modern interface with a Dark/Light mode toggle, designed for a premium user experience.
- **Grounded Generation:** Uses `gemini-2.5-flash-lite` to answer questions strictly based on the retrieved context, eliminating hallucinations.

## Tech Stack

- **Frontend:** Next.js (App Router), React, CSS Modules
- **Backend:** Next.js API Routes (Serverless)
- **AI/RAG:** LangChain, Google Generative AI
- **Vector Store:** Qdrant
- **Icons:** Lucide React

## Setup Instructions

1. Clone the repository.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env.local` file in the root directory with the following variables:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   QDRANT_URL=your_qdrant_instance_url
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_COLLECTION_NAME=notebooklm-rag
   ```
4. Run the development server:
   ```bash
   npm run dev
   ```
5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## How it Works

1. **Upload**: When a document is uploaded, it is parsed into raw text.
2. **Clear & Index**: The application drops any previous Qdrant collection to ensure completely fresh context, then recreates it.
3. **Chunk**: The text is split into overlapping chunks to preserve semantic boundaries.
4. **Embed**: Chunks are converted to embeddings and pushed to Qdrant.
5. **Retrieve**: User queries are embedded and matched against the vector database to retrieve the top `k` most relevant chunks.
6. **Generate**: The Gemini model synthesizes an answer using *only* the retrieved context.
