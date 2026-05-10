"use client";

import { useState, useRef, useEffect } from "react";
import styles from "./page.module.css";
import { UploadCloud, FileText, Send, User, Bot, Loader2, Moon, Sun, RefreshCw } from "lucide-react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isReady, setIsReady] = useState(false);
  
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [theme, setTheme] = useState("dark");
  const messagesEndRef = useRef(null);

  // Load state from localStorage on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem("rag_theme");
    if (savedTheme) setTheme(savedTheme);

    const savedMessages = localStorage.getItem("rag_messages");
    if (savedMessages) setMessages(JSON.parse(savedMessages));

    const savedFile = localStorage.getItem("rag_file");
    if (savedFile) setFile({ name: savedFile });

    const savedIsReady = localStorage.getItem("rag_isReady");
    if (savedIsReady) setIsReady(savedIsReady === "true");
  }, []);

  // Save state to localStorage when it changes
  useEffect(() => {
    localStorage.setItem("rag_theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem("rag_messages", JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    if (file && file.name) {
      localStorage.setItem("rag_file", file.name);
    } else {
      localStorage.removeItem("rag_file");
    }
  }, [file]);

  useEffect(() => {
    localStorage.setItem("rag_isReady", isReady);
  }, [isReady]);

  const toggleTheme = () => {
    setTheme(prev => prev === "dark" ? "light" : "dark");
  };

  const handleReset = () => {
    setFile(null);
    setIsReady(false);
    setMessages([]);
    setInput("");
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        setIsReady(true);
        setMessages([
          {
            role: "bot",
            content: `I've successfully read "${file.name}" and broke it into ${data.chunksCount} chunks. What would you like to know about it?`,
          },
        ]);
      } else {
        alert("Upload failed: " + data.error);
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert("An error occurred during upload.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !isReady) return;

    const userMsg = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setIsTyping(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg, history: messages }),
      });

      const data = await res.json();
      if (res.ok) {
        setMessages((prev) => [
          ...prev,
          { role: "bot", content: data.response, sources: data.sources },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "bot", content: "Sorry, I encountered an error: " + data.error },
        ]);
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        { role: "bot", content: "Network error occurred." },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.title}>NotebookLM RAG</div>
        <button onClick={toggleTheme} style={{background: 'var(--panel-bg)', border: '1px solid var(--panel-border)', padding: '0.5rem', borderRadius: '8px', color: 'var(--foreground)'}}>
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </header>

      <main className={styles.mainGrid}>
        {/* Left Side: Upload & Doc Info */}
        <section className={styles.uploadPanel}>
          <h3 className={styles.panelTitle}>Source Material</h3>
          
          {!isReady ? (
            <div className={styles.uploadZone}>
              <UploadCloud size={32} className={styles.uploadIcon} />
              <p style={{fontSize: '0.875rem', color: 'var(--muted)'}}>Drag & drop or select a PDF, TXT, or CSV</p>
              <input 
                type="file" 
                accept=".pdf,.txt,.csv" 
                onChange={handleFileChange} 
                style={{ display: "none" }} 
                id="file-upload" 
              />
              <label htmlFor="file-upload" className={styles.uploadButton} style={{cursor: 'pointer', textAlign: 'center'}}>
                Select File
              </label>

              {file && (
                <div className={styles.fileList} style={{width: '100%'}}>
                  <div className={styles.fileItem}>
                    <FileText size={16} className={styles.fileItemIcon} />
                    <div>
                      <div className={styles.fileName}>{file.name}</div>
                    </div>
                  </div>
                  <button 
                    className={styles.uploadButton} 
                    onClick={handleUpload}
                    disabled={isUploading}
                    style={{background: 'var(--primary)', color: 'var(--primary-foreground)'}}
                  >
                    {isUploading ? <span style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem'}}><Loader2 size={16} className="animate-spin" /> Ingesting...</span> : "Upload & Process"}
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className={styles.fileList}>
               <div className={styles.fileItem}>
                 <FileText size={18} className={styles.fileItemIcon} color="var(--primary)" />
                 <div style={{flex: 1}}>
                   <div className={styles.fileName}>{file?.name}</div>
                   <div className={styles.fileMeta}>Indexed in Vector DB</div>
                 </div>
               </div>
               <button 
                 onClick={handleReset} 
                 className={styles.uploadButton}
                 style={{background: 'var(--panel-bg)', color: 'var(--foreground)', border: '1px solid var(--panel-border)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem'}}
               >
                 <RefreshCw size={14} /> New Upload
               </button>
            </div>
          )}
        </section>

        {/* Right Side: Chat Interface */}
        <section className={styles.chatPanel}>
          <div className={styles.chatHistory}>
            {messages.length === 0 ? (
              <div className={styles.emptyState}>
                <Bot size={32} />
                <p>Upload a document to start a conversation.</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} className={`${styles.message} ${styles[msg.role]}`}>
                  <div className={styles.avatar}>
                    {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  <div className={styles.messageContent}>
                    <p>{msg.content}</p>
                    {msg.sources && msg.sources.length > 0 && (
                      <div className={styles.sources}>
                        <span className={styles.sourceLabel}>Sources:</span>
                        {msg.sources.map((src, i) => (
                          <span key={i} className={styles.sourceTag}>{src}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            
            {isTyping && (
              <div className={`${styles.message} ${styles.bot}`}>
                <div className={styles.avatar}>
                  <Bot size={16} />
                </div>
                <div className={styles.messageContent}>
                  <Loader2 size={18} className="animate-spin" style={{marginTop: '4px', color: 'var(--muted)'}} />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className={styles.inputAreaWrapper}>
            <div className={styles.inputArea}>
              <input 
                type="text" 
                className={styles.input} 
                placeholder={isReady ? "Ask a question..." : "Upload a document first..."}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                disabled={!isReady || isTyping}
              />
              <button 
                className={styles.sendButton} 
                onClick={handleSend}
                disabled={!isReady || isTyping || !input.trim()}
              >
                <Send size={16} />
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
