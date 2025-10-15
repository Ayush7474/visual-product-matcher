import React, { useState, useRef } from "react";
import "./App.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8000";

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewSrc, setPreviewSrc] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [minScore, setMinScore] = useState(0.5); // filter threshold (0..1)
  const inputFileRef = useRef(null);

  function onFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setSelectedFile(f);
    setImageUrl("");
    const reader = new FileReader();
    reader.onload = (ev) => setPreviewSrc(ev.target.result);
    reader.readAsDataURL(f);
  }

  function onUrlChange(e) {
    setImageUrl(e.target.value);
    setSelectedFile(null);
    setPreviewSrc(e.target.value);
  }

  async function handleSearch() {
    setError("");
    setResults([]);
    if (!selectedFile && !imageUrl) {
      setError("Please upload a file or paste an image URL.");
      return;
    }

    setLoading(true);
    try {
      const form = new FormData();
      if (selectedFile) form.append("file", selectedFile);
      if (imageUrl) form.append("image_url", imageUrl);
      form.append("top_k", 20);

      const resp = await fetch(`${BACKEND_URL}/search`, {
        method: "POST",
        body: form,
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "Search failed");
      }

      const json = await resp.json();
      const filtered = (json.results || []).filter((r) => (r.score ?? 0) >= minScore);
      setResults(filtered);
      if ((json.results || []).length === 0) setError("No matches found for this image.");
    } catch (err) {
      console.error(err);
      setError("Search failed. Check backend is running and the image is valid.");
    } finally {
      setLoading(false);
    }
  }

  function clearAll() {
    setSelectedFile(null);
    setImageUrl("");
    setPreviewSrc("");
    setResults([]);
    setError("");
    if (inputFileRef.current) inputFileRef.current.value = "";
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Visual Product Matcher</h1>
        <p className="subtitle">Upload an image or paste URL → find visually similar products</p>
      </header>

      <main className="main">
        <section className="controls">
          <div className="uploader">
            <label className="file-label">
              <input ref={inputFileRef} type="file" accept="image/*" onChange={onFileChange} />
              <span>Select file</span>
            </label>

            <div className="or">or</div>

            <input
              className="url-input"
              type="text"
              placeholder="Paste image URL..."
              value={imageUrl}
              onChange={onUrlChange}
            />

            <div className="actions">
              <button className="btn" onClick={handleSearch} disabled={loading}>
                {loading ? "Searching..." : "Search"}
              </button>
              <button className="btn btn-ghost" onClick={clearAll}>
                Reset
              </button>
            </div>

            <div className="preview-wrap">
              {previewSrc ? (
                <img src={previewSrc} alt="preview" className="preview" />
              ) : (
                <div className="preview placeholder">Preview will appear here</div>
              )}
            </div>

            <div className="slider-row">
              <label>Minimum similarity: {(minScore * 100).toFixed(0)}%</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={minScore}
                onChange={(e) => setMinScore(Number(e.target.value))}
              />
            </div>

            {error && <div className="error">{error}</div>}
          </div>
        </section>

        <section className="results">
          {loading && <div className="info">Processing — this can take a few seconds...</div>}
          {!loading && results.length === 0 && <div className="info">No results to show.</div>}

          <div className="results-grid">
            {results.map((r) => (
              <article className="card" key={r.id}>
                <img className="card-image" src={r.image_url} alt={r.name} />
                <div className="card-body">
                  <h3 className="card-title">{r.name}</h3>
                  <div className="meta">
                    <span>{r.category}</span>
                    <span>₹{r.price}</span>
                  </div>
                  <div className="score">Similarity: {(r.score * 100).toFixed(2)}%</div>
                </div>
              </article>
            ))}
          </div>
        </section>
      </main>

      <footer className="footer">
        <small>Backend: {BACKEND_URL} · Items shown: {results.length}</small>
      </footer>
    </div>
  );
}
