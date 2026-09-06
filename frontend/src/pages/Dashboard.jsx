import { useRef, useState } from 'react'
import Navbar from '../components/Navbar'
import { uploadDocument } from '../services/documentService'
import { askMemory } from '../services/ragService'
import './Dashboard.css'

// ── Upload states ────────────────────────────────────────────────────────────
// idle | loading | success | error

export default function Dashboard() {
  const fileInputRef = useRef(null)
  const [uploadState, setUploadState] = useState('idle') // idle | loading | success | error
  const [uploadResult, setUploadResult] = useState(null)
  const [uploadError, setUploadError] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [isDraggingOver, setIsDraggingOver] = useState(false)

  // ── Ask / RAG states ────────────────────────────────────────────────────────
  // idle | loading | success | error | empty
  const [searchQuery, setSearchQuery] = useState('')
  const [searchState, setSearchState] = useState('idle')
  const [ragAnswer, setRagAnswer] = useState('')
  const [ragSources, setRagSources] = useState([])
  const [searchError, setSearchError] = useState('')
  const [queriesAskedCount, setQueriesAskedCount] = useState(0)

  // ── File selection handlers ────────────────────────────────────────────────

  function handleFileChange(e) {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setUploadState('idle')
      setUploadResult(null)
      setUploadError('')
    }
  }

  function handleDrop(e) {
    e.preventDefault()
    setIsDraggingOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) {
      setSelectedFile(file)
      setUploadState('idle')
      setUploadResult(null)
      setUploadError('')
    }
  }

  function handleDragOver(e) {
    e.preventDefault()
    setIsDraggingOver(true)
  }

  function handleDragLeave() {
    setIsDraggingOver(false)
  }

  // ── Upload handler ─────────────────────────────────────────────────────────

  async function handleUpload() {
    if (!selectedFile) return

    setUploadState('loading')
    setUploadError('')
    setUploadResult(null)

    try {
      const result = await uploadDocument(selectedFile)
      setUploadResult(result)
      setUploadState('success')
    } catch (err) {
      setUploadError(err.message || 'Upload failed. Please try again.')
      setUploadState('error')
    }
  }

  // ── Ask handler (RAG) ──────────────────────────────────────────────────────

  async function handleSearch() {
    if (!searchQuery.trim() || searchState === 'loading') return

    setSearchState('loading')
    setSearchError('')
    setRagAnswer('')
    setRagSources([])

    try {
      const response = await askMemory(searchQuery.trim(), 5)

      setRagAnswer(response?.answer || '')
      setRagSources(response?.sources || [])
      setQueriesAskedCount((prev) => prev + 1)

      if (!response?.answer && (response?.sources || []).length === 0) {
        setSearchState('empty')
      } else {
        setSearchState('success')
      }
    } catch (err) {
      setSearchError(err.message || 'Request failed. Please try again.')
      setSearchState('error')
    }
  }

  // ── Drop zone label ────────────────────────────────────────────────────────

  function dropZoneLabel() {
    if (selectedFile) return selectedFile.name
    return null
  }

  const isLoading = uploadState === 'loading'
  const isSuccess = uploadState === 'success'
  const isError = uploadState === 'error'

  return (
    <div className="dashboard">
      <Navbar />

      {/* ── Hero ── */}
      <section className="hero">
        {/* Ambient glow orbs */}
        <div className="hero__orb hero__orb--1" aria-hidden="true" />
        <div className="hero__orb hero__orb--2" aria-hidden="true" />

        <div className="hero__content">
          <div className="hero__eyebrow">
            <span className="hero__dot" />
            Your personal knowledge layer
          </div>

          <h1 className="hero__title">
            <span className="gradient-text">MemoryOS</span>
          </h1>

          <p className="hero__tagline">
            "Don't remember where you saved it.
            <br />
            Just remember what you need."
          </p>

          <p className="hero__sub">
            Upload your documents, notes, and files — then ask questions in
            plain language. Your memory, always within reach.
          </p>
        </div>
      </section>

      {/* ── Action Cards ── */}
      <section className="cards-section" aria-label="Main actions">
        <div className="cards-grid">
          {/* Upload Memory Card */}
          <article className="card card--upload" id="upload-memory-card">
            <div className="card__icon-wrap card__icon-wrap--upload">
              <svg className="card__icon" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"
                strokeLinejoin="round" aria-hidden="true">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>

            <div className="card__body">
              <h2 className="card__title">Upload Memory</h2>
              <p className="card__desc">
                Add PDFs or text files to your personal knowledge base.
                MemoryOS extracts and indexes them for you.
              </p>
            </div>

            {/* Drop Zone */}
            <div
              className={`card__drop-zone${isDraggingOver ? ' card__drop-zone--active' : ''}${isSuccess ? ' card__drop-zone--success' : ''}${isError ? ' card__drop-zone--error' : ''}`}
              aria-label="Upload drop zone — drag a PDF or TXT file here"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => !isLoading && fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && !isLoading && fileInputRef.current?.click()}
              style={{ cursor: isLoading ? 'wait' : 'pointer' }}
            >
              {isLoading ? (
                <>
                  <span className="card__drop-icon upload-spin">⏳</span>
                  <span className="card__drop-text">Uploading…</span>
                </>
              ) : isSuccess ? (
                <>
                  <span className="card__drop-icon">✅</span>
                  <span className="card__drop-text upload-success-name">{uploadResult.filename}</span>
                </>
              ) : isError ? (
                <>
                  <span className="card__drop-icon">❌</span>
                  <span className="card__drop-text upload-error-msg">{uploadError}</span>
                </>
              ) : selectedFile ? (
                <>
                  <span className="card__drop-icon">📄</span>
                  <span className="card__drop-text upload-selected-name">{dropZoneLabel()}</span>
                  <span className="card__drop-text" style={{ fontSize: '0.78rem', opacity: 0.6 }}>
                    {(selectedFile.size / 1024).toFixed(1)} KB · click to change
                  </span>
                </>
              ) : (
                <>
                  <span className="card__drop-icon">📄</span>
                  <span className="card__drop-text">
                    Drop a PDF or TXT here, or <em>click to browse</em>
                  </span>
                  <span className="card__file-hint">PDF · TXT · max 10 MB</span>
                </>
              )}
            </div>

            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              id="file-input"
              type="file"
              accept=".pdf,.txt"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />

            {/* Upload result details */}
            {isSuccess && uploadResult && (
              <div className="upload-result" id="upload-result">
                <div className="upload-result__row">
                  <span className="upload-result__label">File</span>
                  <span className="upload-result__value">{uploadResult.filename}</span>
                </div>
                <div className="upload-result__row">
                  <span className="upload-result__label">Type</span>
                  <span className="upload-result__value upload-result__badge">{uploadResult.file_type.toUpperCase()}</span>
                </div>
                <div className="upload-result__row">
                  <span className="upload-result__label">Pages</span>
                  <span className="upload-result__value">{uploadResult.pages}</span>
                </div>
                <div className="upload-result__row">
                  <span className="upload-result__label">Text extracted</span>
                  <span className="upload-result__value">{uploadResult.text_length.toLocaleString()} chars</span>
                </div>
              </div>
            )}

            {/* Action buttons */}
            <div className="card__actions">
              <button
                className="btn btn--secondary"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                id="choose-file-btn"
              >
                Choose File
              </button>
              <button
                className="btn btn--primary"
                onClick={handleUpload}
                disabled={!selectedFile || isLoading}
                id="upload-btn"
              >
                {isLoading ? 'Uploading…' : 'Upload'}
              </button>
            </div>
          </article>

          {/* Ask Memory Card */}
          <article className="card card--ask" id="ask-memory-card">
            <div className="card__icon-wrap card__icon-wrap--ask">
              <svg className="card__icon" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"
                strokeLinejoin="round" aria-hidden="true">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </div>

            <div className="card__body">
              <h2 className="card__title">Ask Your Memory</h2>
              <p className="card__desc">
                Ask anything in natural language. MemoryOS searches your
                uploaded knowledge and surfaces exactly what you need.
              </p>
            </div>

            <form
              className="search-form"
              onSubmit={(e) => {
                e.preventDefault()
                handleSearch()
              }}
            >
              <div className="search-input-wrap">
                <span className="search-input-icon" aria-hidden="true">🔍</span>
                <input
                  id="search-input"
                  type="text"
                  className="search-input"
                  placeholder='e.g. "What were the electrician rates?"'
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  disabled={searchState === 'loading'}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      handleSearch()
                    }
                  }}
                />
                {searchQuery && (
                  <button
                    type="button"
                    className="search-clear-btn"
                    onClick={() => {
                      setSearchQuery('')
                      setSearchState('idle')
                      setRagAnswer('')
                      setRagSources([])
                    }}
                    title="Clear search"
                  >
                    ✕
                  </button>
                )}
              </div>

              <button
                type="submit"
                className="btn btn--primary"
                disabled={!searchQuery.trim() || searchState === 'loading'}
                id="ask-btn"
              >
                {searchState === 'loading' ? 'Thinking…' : 'Ask MemoryOS'}
              </button>
            </form>

            {/* State Banners */}
            {searchState === 'loading' && (
              <div className="search-banner search-banner--loading" id="search-loading-banner">
                <span className="upload-spin">⏳</span> Searching your memory and generating answer…
              </div>
            )}

            {searchState === 'error' && (
              <div className="search-banner search-banner--error" id="search-error-banner">
                <span>⚠️</span> {searchError}
              </div>
            )}

            {searchState === 'empty' && (
              <div className="search-banner search-banner--empty" id="search-empty-banner">
                <span>🔍</span> No matching memories found. Try uploading a document first.
              </div>
            )}

            {/* AI Answer */}
            {searchState === 'success' && ragAnswer && (
              <div className="rag-answer" id="rag-answer-section">
                <div className="rag-answer__header">
                  <span className="rag-answer__icon">✨</span>
                  <span className="rag-answer__label">AI Answer</span>
                </div>
                <p className="rag-answer__text">{ragAnswer}</p>
              </div>
            )}

            {/* Sources / Evidence */}
            {searchState === 'success' && ragSources.length > 0 && (
              <div className="search-results" id="search-results-section">
                <div className="search-results-header">
                  <span className="search-results-title">Sources / Evidence</span>
                  <span className="search-results-count">
                    {ragSources.length} {ragSources.length === 1 ? 'source' : 'sources'}
                  </span>
                </div>

                <div className="search-results-list">
                  {ragSources.map((item, idx) => {
                    const sourceName = item.metadata?.source || item.metadata?.filename || 'Uploaded Document'
                    const chunkIdx = item.metadata?.chunk_index !== undefined ? ` (Chunk #${item.metadata.chunk_index + 1})` : ''
                    const distFormatted = item.distance !== undefined && item.distance !== null
                      ? `Distance: ${Number(item.distance).toFixed(3)}`
                      : null

                    return (
                      <div className="search-result-card" key={item.id || idx}>
                        <div className="search-result-meta">
                          <span className="search-result-source">
                            📄 {sourceName}{chunkIdx}
                          </span>
                          {distFormatted && (
                            <span className="search-result-distance">{distFormatted}</span>
                          )}
                        </div>
                        <p className="search-result-text">{item.text}</p>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </article>
        </div>
      </section>

      {/* ── Stats Row ── */}
      <section className="stats-section" aria-label="Status overview">
        <div className="stats-grid">
          {[
            { label: 'Memories Stored',  value: uploadResult ? 'Indexed' : '0',   icon: '🧠' },
            { label: 'Files Indexed',    value: uploadResult ? '1' : '0',   icon: '📂' },
            { label: 'Queries Asked',    value: queriesAskedCount.toString(),   icon: '💬' },
            { label: 'Last Activity',    value: queriesAskedCount > 0 || uploadResult ? 'Just now' : '—',   icon: '⏱' },
          ].map((s) => (
            <div className="stat-card" key={s.label} id={`stat-${s.label.toLowerCase().replace(/\s+/g, '-')}`}>
              <span className="stat-card__icon">{s.icon}</span>
              <span className="stat-card__value">{s.value}</span>
              <span className="stat-card__label">{s.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="footer">
        <span>MemoryOS</span>
        <span className="footer__sep">·</span>
        <span>Upload → Extract → Remember</span>
      </footer>
    </div>
  )
}
