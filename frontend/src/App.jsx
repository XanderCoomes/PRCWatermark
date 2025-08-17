import React, { useRef, useState } from 'react'
import { generateText } from './api'

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [numWords, setNumWords] = useState(300)
  const [isWatermarked, setIsWatermarked] = useState(true)
  const [out, setOut] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const abortRef = useRef(null)

  const onSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setOut('')
    setLoading(true)
    if (abortRef.current) abortRef.current.abort()
    abortRef.current = new AbortController()
    try {
      const text = await generateText({
        prompt,
        numWords,
        isWatermarked,
        signal: abortRef.current.signal,
      })
      setOut(text)
    } catch (err) {
      setError(err.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  const onCancel = () => {
    if (abortRef.current) abortRef.current.abort()
    setLoading(false)
  }

  return (
    <div className="container">
      <div className="card">
        <h2 style={{marginTop:0}}>Watermarked Text Generator</h2>
        <div className="badge">API: {import.meta.env.VITE_API_BASE_URL || 'not set'}</div>

        <form onSubmit={onSubmit}>
          <label>Prompt</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Write an essay about a historic murder..."
          />

          <div className="row" style={{marginTop:12}}>
            <div style={{flex:'1 1 160px'}}>
              <label>Word budget (approx)</label>
              <input
                type="number"
                value={numWords}
                min={10}
                max={5000}
                onChange={(e) => setNumWords(e.target.value)}
              />
            </div>
            <div style={{display:'flex', alignItems:'center', gap:10}}>
              <input
                id="wm"
                type="checkbox"
                checked={isWatermarked}
                onChange={(e) => setIsWatermarked(e.target.checked)}
              />
              <label htmlFor="wm" style={{margin:0}}>Watermark</label>
            </div>
          </div>

          <div className="row" style={{marginTop:16}}>
            <button type="submit" disabled={loading || !prompt.trim()}>
              {loading ? 'Generating…' : 'Generate'}
            </button>
            <button type="button" onClick={onCancel} disabled={!loading}>Cancel</button>
          </div>
        </form>

        <hr />
        <label>Output</label>
        <div className="output">
          {error ? `⚠️ ${error}` : (out || (loading ? 'Waiting for response…' : '—'))}
        </div>
      </div>
    </div>
  )
}
