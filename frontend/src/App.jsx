import React, { useState, useRef, useEffect } from 'react'

export default function App() {
  const [text, setText] = useState('')
  const [response, setResponse] = useState('')

  // numeric input: keep display as string (avoid 02 etc.), parse number for logic
  const [targetCountInput, setTargetCountInput] = useState('') // starts BLANK
  const [targetCount, setTargetCount] = useState(0)            // logic default

  // Detect panel
  const [detectText, setDetectText] = useState('')
  const [detectProb, setDetectProb] = useState(null)  // null until run
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectStatusText, setDetectStatusText] = useState('') // animated "Detecting..." text
  const detectControllerRef = useRef(null)
  const dotsIntervalRef = useRef(null)

  // Card + field styles
  const cardStyle = {
    borderRadius: 12,
    backgroundColor: '#1e293b',
    color: '#e2e8f0',
    boxShadow: '0 6px 18px rgba(0,0,0,0.35)',
    padding: 24,
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    minHeight: 0,
  }

  const boxStyle = {
    borderRadius: 10,
    border: '1px solid #475569',
    backgroundColor: '#334155',
    color: '#e2e8f0',
    padding: '14px 16px',
    boxSizing: 'border-box',
    fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
    fontSize: 17,
    lineHeight: 1.5,
  }

  // ----- Run the streaming check ONLY when user presses Enter in the prompt -----
  async function runCheck() {
    const story = text.trim()
    if (!story) {
      setResponse('')
      return
    }

    try {
      setResponse('') // clear previous result
      const res = await fetch('/api/check_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ story, word_count: targetCount }),
      })
      if (!res.ok) {
        const msg = await res.text().catch(() => '')
        throw new Error(`HTTP ${res.status} ${msg}`)
      }

      if (!res.body) {
        const full = await res.text()
        setResponse(full)
        return
      }

      // stream word-by-word
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let acc = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        acc += decoder.decode(value, { stream: true })
        setResponse(acc)
      }
    } catch (err) {
      console.error(err)
      setResponse('Error contacting server')
    }
  }

  // Intercept Enter in the left textarea: Enter -> runCheck, Shift+Enter -> newline
  function handleTextareaKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      runCheck()
    }
  }

  // --- numeric input normalization (prevents "02" etc.) ---
  function handleTargetChange(e) {
    let v = e.target.value

    if (v === '') {
      setTargetCountInput('')
      setTargetCount(0) // blank means 0 for logic
      return
    }

    // digits only
    v = v.replace(/[^\d]/g, '')

    // strip leading zeros (but allow single "0")
    if (v.length > 1) v = v.replace(/^0+(?=\d)/, '')

    setTargetCountInput(v)
    setTargetCount(parseInt(v || '0', 10))
  }

  function handleTargetBlur() {
    if (targetCountInput === '') {
      setTargetCount(0)
      // leave targetCountInput as '' so the box stays blank
    }
  }

  function handleWordCountKeyDown(e) {
    if (e.key === 'Enter') {
      e.preventDefault()
      // Make sure we parse the latest value before running
      let v = e.currentTarget.value.replace(/[^\d]/g, '')
      if (v.length > 1) v = v.replace(/^0+(?=\d)/, '')
      setTargetCountInput(v)
      setTargetCount(parseInt(v || '0', 10))
      runCheck()
    }
  }
  // --------------------------------------------------------

  // ----- Detect: only run on Enter in the Detect textarea -----
  async function runDetect() {
    const payload = detectText.trim()
    if (!payload || isDetecting) return

    // Abort any in-flight request (just in case)
    if (detectControllerRef.current) {
      detectControllerRef.current.abort()
    }
    const controller = new AbortController()
    detectControllerRef.current = controller

    // Start animated "Detecting..." status
    setIsDetecting(true)
    setDetectProb(null)
    startDetectingDots()

    try {
      const res = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: payload }),
        signal: controller.signal,
      })
      if (!res.ok) {
        const msg = await res.text().catch(() => '')
        throw new Error(`HTTP ${res.status} ${msg}`)
      }
      const data = await res.json()
      if (typeof data.prob === 'number') setDetectProb(data.prob)
      else setDetectProb(null)
    } catch (err) {
      if (err?.name !== 'AbortError') {
        console.error(err)
        setDetectProb(null)
      }
    } finally {
      stopDetectingDots()
      if (detectControllerRef.current === controller) {
        detectControllerRef.current = null
      }
      setIsDetecting(false)
    }
  }

  // Animated "Detecting..." helper
  function startDetectingDots() {
    setDetectStatusText('Detecting')
    if (dotsIntervalRef.current) clearInterval(dotsIntervalRef.current)
    const frames = ['Detecting', 'Detecting.', 'Detecting..', 'Detecting...']
    let i = 0
    dotsIntervalRef.current = setInterval(() => {
      i = (i + 1) % frames.length
      setDetectStatusText(frames[i])
    }, 300)
  }

  function stopDetectingDots() {
    if (dotsIntervalRef.current) {
      clearInterval(dotsIntervalRef.current)
      dotsIntervalRef.current = null
    }
    setDetectStatusText('')
  }

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (dotsIntervalRef.current) clearInterval(dotsIntervalRef.current)
      if (detectControllerRef.current) detectControllerRef.current.abort()
    }
  }, [])

  // Enter to run detection, Shift+Enter for newline
  function handleDetectKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      runDetect()
    }
  }
  // --------------------------------------------------------

  return (
    <div style={{ backgroundColor: '#0f172a', minHeight: '100vh' }}>
      <div
        style={{
          maxWidth: 'min(1600px, 96vw)',
          margin: '0 auto',
          padding: 28,
          height: '100vh',
          boxSizing: 'border-box',
          fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
          fontSize: 17, // base size for everything
          lineHeight: 1.6,
        }}
      >
        {/* Full-height grid: left generator (2fr) | right detect (1.1fr) */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '2fr 1.1fr',
            gap: 28,
            alignItems: 'stretch',
            height: 'calc(100vh - 56px)', // account for padding
          }}
        >
          {/* LEFT: Generator panel */}
          <div style={{ minHeight: 0 }}>
            <div style={cardStyle}>
              {/* Response area fills available vertical space */}
              <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
                <h3 style={{ color: '#38bdf8', marginTop: 0, fontSize: 22, lineHeight: 1.3 }}>
                  Response
                </h3>
                <div
                  style={{
                    ...boxStyle,
                    minHeight: 0,
                    height: '100%',
                    whiteSpace: 'pre-wrap',
                    fontSize: 17,
                  }}
                >
                  {response}
                </div>
              </div>

              {/* Prompt area at the bottom */}
              <div style={{ marginTop: 20, borderTop: '1px solid #475569', paddingTop: 18 }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-end',
                  }}
                >
                  <h2 style={{ color: '#38bdf8', margin: 0, fontSize: 26, lineHeight: 1.2 }}>
                    What can I help with? 
                  </h2>
                  <span
                    style={{
                      fontSize: 24,
                      color: '#38bdf8',
                      fontWeight: 600,
                      lineHeight: 1.2,
                    }}
                  >
                    Word Count
                  </span>
                </div>

                <div style={{ display: 'flex', gap: 14, marginTop: 10 }}>
                  <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    onKeyDown={handleTextareaKeyDown}
                    placeholder="Ask Anything"
                    style={{
                      ...boxStyle,
                      flex: 1,
                      minHeight: 160,
                      resize: 'vertical',
                      outline: 'none',
                      fontSize: 18,
                    }}
                  />
                  <input
                    type="text"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    value={targetCountInput}
                    onChange={handleTargetChange}
                    onBlur={handleTargetBlur}
                    onKeyDown={handleWordCountKeyDown}   // ← add this
                    style={{
                      ...boxStyle,
                      width: 172,
                      textAlign: 'center',
                      fontSize: 20,
                      paddingTop: 12,
                      paddingBottom: 12,
                    }}
                    placeholder="Word Count" // starts blank
                  />
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT: Detect panel */}
          <div style={{ minHeight: 0 }}>
            <div style={cardStyle}>
              <h2 style={{ color: '#38bdf8', marginTop: 0, fontSize: 26, lineHeight: 1.2 }}>
                Detect
              </h2>

              <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                <textarea
                  value={detectText}
                  onChange={(e) => setDetectText(e.target.value)}
                  onKeyDown={handleDetectKeyDown}  // Enter to run detection (Shift+Enter = newline)
                  placeholder="Enter text to check for AI"
                  style={{
                    ...boxStyle,
                    flex: 1,
                    minHeight: 0,
                    height: '100%',
                    resize: 'vertical',
                    outline: 'none',
                    fontSize: 18,
                  }}
                />
              </div>

              {/* Probability Text is AI (title + status). Box shows ONLY percentage */}
              <div style={{ marginTop: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', margin: '0 0 8px 0' }}>
                  <h3 style={{ color: '#38bdf8', margin: 0, fontSize: 20, lineHeight: 1.3 }}>
                    Probability Text is AI
                  </h3>
                  {isDetecting && (
                    <span style={{ color: '#94a3b8', fontSize: 14 }}>{`🔄 ${detectStatusText}`}</span>
                  )}
                </div>
                <div
                  style={{
                    ...boxStyle,
                    whiteSpace: 'pre-wrap',
                    textAlign: 'center',
                    fontSize: 24,
                    fontWeight: 700,
                  }}
                >
                  {/* Only show percentage; blank otherwise */}
                  {detectProb == null ? '' : `${Math.round(detectProb * 100)}%`}
                </div>
              </div>
            </div>
          </div>
          {/* END right panel */}
        </div>
      </div>
    </div>
  )
}
