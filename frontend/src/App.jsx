import React, { useState, useRef, useEffect } from 'react'

export default function App() {
  const [text, setText] = useState('')
  const [response, setResponse] = useState('')

  // Word Count
  const [targetCountInput, setTargetCountInput] = useState('')
  const [targetCount, setTargetCount] = useState(0)

  // Detect panel
  const [detectText, setDetectText] = useState('')
  const [detectProb, setDetectProb] = useState(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectStatusText, setDetectStatusText] = useState('')
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
    minHeight: 0,  // allow inner flex children to shrink & scroll
    minWidth: 0,   // prevent grid overflow
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
    whiteSpace: 'pre-wrap',
    overflowWrap: 'anywhere',
    wordBreak: 'break-word',
    maxWidth: '100%',
  }

  // ---- Helpers ----
  function canGenerate(promptVal = text, wcInputVal = targetCountInput) {
    const hasPrompt = (promptVal || '').trim().length > 0
    const wcNum = parseInt(wcInputVal, 10)
    const hasWC = wcInputVal !== '' && Number.isFinite(wcNum) && wcNum > 0
    return hasPrompt && hasWC
  }

  // ----- Run the streaming check -----
  async function runCheck(wordCountOverride) {
    if (!canGenerate()) return

    const story = text.trim()
    const wc = typeof wordCountOverride === 'number' ? wordCountOverride : targetCount

    try {
      setResponse('')
      const res = await fetch('/api/check_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ story, word_count: wc }),
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

  // Enter in the left textarea: runCheck; Shift+Enter: newline
  function handleTextareaKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (canGenerate(e.currentTarget.value, targetCountInput)) {
        runCheck()
      }
    }
  }

  // --- Word Count normalization ---
  function handleTargetChange(e) {
    let v = e.target.value
    if (v === '') {
      setTargetCountInput('')
      setTargetCount(0)
      return
    }
    v = v.replace(/[^\d]/g, '')
    if (v.length > 1) v = v.replace(/^0+(?=\d)/, '')
    setTargetCountInput(v)
    setTargetCount(parseInt(v || '0', 10))
  }

  function handleTargetBlur() {
    if (targetCountInput === '') {
      setTargetCount(0)
    }
  }

  function handleWordCountKeyDown(e) {
    if (e.key === 'Enter') {
      e.preventDefault()
      let v = e.currentTarget.value.replace(/[^\d]/g, '')
      if (v.length > 1) v = v.replace(/^0+(?=\d)/, '')
      const parsed = parseInt(v || '0', 10)
      setTargetCountInput(v)
      setTargetCount(parsed)
      if (canGenerate(text, v)) {
        runCheck(parsed)
      }
    }
  }

  // ----- Detect: only run on Enter in the Detect textarea -----
  async function runDetect() {
    const payload = detectText.trim()
    if (!payload || isDetecting) return

    if (detectControllerRef.current) {
      detectControllerRef.current.abort()
    }
    const controller = new AbortController()
    detectControllerRef.current = controller

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
  useEffect(() => {
    return () => {
      if (dotsIntervalRef.current) clearInterval(dotsIntervalRef.current)
      if (detectControllerRef.current) detectControllerRef.current.abort()
    }
  }, [])

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
          fontSize: 17,
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
            height: 'calc(100vh - 56px)',
            minHeight: 0, // critical for children to be able to scroll
          }}
        >
          {/* LEFT: Generator panel */}
          <div style={{ minHeight: 0, minWidth: 0 }}>
            <div style={cardStyle}>
              {/* Response area */}
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  flex: 1,
                  minHeight: 0, // allow inner flex child to take remaining space
                }}
              >
                <h3 style={{ color: '#38bdf8', marginTop: 0, fontSize: 22, lineHeight: 1.3 }}>
                  Response
                </h3>
                {/* Make THIS grey box the scroll container */}
                <div
                  style={{
                    ...boxStyle,
                    flex: 1,
                    minHeight: 0,
                    overflowY: 'auto',   // vertical scroll here
                  }}
                >
                  {response}
                </div>
              </div>

              {/* Prompt + Word Count */}
              <div style={{ marginTop: 20, borderTop: '1px solid #475569', paddingTop: 18 }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-end',
                    gap: 12,
                    flexWrap: 'wrap',
                  }}
                >
                  <h2 style={{ color: '#38bdf8', margin: 0, fontSize: 26, lineHeight: 1.2 }}>
                    What can I help you with?
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

                <div style={{ display: 'flex', gap: 14, marginTop: 10, minHeight: 0 }}>
                  <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    onKeyDown={handleTextareaKeyDown}
                    placeholder="Ask Anything"
                    style={{
                      ...boxStyle,
                      flex: 1,
                      minHeight: 160,
                      maxHeight: 280,
                      overflowY: 'auto',   // scroll inside the textarea
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
                    onKeyDown={handleWordCountKeyDown}
                    style={{
                      ...boxStyle,
                      width: 172,
                      textAlign: 'center',
                      fontSize: 20,
                      paddingTop: 12,
                      paddingBottom: 12,
                    }}
                    placeholder="Word Count"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT: Detect panel */}
          <div style={{ minHeight: 0, minWidth: 0 }}>
            <div style={cardStyle}>
              <h2 style={{ color: '#38bdf8', marginTop: 0, fontSize: 26, lineHeight: 1.2 }}>
                Detect
              </h2>

              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  flex: 1,
                  minHeight: 0,
                }}
              >
                <textarea
                  value={detectText}
                  onChange={(e) => setDetectText(e.target.value)}
                  onKeyDown={handleDetectKeyDown}
                  placeholder="Enter text to check for AI (Enter to detect)"
                  style={{
                    ...boxStyle,
                    flex: 1,
                    minHeight: 140,
                    maxHeight: '60vh',
                    overflowY: 'auto',
                    resize: 'vertical',
                    outline: 'none',
                    fontSize: 18,
                  }}
                />
              </div>

              {/* Probability Text is AI */}
              <div style={{ marginTop: 16 }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'baseline',
                    margin: '0 0 8px 0',
                  }}
                >
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
                    textAlign: 'center',
                    fontSize: 24,
                    fontWeight: 700,
                  }}
                >
                  {detectProb == null ? '' : `${detectProb * 100}%`}
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
