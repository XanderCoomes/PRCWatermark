import React, { useState, useRef, useEffect } from 'react'

export default function App() {
  const [text, setText] = useState('')
  const [response, setResponse] = useState('')
  const [targetCountInput, setTargetCountInput] = useState('')
  const [targetCount, setTargetCount] = useState(0)
  const [isWatermark, setIsWatermark] = useState(true)
  const [temperature, setTemperature] = useState(1.0)
  const [detectText, setDetectText] = useState('')
  const [detectProb, setDetectProb] = useState(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectStatusText, setDetectStatusText] = useState('')
  const [copyStatus, setCopyStatus] = useState('Copy')

  const detectControllerRef = useRef(null)
  const dotsIntervalRef = useRef(null)

  // --- Temperature slider refs/state ---
  const tempBarRef = useRef(null)
  const isDraggingTempRef = useRef(false)

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
    minWidth: 0,
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
        body: JSON.stringify({
          story,
          word_count: wc,
          temperature,
          is_watermarked: isWatermark,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      if (!res.body) {
        setResponse(await res.text())
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

  function handleTextareaKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (canGenerate(e.currentTarget.value, targetCountInput)) runCheck()
    }
  }

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
    if (targetCountInput === '') setTargetCount(0)
  }

  function handleWordCountKeyDown(e) {
    if (e.key === 'Enter') {
      e.preventDefault()
      let v = e.currentTarget.value.replace(/[^\d]/g, '')
      if (v.length > 1) v = v.replace(/^0+(?=\d)/, '')
      const parsed = parseInt(v || '0', 10)
      setTargetCountInput(v)
      setTargetCount(parsed)
      if (canGenerate(text, v)) runCheck(parsed)
    }
  }

  // ----- Detect -----
  async function runDetect() {
    const payload = detectText.trim()
    if (!payload || isDetecting) return
    if (detectControllerRef.current) detectControllerRef.current.abort()
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
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setDetectProb(typeof data.prob === 'number' ? data.prob : null)
    } catch (err) {
      if (err?.name !== 'AbortError') {
        console.error(err)
        setDetectProb(null)
      }
    } finally {
      stopDetectingDots()
      if (detectControllerRef.current === controller) detectControllerRef.current = null
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

  // Copy function
  function copyResponse() {
    if (response) {
      navigator.clipboard.writeText(response).then(() => {
        setCopyStatus('Copied!')
        setTimeout(() => setCopyStatus('Copy'), 1200)
      })
    }
  }

  // ---- Temperature visuals and interactions ----
  const tempPct = Math.max(0, Math.min(1, (temperature - 0.5) / 1.0)) // 0..1
  const tempHue = 200 - 190 * tempPct // blue -> red
  const tempColor = `hsl(${tempHue}, 80%, 55%)`
  const tempEmoji = temperature < 0.8 ? '🧊' : temperature < 1.2 ? '🌡️' : '🔥'

  function pctToTemp(p) {
    const clamped = Math.max(0, Math.min(1, p))
    return 0.5 + clamped * 1.0 // maps 0..1 to 0.5..1.5
  }
  function tempToPct(t) {
    return Math.max(0, Math.min(1, (t - 0.5) / 1.0))
  }
  function updateTempFromClientX(clientX) {
    const el = tempBarRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const p = (clientX - rect.left) / rect.width
    setTemperature(parseFloat(pctToTemp(p).toFixed(2)))
  }

  // Drag handlers
  useEffect(() => {
    function onMove(e) {
      if (!isDraggingTempRef.current) return
      if (e.type === 'mousemove') {
        updateTempFromClientX(e.clientX)
      } else if (e.type === 'touchmove') {
        if (e.touches && e.touches[0]) updateTempFromClientX(e.touches[0].clientX)
      }
    }
    function onUp() {
      isDraggingTempRef.current = false
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    window.addEventListener('touchmove', onMove, { passive: false })
    window.addEventListener('touchend', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      window.removeEventListener('touchmove', onMove)
      window.removeEventListener('touchend', onUp)
    }
  }, [])

  function handleTempBarMouseDown(e) {
    isDraggingTempRef.current = true
    updateTempFromClientX(e.clientX)
  }
  function handleTempBarTouchStart(e) {
    isDraggingTempRef.current = true
    if (e.touches && e.touches[0]) updateTempFromClientX(e.touches[0].clientX)
  }
  function handleTempBarClick(e) {
    // Single click anywhere on the bar sets the temperature
    updateTempFromClientX(e.clientX)
  }
  function handleTempBarKeyDown(e) {
    // Accessible keyboard interactions
    const stepSmall = 0.01
    const stepBig = 0.05
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
      setTemperature(t => Math.min(1.5, parseFloat((t + stepSmall).toFixed(2))))
      e.preventDefault()
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
      setTemperature(t => Math.max(0.5, parseFloat((t - stepSmall).toFixed(2))))
      e.preventDefault()
    } else if (e.key === 'PageUp') {
      setTemperature(t => Math.min(1.5, parseFloat((t + stepBig).toFixed(2))))
      e.preventDefault()
    } else if (e.key === 'PageDown') {
      setTemperature(t => Math.max(0.5, parseFloat((t - stepBig).toFixed(2))))
      e.preventDefault()
    } else if (e.key === 'Home') {
      setTemperature(0.5)
      e.preventDefault()
    } else if (e.key === 'End') {
      setTemperature(1.5)
      e.preventDefault()
    }
  }

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
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '2fr 1.1fr',
            gap: 28,
            alignItems: 'stretch',
            height: 'calc(100vh - 56px)',
            minHeight: 0,
          }}
        >
          {/* LEFT: Generator */}
          <div style={{ minHeight: 0, minWidth: 0 }}>
            <div style={cardStyle}>
              {/* Response area */}
              <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 style={{ color: '#38bdf8', marginTop: 0, fontSize: 22, lineHeight: 1.3 }}>Response</h3>
                  <button
                    onClick={copyResponse}
                    style={{
                      background: '#1e293b',
                      color: '#38bdf8',
                      border: '1px solid #38bdf8',
                      borderRadius: 6,
                      padding: '4px 8px',
                      cursor: 'pointer',
                      fontSize: 14,
                      transition: 'all 0.2s ease',
                    }}
                  >
                    {copyStatus}
                  </button>
                </div>
                <div
                  style={{
                    ...boxStyle,
                    flex: 1,
                    minHeight: 0,
                    overflowY: 'auto',
                  }}
                >
                  {response}
                </div>
              </div>

              {/* Prompt + Word Count + Watermark + Temperature */}
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

                {/* Inputs row */}
                <div style={{ display: 'flex', gap: 14, minHeight: 0 }}>
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
                      overflowY: 'auto',
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

                {/* Watermark toggle */}
                <div style={{ display: 'flex', gap: 10, marginTop: 12, flexWrap: 'wrap' }}>
                  <button
                    type="button"
                    aria-pressed={isWatermark}
                    onClick={() => setIsWatermark(true)}
                    style={{
                      padding: '10px 14px',
                      borderRadius: 10,
                      border: isWatermark ? '4px solid #38bdf8' : '0.5px solid #38bdf8',
                      cursor: 'pointer',
                      background: '#334155',
                      color: '#e2e8f0',
                      fontWeight: 600,
                    }}
                  >
                    Watermark
                  </button>
                  <button
                    type="button"
                    aria-pressed={!isWatermark}
                    onClick={() => setIsWatermark(false)}
                    style={{
                      padding: '10px 14px',
                      borderRadius: 10,
                      border: !isWatermark ? '4px solid #38bdf8' : '0.5px solid #38bdf8',
                      cursor: 'pointer',
                      background: '#334155',
                      color: '#e2e8f0',
                      fontWeight: 600,
                    }}
                  >
                    Don’t Watermark
                  </button>
                </div>

                {/* Temperature slider (click/drag/keyboard) */}
                <div style={{ marginTop: 16 }}>
                  <h3 style={{ color: '#38bdf8', margin: '0 0 8px 0', fontSize: 20, lineHeight: 1.3 }}>
                    Model Temperature
                  </h3>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ color: tempColor, fontWeight: 700 }}>
                      {tempEmoji} {temperature.toFixed(2)}
                    </div>
                  </div>

                  {/* Interactive temperature bar */}
                  <div
                    ref={tempBarRef}
                    role="slider"
                    aria-label="Model temperature"
                    aria-valuemin={0.5}
                    aria-valuemax={1.5}
                    aria-valuenow={Number(temperature.toFixed(2))}
                    tabIndex={0}
                    onKeyDown={handleTempBarKeyDown}
                    onMouseDown={handleTempBarMouseDown}
                    onTouchStart={handleTempBarTouchStart}
                    onClick={handleTempBarClick}
                    style={{
                      marginTop: 8,
                      height: 16,
                      borderRadius: 999,
                      background: '#0b1220',
                      border: '1px solid #1f2937',
                      position: 'relative',
                      cursor: 'pointer',
                      outline: 'none',
                      boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.25)',
                    }}
                  >
                    {/* fill */}
                    <div
                      style={{
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        bottom: 0,
                        width: `${tempToPct(temperature) * 100}%`,
                        background: tempColor,
                        borderRadius: 999,
                        transition: isDraggingTempRef.current ? 'none' : 'width 120ms linear, background 120ms linear',
                      }}
                    />
                    {/* knob */}
                    <div
                      style={{
                        position: 'absolute',
                        top: '50%',
                        left: `calc(${tempToPct(temperature) * 100}% - 8px)`,
                        transform: 'translateY(-50%)',
                        width: 16,
                        height: 16,
                        borderRadius: '50%',
                        background: '#e2e8f0',
                        boxShadow: '0 0 0 2px ' + tempColor,
                        pointerEvents: 'none',
                      }}
                    />
                  </div>
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

              <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
                <textarea
                  value={detectText}
                  onChange={(e) => setDetectText(e.target.value)}
                  onKeyDown={handleDetectKeyDown}
                  placeholder="Enter text to check for AI"
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

              {/* Probability Text is Watermarked */}
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
                    Probability Text is Watermarked
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
                  {detectProb == null ? '' : `${(detectProb * 100).toFixed(1)}%`}
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
