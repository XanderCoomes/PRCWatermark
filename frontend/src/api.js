const BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

export async function generateText({ prompt, numWords, isWatermarked, signal }) {
  const res = await fetch(`${BASE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      num_words: Number(numWords) || 300,
      is_watermarked: !!isWatermarked,
    }),
    signal,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Backend error ${res.status}: ${text || res.statusText}`)
  }
  const data = await res.json()
  return data.text ?? ''
}
