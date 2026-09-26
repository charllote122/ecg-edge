import { useEffect, useState } from 'react'

const API_BASE = '/api'

const CLASS_COLORS = {
  NORM: 'bg-green-500/20 text-green-300 border-green-500/50',
  MI: 'bg-red-500/20 text-red-300 border-red-500/50',
  STTC: 'bg-orange-500/20 text-orange-300 border-orange-500/50',
  CD: 'bg-blue-500/20 text-blue-300 border-blue-500/50',
  HYP: 'bg-purple-500/20 text-purple-300 border-purple-500/50',
}

function StatCard({ label, value, sub }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <div className="text-xs uppercase tracking-wider text-neutral-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {sub && <div className="mt-1 text-xs text-neutral-500">{sub}</div>}
    </div>
  )
}

function AlertRow({ alert }) {
  const detected = alert.detected.split(',').filter(Boolean)
  const topColor = CLASS_COLORS[alert.top_class] || 'bg-neutral-500/20 text-neutral-300'

  return (
    <div className="flex items-center gap-3 border-b border-neutral-800 px-4 py-3 text-sm hover:bg-neutral-900/50">
      <div className="w-12 text-neutral-600 font-mono">#{alert.id}</div>
      <div className="w-28 font-mono text-neutral-400">{alert.device_id}</div>
      <div className={`rounded border px-2 py-0.5 text-xs font-medium ${topColor}`}>
        {alert.top_class} {(alert.top_prob * 100).toFixed(1)}%
      </div>
      <div className="flex gap-1 flex-wrap">
        {detected.map((c) => (
          <span
            key={c}
            className={`rounded border px-1.5 py-0.5 text-[10px] ${CLASS_COLORS[c] || ''}`}
          >
            {c}
          </span>
        ))}
      </div>
      <div className="ml-auto text-xs text-neutral-500 font-mono">
        {new Date(alert.timestamp).toLocaleTimeString()}
      </div>
      <div className="w-20 text-right text-xs text-neutral-500">
        {alert.latency_ms.toFixed(0)} ms
      </div>
    </div>
  )
}

export default function App() {
  const [alerts, setAlerts] = useState([])
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  async function refresh() {
    try {
      const [h, a] = await Promise.all([
        fetch(`${API_BASE}/health`).then((r) => r.json()),
        fetch(`${API_BASE}/alerts?limit=50`).then((r) => r.json()),
      ])
      setHealth(h)
      setAlerts(a)
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
    const interval = setInterval(refresh, 3000) // poll every 3 seconds
    return () => clearInterval(interval)
  }, [])

  // Compute stats
  const totalAlerts = alerts.length
  const classCounts = alerts.reduce((acc, a) => {
    acc[a.top_class] = (acc[a.top_class] || 0) + 1
    return acc
  }, {})
  const avgLatency = alerts.length
    ? (alerts.reduce((s, a) => s + a.latency_ms, 0) / alerts.length).toFixed(0)
    : 0

  return (
    <div className="min-h-screen p-6">
      <header className="mb-6">
        <div className="flex items-baseline gap-3">
          <h1 className="text-3xl font-bold">ECG Edge</h1>
          <span className="text-sm text-neutral-500">
            Real-time arrhythmia detection dashboard
          </span>
        </div>
        <div className="mt-1 flex items-center gap-2 text-xs">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              health?.status === 'ok' ? 'bg-green-500 animate-pulse' : 'bg-red-500'
            }`}
          />
          <span className="text-neutral-500">
            {health
              ? `API ${health.status} · model ${health.model_loaded ? 'loaded' : 'missing'} · db ${health.db_connected ? 'ok' : 'down'}`
              : 'connecting...'}
          </span>
        </div>
      </header>

      {/* Stats grid */}
      <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard label="Total alerts" value={totalAlerts} />
        <StatCard
          label="Top class"
          value={
            Object.entries(classCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || '—'
          }
        />
        <StatCard label="Avg latency" value={`${avgLatency} ms`} />
        <StatCard
          label="Devices"
          value={new Set(alerts.map((a) => a.device_id)).size}
        />
      </div>

      {/* Class distribution */}
      <div className="mb-6 rounded-lg border border-neutral-800 bg-neutral-900 p-4">
        <div className="mb-3 text-sm font-medium text-neutral-300">
          Class distribution
        </div>
        <div className="flex gap-2 flex-wrap">
          {Object.keys(CLASS_COLORS).map((cls) => (
            <div
              key={cls}
              className={`rounded border px-3 py-1.5 text-xs ${CLASS_COLORS[cls]}`}
            >
              {cls}: <span className="font-semibold">{classCounts[cls] || 0}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Alerts table */}
      <div className="rounded-lg border border-neutral-800 bg-neutral-900 overflow-hidden">
        <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
          <div className="text-sm font-medium">Recent alerts</div>
          <div className="text-xs text-neutral-500">
            auto-refresh every 3s
          </div>
        </div>

        {loading && (
          <div className="px-4 py-8 text-center text-sm text-neutral-500">
            Loading...
          </div>
        )}

        {error && (
          <div className="px-4 py-8 text-center text-sm text-red-400">
            Error: {error}
          </div>
        )}

        {!loading && alerts.length === 0 && !error && (
          <div className="px-4 py-8 text-center text-sm text-neutral-500">
            No alerts yet. Run <code className="text-neutral-300">python -m edge.run_demo</code> to generate some.
          </div>
        )}

        {alerts.map((a) => (
          <AlertRow key={a.id} alert={a} />
        ))}
      </div>

      <footer className="mt-6 text-xs text-neutral-600">
        Endpoint: {API_BASE}/alerts · Model: ONNX INT8
      </footer>
    </div>
  )
}
