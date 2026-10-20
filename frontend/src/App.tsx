import { useEffect, useState } from 'react'
import { fileUrl, listJobs, uploadVideo, type Job, type Player } from './api'

const ACTIVE = ['pending', 'processing']

function ProgressBar({ value }: { value: number }) {
  return (
    <div className="bar">
      <div style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  )
}

function PlayerCard({ jobId, p, onOpen }: { jobId: string; p: Player; onOpen: () => void }) {
  return (
    <button className="card" onClick={onOpen}>
      <img src={fileUrl(jobId, `player_${p.track_id}.png`)} alt={`Player ${p.track_id} heatmap`} loading="lazy" />
      <div className="meta">
        <strong>Player {p.track_id}</strong>
        <span className={`badge ${p.reliability}`}>{p.reliability}</span>
      </div>
      <small>
        tracked {p.coverage_pct.toFixed(0)}% | ~{p.approx_distance_m.toFixed(0)} m
      </small>
    </button>
  )
}

function JobView({ job }: { job: Job }) {
  const [open, setOpen] = useState<Player | null>(null)
  const [hideLow, setHideLow] = useState(false)

  if (ACTIVE.includes(job.status)) {
    return (
      <section>
        <h2>{job.filename}</h2>
        <p>{job.status === 'pending' ? 'Waiting in queue...' : `Processing ${Math.round(job.progress * 100)}%`}</p>
        <ProgressBar value={job.progress} />
      </section>
    )
  }
  if (job.status === 'failed') {
    return (
      <section>
        <h2>{job.filename}</h2>
        <p className="error">Failed: {job.error}</p>
      </section>
    )
  }

  const players = job.players.filter((p) => !(hideLow && p.reliability === 'low'))
  return (
    <section>
      <h2>{job.filename}</h2>
      <h3>1. Find your player</h3>
      <p>Each box shows the player number. Open that player's heatmap below.</p>
      <img className="overview" src={fileUrl(job.id, 'overview.png')} alt="Players with IDs" />
      <h3>2. Heatmaps ({players.length})</h3>
      <label>
        <input type="checkbox" checked={hideLow} onChange={(e) => setHideLow(e.target.checked)} /> Hide low reliability
      </label>
      <div className="grid">
        {players.map((p) => (
          <PlayerCard key={p.track_id} jobId={job.id} p={p} onOpen={() => setOpen(p)} />
        ))}
      </div>
      {open && (
        <div className="modal" onClick={() => setOpen(null)}>
          <div onClick={(e) => e.stopPropagation()}>
            <img src={fileUrl(job.id, `player_${open.track_id}.png`)} alt={`Player ${open.track_id}`} />
            <a href={fileUrl(job.id, `player_${open.track_id}.png`)} download={`player_${open.track_id}.png`}>
              Download PNG
            </a>{' '}
            <button onClick={() => setOpen(null)}>Close</button>
          </div>
        </div>
      )}
    </section>
  )
}

export default function App() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)

  const refresh = () => listJobs().then(setJobs).catch((e) => setError(e.message))

  useEffect(() => {
    refresh()
  }, [])

  const anyActive = jobs.some((j) => ACTIVE.includes(j.status))
  useEffect(() => {
    if (!anyActive) return
    const t = setInterval(refresh, 2000)
    return () => clearInterval(t)
  }, [anyActive])

  async function onFile(file: File | undefined) {
    if (!file) return
    setError(null)
    setUploading(true)
    try {
      const job = await uploadVideo(file)
      setSelected(job.id)
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setUploading(false)
    }
  }

  const current = jobs.find((j) => j.id === selected) ?? null

  return (
    <div className="layout">
      <aside>
        <h1>PlayerVision</h1>
        <label className="upload">
          {uploading ? 'Uploading...' : 'Upload match clip'}
          <input type="file" accept="video/*" disabled={uploading} onChange={(e) => onFile(e.target.files?.[0])} hidden />
        </label>
        {error && <p className="error">{error}</p>}
        <ul>
          {jobs.map((j) => (
            <li key={j.id}>
              <button className={j.id === selected ? 'active' : ''} onClick={() => setSelected(j.id)}>
                {j.filename}
                <small className={`status ${j.status}`}>{j.status}</small>
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <main>{current ? <JobView key={current.id} job={current} /> : <p>Upload a clip or pick a previous job.</p>}</main>
    </div>
  )
}
