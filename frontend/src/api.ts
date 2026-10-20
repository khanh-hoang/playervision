export type Player = {
  track_id: number
  frames: number
  coverage_pct: number
  reliability: 'high' | 'medium' | 'low'
  approx_distance_m: number
}

export type Job = {
  id: string
  filename: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  error: string | null
  players: Player[]
}

async function check(r: Response) {
  if (!r.ok) {
    const body = await r.json().catch(() => null)
    throw new Error(body?.detail ?? `Request failed (${r.status})`)
  }
  return r
}

export const listJobs = async (): Promise<Job[]> => (await check(await fetch('/api/jobs'))).json()

export async function uploadVideo(file: File): Promise<Job> {
  const form = new FormData()
  form.append('file', file)
  return (await check(await fetch('/api/jobs', { method: 'POST', body: form }))).json()
}

export const fileUrl = (jobId: string, name: string) => `/api/jobs/${jobId}/files/${name}`
