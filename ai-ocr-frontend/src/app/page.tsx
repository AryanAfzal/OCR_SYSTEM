"use client"

import { useState, useCallback, useEffect } from "react"
import { useDropzone } from "react-dropzone"
import axios from "axios"
import { FileText, Upload, CheckCircle, AlertCircle, Loader2, Clock, ChevronRight } from "lucide-react"

const API = "http://127.0.0.1:8000"

// ── Types ──────────────────────────────────────────────────────────
type Question = {
  question:   string
  answer:     string
  confidence: number
}
type OCRResult = {
  id:          number
  filename:    string
  total_pages: number
  questions:   Question[]
  raw_text?:   { text: string; confidence: number }[]
  images?:     string[]
}
type HistoryItem = {
  id:          number
  filename:    string
  total_pages: number
  uploaded_at: string
}

// ── Main Page ──────────────────────────────────────────────────────
export default function Home() {
  const [status,   setStatus]   = useState<"idle"|"uploading"|"done"|"error">("idle")
  const [result,   setResult]   = useState<OCRResult | null>(null)
  const [fileName, setFileName] = useState("")
  const [errorMsg, setErrorMsg] = useState("")
  const [history,  setHistory]  = useState<HistoryItem[]>([])
  
  // Real-time progress state
  const [progress, setProgress] = useState(0)
  const [statusText, setStatusText] = useState("")

  // Load history when page opens
  useEffect(() => { fetchHistory() }, [])

  async function fetchHistory() {
    try {
      const res = await axios.get(`${API}/history`)
      setHistory(res.data)
    } catch {
      // backend might not be running yet — fail silently
    }
  }

  async function loadHistoryItem(id: number) {
    try {
      const res = await axios.get(`${API}/history/${id}`)
      setResult(res.data)
      setStatus("done")
    } catch {
      setErrorMsg("Could not load this result.")
      setStatus("error")
    }
  }

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    setFileName(file.name)
    setStatus("uploading")
    setResult(null)
    setErrorMsg("")
    setProgress(0)
    setStatusText("")

    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await fetch(`${API}/upload`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Upload failed with status ${response.status}`)
      }
      if (!response.body) {
        throw new Error("No response body from server")
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder("utf-8")
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (!line.trim()) continue
          const data = JSON.parse(line)
          
          if (data.error) {
            throw new Error(data.error)
          }
          if (data.progress !== undefined) {
            setProgress(data.progress)
          }
          if (data.status) {
            setStatusText(data.status)
          }
          if (data.result) {
            setResult(data.result)
            setStatus("done")
            fetchHistory()
          }
        }
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Upload failed. Is the backend running?")
      setStatus("error")
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
  })

  return (
    <main className="min-h-screen bg-gray-950 text-white flex">

      {/* ── Sidebar ── */}
      <aside className="w-64 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
          <h2 className="text-cyan-400 font-bold text-sm uppercase tracking-widest">
            AI Exam Grader
          </h2>
          <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800">
            {history.length}
          </span>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
            Recent Uploads
          </h3>
          {history.length === 0 ? (
            <p className="text-gray-600 text-sm italic">No history yet</p>
          ) : (
            <div className="space-y-2">
              {history.map(item => (
                <button
                  key={item.id}
                  onClick={() => loadHistoryItem(item.id)}
                  className="w-full flex items-center justify-between p-3 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors text-left group"
                >
                  <div className="overflow-hidden">
                    <p className="text-sm font-medium text-gray-300 truncate">
                      {item.filename}
                    </p>
                    <p className="text-xs text-gray-500 mt-1 flex items-center gap-1">
                      <Clock size={12} />
                      <span suppressHydrationWarning>{new Date(item.uploaded_at).toLocaleDateString()}</span>
                    </p>
                  </div>
                  <ChevronRight size={16} className="text-gray-600 group-hover:text-cyan-400" />
                </button>
              ))}
            </div>
          )}
        </div>
      </aside>

      {/* ── Main Content ── */}
      <div className="flex-1 p-8 overflow-y-auto">
        <div className="max-w-6xl mx-auto">
          
          <header className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">Upload Student Answer Sheet</h1>
            <p className="text-gray-400">
              Upload a scanned PDF. The TrOCR engine reads the handwriting and automatically splits questions (e.g. Q#01, Q1).
            </p>
          </header>

          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-2xl p-14 text-center cursor-pointer transition-all duration-200
              ${isDragActive
                ? "border-cyan-400 bg-cyan-950"
                : "border-gray-600 hover:border-cyan-500 hover:bg-gray-900"
              }`}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto mb-4 text-cyan-400" size={44} />
            <p className="text-xl font-semibold mb-1">
              {isDragActive ? "Drop your PDF here!" : "Drag & drop your PDF"}
            </p>
            <p className="text-gray-500 text-sm">or click to browse — PDF files only</p>
          </div>

          {/* Status */}
          <div className="mt-5">
            {status === "uploading" && (
              <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3 text-cyan-400">
                    <Loader2 className="animate-spin" size={20} />
                    <span className="font-medium">{statusText || `Processing ${fileName}...`}</span>
                  </div>
                  <span className="text-cyan-400 font-bold">{progress}%</span>
                </div>
                {/* Progress Bar */}
                <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
                  <div 
                    className="bg-cyan-500 h-2 rounded-full transition-all duration-300 ease-out" 
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}
            {status === "error" && (
              <div className="flex items-center gap-3 text-red-400 bg-red-950 rounded-xl p-4">
                <AlertCircle size={20} />
                <span>{errorMsg}</span>
              </div>
            )}
            {status === "done" && result && (
              <div className="flex items-center gap-3 text-green-400 bg-green-950 rounded-xl p-4">
                <CheckCircle size={20} />
                <span>
                  Done! Found <strong>{result.questions.length}</strong> question(s)
                  across <strong>{result.total_pages}</strong> page(s).
                </span>
              </div>
            )}
          </div>

          {/* Results */}
          {result && <ResultsPanel result={result} />}
        </div>
      </div>
    </main>
  )
}

// ── Results Panel ──────────────────────────────────────────────────
function ResultsPanel({ result }: { result: OCRResult }) {
  const [activeTab, setActiveTab] = useState<"questions"|"raw">("questions")

  return (
    <div className="mt-7 flex flex-col xl:flex-row gap-6">
      
      {/* Left Side: Images (if available) */}
      {result.images && result.images.length > 0 && (
        <div className="w-full xl:w-1/2 bg-gray-900 rounded-2xl overflow-hidden border border-gray-700 flex flex-col">
          <div className="p-4 border-b border-gray-700 font-medium text-gray-300">
            Scanned Document
          </div>
          <div className="flex-1 p-4 overflow-y-auto max-h-[800px] space-y-4 bg-gray-950">
            {result.images.map((imgUrl, i) => (
              <img 
                key={i} 
                src={imgUrl} 
                alt={`Page ${i+1}`} 
                className="w-full rounded-lg shadow-lg border border-gray-800"
              />
            ))}
          </div>
        </div>
      )}

      {/* Right Side: Results */}
      <div className={`w-full ${result.images ? 'xl:w-1/2' : ''} bg-gray-900 rounded-2xl overflow-hidden border border-gray-700 flex flex-col`}>
        <div className="flex border-b border-gray-700">
          {(["questions","raw"] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 text-sm font-medium transition-colors capitalize
                ${activeTab === tab
                  ? "text-cyan-400 border-b-2 border-cyan-400"
                  : "text-gray-500 hover:text-gray-300"
                }`}
            >
              {tab === "questions"
                ? `Questions (${result.questions.length})`
                : `Raw Text (${result.raw_text?.length ?? "?"} lines)`}
            </button>
          ))}
        </div>

        <div className="flex-1 p-6 max-h-[800px] overflow-y-auto">
          {activeTab === "questions" && (
            result.questions.length > 0 ? (
              <div className="space-y-4">
                {result.questions.map((q, i) => (
                  <QuestionCard key={i} question={q} index={i} />
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">
                No questions detected. Make sure the PDF has labels like Q1, Q#01.
              </p>
            )
          )}

          {activeTab === "raw" && (
            result.raw_text && result.raw_text.length > 0 ? (
              <div className="space-y-2">
                {result.raw_text.map((item, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <span className="text-xs text-gray-600 w-8 text-right mt-1 shrink-0">{i + 1}</span>
                    <span className="text-gray-300 text-sm flex-1">{item.text}</span>
                    <span className="text-xs text-gray-600 shrink-0">
                      {(item.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">
                Raw text not available for past uploads.
              </p>
            )
          )}
        </div>
      </div>
    </div>
  )
}

// ── Question Card ──────────────────────────────────────────────────
function QuestionCard({ question, index }: { question: Question; index: number }) {
  const confidence = Math.round(question.confidence * 100)
  const barColor   =
    confidence >= 90 ? "bg-green-500" :
    confidence >= 70 ? "bg-yellow-500" : "bg-red-500"

  return (
    <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex items-center gap-2">
          <FileText size={16} className="text-cyan-400 shrink-0" />
          <span className="text-cyan-400 font-semibold">{question.question}</span>
        </div>
        <span className="text-xs text-gray-500 shrink-0">#{index + 1}</span>
      </div>

      <div className="text-gray-300 text-sm mb-4 leading-relaxed whitespace-pre-line font-sans">
        {question.answer || <span className="text-gray-600 italic">No answer text found</span>}
      </div>

      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-500 w-20 shrink-0">Confidence</span>
        <div className="flex-1 bg-gray-700 rounded-full h-1.5">
          <div
            className={`h-1.5 rounded-full transition-all ${barColor}`}
            style={{ width: `${confidence}%` }}
          />
        </div>
        <span className="text-xs font-medium text-gray-400 w-8 text-right">
          {confidence}%
        </span>
      </div>
    </div>
  )
}
