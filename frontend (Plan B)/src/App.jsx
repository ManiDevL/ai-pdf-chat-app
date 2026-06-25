import { useId, useState } from 'react'
import './App.css'

const initialPdfs = [
  { id: 1, name: 'Annual research brief.pdf', pages: 48, status: 'Indexed' },
  { id: 2, name: 'Model evaluation notes.pdf', pages: 23, status: 'Ready' },
  { id: 3, name: 'Customer discovery transcript.pdf', pages: 16, status: 'Ready' },
]

const initialMessages = [
  {
    id: 1,
    role: 'assistant',
    text: 'Upload a PDF or choose one from the library, then ask a question. I can summarize sections, compare findings, and point you to cited pages.',
    meta: 'PDF assistant',
  },
  {
    id: 2,
    role: 'user',
    text: 'Summarize the main risks in the research brief.',
    meta: 'You',
  },
  {
    id: 3,
    role: 'assistant',
    text: 'The brief highlights three core risks: unclear data provenance, delayed review cycles, and inconsistent evaluation criteria across teams. The strongest recommendation is to standardize review checkpoints before expanding usage.',
    meta: 'Annual research brief.pdf',
  },
]

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 15V4m0 0 4 4m-4-4-4 4" />
      <path d="M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3" />
    </svg>
  )
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m5 12 14-7-4 14-3-6-7-1Z" />
      <path d="m12 13 7-8" />
    </svg>
  )
}

function FileIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 3h7l4 4v14H7z" />
      <path d="M14 3v5h4" />
      <path d="M9 13h6M9 17h4" />
    </svg>
  )
}

function Sidebar({ documents, activeId, onSelectDocument, onUpload }) {
  const uploadId = useId()

  return (
    <aside className="sidebar" aria-label="Uploaded PDFs">
      <div className="sidebarHeader">
        <div>
          <p className="eyebrow">Workspace</p>
          <h1>PDF Chat</h1>
        </div>
        <span className="statusDot" aria-label="Online" />
      </div>

      <label className="uploadButton" htmlFor={uploadId}>
        <UploadIcon />
        Upload PDF
      </label>
      <input
        id={uploadId}
        className="fileInput"
        type="file"
        accept="application/pdf,.pdf"
        multiple
        onChange={onUpload}
      />

      <div className="documentSection">
        <div className="sectionTitle">
          <span>Documents</span>
          <span>{documents.length}</span>
        </div>
        <div className="documentList">
          {documents.map((document) => (
            <button
              className={`documentItem ${document.id === activeId ? 'active' : ''}`}
              key={document.id}
              type="button"
              onClick={() => onSelectDocument(document.id)}
            >
              <span className="fileBadge">
                <FileIcon />
              </span>
              <span className="documentText">
                <span className="documentName">{document.name}</span>
                <span className="documentMeta">
                  {document.pages ? `${document.pages} pages` : 'New upload'} · {document.status}
                </span>
              </span>
            </button>
          ))}
        </div>
      </div>
    </aside>
  )
}

function ChatHeader({ activeDocument }) {
  return (
    <header className="chatHeader">
      <div>
        <p className="eyebrow">Current PDF</p>
        <h2>{activeDocument?.name ?? 'No document selected'}</h2>
      </div>
      <div className="headerStats" aria-label="Document status">
        <span>{activeDocument?.status ?? 'Waiting'}</span>
        <span>{activeDocument?.pages ? `${activeDocument.pages} pages` : 'Upload to start'}</span>
      </div>
    </header>
  )
}

function MessageBubble({ message }) {
  return (
    <article className={`message ${message.role}`}>
      <div className="avatar">{message.role === 'assistant' ? 'AI' : 'ME'}</div>
      <div className="messageContent">
        <div className="messageMeta">{message.meta}</div>
        <p>{message.text}</p>
      </div>
    </article>
  )
}

function ChatWindow({ messages }) {
  return (
    <main className="chatWindow" aria-label="Chat messages">
      <div className="messageStack">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
      </div>
    </main>
  )
}

function MessageInput({ onSend }) {
  const [value, setValue] = useState('')

  function handleSubmit(event) {
    event.preventDefault()
    const trimmedValue = value.trim()
    if (!trimmedValue) return
    onSend(trimmedValue)
    setValue('')
  }

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <textarea
        aria-label="Ask a question about the selected PDF"
        placeholder="Ask about summaries, clauses, risks, citations..."
        rows="1"
        value={value}
        onChange={(event) => setValue(event.target.value)}
      />
      <button type="submit" aria-label="Send message">
        <SendIcon />
      </button>
    </form>
  )
}

function App() {
  const [documents, setDocuments] = useState(initialPdfs)
  const [activeId, setActiveId] = useState(initialPdfs[0].id)
  const [messages, setMessages] = useState(initialMessages)
  const activeDocument = documents.find((document) => document.id === activeId)

  function handleUpload(event) {
    const uploadedFiles = Array.from(event.target.files ?? [])
    if (!uploadedFiles.length) return

    const nextDocuments = uploadedFiles.map((file, index) => ({
      id: Date.now() + index,
      name: file.name,
      pages: null,
      status: 'Processing',
    }))

    setDocuments((currentDocuments) => [...nextDocuments, ...currentDocuments])
    setActiveId(nextDocuments[0].id)
    event.target.value = ''
  }

  function handleSend(text) {
    setMessages((currentMessages) => [
      ...currentMessages,
      {
        id: Date.now(),
        role: 'user',
        text,
        meta: 'You',
      },
      {
        id: Date.now() + 1,
        role: 'assistant',
        text: `I would search ${activeDocument?.name ?? 'the selected PDF'} for evidence and return an answer with page references once document parsing is connected.`,
        meta: activeDocument?.name ?? 'PDF assistant',
      },
    ])
  }

  return (
    <div className="appShell">
      <Sidebar
        documents={documents}
        activeId={activeId}
        onSelectDocument={setActiveId}
        onUpload={handleUpload}
      />
      <section className="chatPanel">
        <ChatHeader activeDocument={activeDocument} />
        <ChatWindow messages={messages} />
        <MessageInput onSend={handleSend} />
      </section>
    </div>
  )
}

export default App
