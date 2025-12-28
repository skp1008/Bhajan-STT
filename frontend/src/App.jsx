import React, { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  const [csvFile, setCsvFile] = useState(null)
  const [audioFile, setAudioFile] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [lyricsBase, setLyricsBase] = useState([])
  const [currentLine, setCurrentLine] = useState(1)
  const [asrOutputs, setAsrOutputs] = useState([])
  const [lineHistory, setLineHistory] = useState([])
  const wsRef = useRef(null)

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  const handleCsvUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setCsvFile(file)
    }
  }

  const handleAudioUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setAudioFile(file)
    }
  }

  const startProcessing = async () => {
    if (!csvFile || !audioFile) {
      alert('Please upload both CSV and audio files')
      return
    }

    setIsProcessing(true)
    setProgress(0)
    setAsrOutputs([])
    setLineHistory([])
    setCurrentLine(1)

    const formData = new FormData()
    formData.append('csv_file', csvFile)
    formData.append('audio_file', audioFile)

    try {
      // Upload files
      const uploadResponse = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      })

      if (!uploadResponse.ok) {
        throw new Error('Upload failed')
      }

      const { csv_filename, audio_filename } = await uploadResponse.json()

      // Connect WebSocket
      const ws = new WebSocket(`ws://localhost:8000/ws/process?csv_file=${csv_filename}&audio_file=${audio_filename}`)
      wsRef.current = ws

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        
        if (data.type === 'progress') {
          setProgress(data.progress)
        } else if (data.type === 'lyrics_loaded') {
          setLyricsBase(data.lyrics)
        } else if (data.type === 'asr_output') {
          setAsrOutputs(prev => [...prev, {
            timestamp: data.timestamp,
            asr_text: data.asr_text,
            action: data.action,
            line_number: data.line_number,
          }])
          
          if (data.action === 'switch') {
            setCurrentLine(data.line_number)
            setLineHistory(prev => [...prev, {
              timestamp: data.timestamp,
              line_number: data.line_number,
              lyrics: data.lyrics_line,
            }])
          } else if (data.action === 'stay') {
            // Update the last entry if it's a stay at the same line
            setLineHistory(prev => {
              const newHistory = [...prev]
              if (newHistory.length > 0 && newHistory[newHistory.length - 1].line_number === data.line_number) {
                newHistory[newHistory.length - 1].timestamp = data.timestamp
              }
              return newHistory
            })
          }
        } else if (data.type === 'complete') {
          setIsProcessing(false)
          setProgress(100)
        } else if (data.type === 'error') {
          alert(`Error: ${data.message}`)
          setIsProcessing(false)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setIsProcessing(false)
      }

      ws.onclose = () => {
        setIsProcessing(false)
      }
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to start processing')
      setIsProcessing(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">AI Bhajan Lyrics Listener</h1>
        <p className="subtitle">Real-time Devanagari lyrics synchronization</p>
      </header>

      <main className="main-content">
        <div className="upload-section">
          <div className="upload-box">
            <h3>Upload CSV File</h3>
            <input
              type="file"
              accept=".csv"
              onChange={handleCsvUpload}
              disabled={isProcessing}
              className="file-input"
              id="csv-upload"
            />
            <label htmlFor="csv-upload" className="file-label">
              {csvFile ? csvFile.name : 'Choose CSV file'}
            </label>
          </div>

          <div className="upload-box">
            <h3>Upload Audio File</h3>
            <input
              type="file"
              accept="audio/*"
              onChange={handleAudioUpload}
              disabled={isProcessing}
              className="file-input"
              id="audio-upload"
            />
            <label htmlFor="audio-upload" className="file-label">
              {audioFile ? audioFile.name : 'Choose audio file'}
            </label>
          </div>

          <button
            onClick={startProcessing}
            disabled={isProcessing || !csvFile || !audioFile}
            className="process-button"
          >
            {isProcessing ? 'Processing...' : 'Start Processing'}
          </button>

          {isProcessing && (
            <div className="progress-container">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }}></div>
              </div>
              <span className="progress-text">{progress}%</span>
            </div>
          )}
        </div>

        <div className="output-section">
          <div className="output-box asr-box">
            <h2>ASR Output</h2>
            <div className="output-content">
              {asrOutputs.length === 0 ? (
                <p className="empty-state">No output yet. Start processing to see results.</p>
              ) : (
                <div className="asr-list">
                  {asrOutputs.map((output, idx) => (
                    <div key={idx} className="asr-item">
                      <div className="timestamp">{output.timestamp}</div>
                      <div className="asr-text">{output.asr_text}</div>
                      <div className={`action ${output.action}`}>
                        {output.action === 'switch' ? `→ Switch to Line ${output.line_number}` : `● Stay at Line ${output.line_number}`}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="output-box lyrics-box">
            <h2>Current Lyrics</h2>
            <div className="output-content">
              {lineHistory.length === 0 ? (
                <div className="empty-state">
                  {lyricsBase.length > 0 ? (
                    <div className="current-line-display">
                      <div className="line-number">Line {currentLine}</div>
                      <div className="lyrics-text">{lyricsBase[currentLine - 1] || ''}</div>
                    </div>
                  ) : (
                    <p>No lyrics loaded yet.</p>
                  )}
                </div>
              ) : (
                <div className="lyrics-list">
                  {lineHistory.map((item, idx) => (
                    <div key={idx} className="lyrics-item">
                      <div className="lyrics-timestamp">{item.timestamp}</div>
                      <div className="lyrics-line-number">Line {item.line_number}</div>
                      <div className="lyrics-text">{item.lyrics}</div>
                    </div>
                  ))}
                  {currentLine > 0 && lyricsBase.length > 0 && (
                    <div className="current-line-display active">
                      <div className="line-number">Line {currentLine}</div>
                      <div className="lyrics-text">{lyricsBase[currentLine - 1]}</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App

