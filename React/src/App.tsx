import { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Square, Send, Bot, User, Volume2 } from 'lucide-react';

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
};

type RecordingState = 'idle' | 'recording' | 'processing';

function StructuredData({ data }: { data: any }) {
  if (typeof data !== 'object' || data === null) return <p className="text-sm">{String(data)}</p>;

  return (
    <div className="space-y-3">
      <div className="flex items-center space-x-2 pb-2 border-b border-slate-100">
        <Bot className="w-4 h-4 text-indigo-500" />
        <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Medical Extraction</span>
      </div>
      <div className="grid grid-cols-1 gap-3">
        {Object.entries(data).map(([key, value]) => (
          <div key={key} className="bg-slate-50 p-3 rounded-lg border border-slate-100">
            <label className="block text-[10px] font-bold uppercase text-slate-400 mb-1">{key}</label>
            <p className="text-sm text-slate-700 leading-relaxed capitalize">
              {String(value) || <span className="text-slate-300 italic">Not extracted</span>}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [recordingState, setRecordingState] = useState<RecordingState>('idle');
  const [streamingResponse, setStreamingResponse] = useState<string>('');
  const [wsConnected, setWsConnected] = useState(false);
  const [lastExtractedData, setLastExtractedData] = useState<any>(null);
  const [currentVolume, setCurrentVolume] = useState(0);
  const [silenceThreshold, setSilenceThreshold] = useState(50);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentStreamIdRef = useRef<string>('');
  const accumulatedResponseRef = useRef<string>('');

  // VAD Refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const silenceStartRef = useRef<number | null>(null);
  const isRecordingRef = useRef<boolean>(false);
  const lastExtractedDataRef = useRef<any>(null);
  const sessionIdRef = useRef<string | null>(null);
  const vadAnimationRef = useRef<number | null>(null);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const wsUrl = (import.meta as any).env?.VITE_WS_URL || 'ws://localhost:8002/ws';
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'stream_start') {
            currentStreamIdRef.current = data.stream_id || '';
            accumulatedResponseRef.current = '';
            setStreamingResponse('');
          } else if (data.type === 'stream_chunk') {
            accumulatedResponseRef.current += data.text;
            setStreamingResponse(accumulatedResponseRef.current);
          } else if (data.type === 'stream_end') {
            const finalContent = accumulatedResponseRef.current;
            if (finalContent) {
              const newMessage: Message = {
                id: Date.now().toString(),
                role: 'assistant',
                content: finalContent,
                timestamp: new Date(),
              };
              setMessages((prev) => [...prev, newMessage]);

              // Update last extracted data for merging in next chunk
              try {
                if (finalContent.trim().startsWith('{')) {
                  const parsed = JSON.parse(finalContent);
                  setLastExtractedData(parsed);
                  lastExtractedDataRef.current = parsed;
                }
              } catch (e) {
                console.error('Failed to parse final content for state:', e);
              }
            }
            accumulatedResponseRef.current = '';
            setStreamingResponse('');

            if (!isRecordingRef.current) {
              setRecordingState('idle');
            }
          } else if (data.type === 'error') {
            console.error('Backend error:', data.message);
            setRecordingState('idle');
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setWsConnected(false);
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingResponse]);

  const startRecording = async () => {
    try {
      if (vadAnimationRef.current) {
        cancelAnimationFrame(vadAnimationRef.current);
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Setup VAD
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      isRecordingRef.current = true;
      sessionIdRef.current = Date.now().toString();

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        sendAudioToBackend(audioBlob);

        // If we are still "recording" (VAD active), restart immediately
        if (isRecordingRef.current) {
          audioChunksRef.current = [];
          mediaRecorder.start();
        } else {
          // Final stop
          sessionIdRef.current = null;
          stream.getTracks().forEach(track => track.stop());
          if (audioContext.state !== 'closed') {
            audioContext.close();
          }
        }
      };

      mediaRecorder.start(1000);
      setRecordingState('recording');

      // VAD Loop
      const checkVolume = () => {
        if (!isRecordingRef.current || !analyserRef.current) {
          vadAnimationRef.current = requestAnimationFrame(checkVolume);
          return;
        }

        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
          vadAnimationRef.current = requestAnimationFrame(checkVolume);
          return;
        }

        analyserRef.current.getByteFrequencyData(dataArray);
        // Using average but with a higher sensitivity window
        const volume = dataArray.reduce((p, c) => p + c, 0) / dataArray.length;
        setCurrentVolume(volume);

        const SILENCE_DURATION = 1500; // 1.5 seconds

        if (volume < silenceThreshold) {
          if (silenceStartRef.current === null) {
            silenceStartRef.current = Date.now();
          } else {
            const duration = Date.now() - silenceStartRef.current;
            if (duration > SILENCE_DURATION) {
              // Silence detected! Trigger send.
              if (mediaRecorderRef.current.state === 'recording') {
                console.log(`VAD: Silence detected (${duration}ms), stopping to send chunk...`);
                mediaRecorderRef.current.stop();
                silenceStartRef.current = null; // Important: Clear here so we don't trigger again immediately
              }
            }
          }
        } else {
          silenceStartRef.current = null;
        }

        vadAnimationRef.current = requestAnimationFrame(checkVolume);
      };

      vadAnimationRef.current = requestAnimationFrame(checkVolume);

      // Add user message placeholder if first time
      if (messages.length === 0 || messages[messages.length - 1].content !== '🎤 Listening...') {
        const userMessage: Message = {
          id: Date.now().toString(),
          role: 'user',
          content: '🎤 Listening...',
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, userMessage]);
      }
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Unable to access microphone. Please check your permissions.');
    }
  };

  const stopRecording = () => {
    isRecordingRef.current = false;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      setRecordingState('processing');
    }
  };

  const sendAudioToBackend = (blob: Blob) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected');
      return;
    }

    // Use current session ID or fallback to timestamp
    const sessionId = sessionIdRef.current || Date.now().toString();

    // Prepare audio data for WebSocket
    const reader = new FileReader();
    reader.onload = () => {
      const arrayBuffer = reader.result as ArrayBuffer;
      const base64Audio = btoa(
        new Uint8Array(arrayBuffer).reduce(
          (data, byte) => data + String.fromCharCode(byte),
          ''
        )
      );

      const message = {
        type: 'audio',
        session_id: sessionId,
        audio_data: base64Audio,
        mime_type: blob.type || 'audio/webm',
        previous_data: lastExtractedDataRef.current
      };

      ws.send(JSON.stringify(message));
    };

    reader.readAsArrayBuffer(blob);
  };

  const handleMicClick = () => {
    if (recordingState === 'idle') {
      startRecording();
    } else if (recordingState === 'recording') {
      stopRecording();
    }
  };

  const handleEndClick = () => {
    if (recordingState === 'recording') {
      stopRecording();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-white to-zinc-100">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-6 py-4 shadow-sm">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-br from-violet-500 to-indigo-600 rounded-xl">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-slate-900">Voice Chat Bot</h1>
              <p className="text-sm text-slate-500">
                {wsConnected ? (
                  <span className="flex items-center text-green-600">
                    <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
                    Connected
                  </span>
                ) : (
                  <span className="flex items-center text-orange-600">
                    <span className="w-2 h-2 bg-orange-500 rounded-full mr-2"></span>
                    Connecting...
                  </span>
                )}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <div className="p-4 bg-slate-100 rounded-full mb-4">
                <Mic className="w-12 h-12 text-slate-400" />
              </div>
              <h2 className="text-lg font-medium text-slate-700 mb-2">
                Start a conversation
              </h2>
              <p className="text-slate-500 max-w-md">
                Click the microphone button below to start recording. Your voice will be sent to the AI assistant.
              </p>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
            >
              <div
                className={`flex items-start space-x-2 max-w-[80%] ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                  }`}
              >
                <div
                  className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${message.role === 'user'
                    ? 'bg-violet-500'
                    : 'bg-indigo-500'
                    }`}
                >
                  {message.role === 'user' ? (
                    <User className="w-5 h-5 text-white" />
                  ) : (
                    <Bot className="w-5 h-5 text-white" />
                  )}
                </div>
                <div
                  className={`px-4 py-3 rounded-2xl ${message.role === 'user'
                    ? 'bg-violet-500 text-white'
                    : 'bg-white border border-slate-200 shadow-sm text-slate-800'
                    }`}
                >
                  {message.role === 'assistant' && message.content.trim().startsWith('{') ? (
                    (() => {
                      try {
                        const parsed = JSON.parse(message.content);
                        return <StructuredData data={parsed} />;
                      } catch (e) {
                        return <p className="text-sm whitespace-pre-wrap">{message.content}</p>;
                      }
                    })()
                  ) : (
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  )}
                  <p
                    className={`text-xs mt-2 ${message.role === 'user'
                      ? 'text-violet-200'
                      : 'text-slate-400 border-t border-slate-50 pt-1'
                      }`}
                  >
                    {message.timestamp.toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </p>
                </div>
              </div>
            </div>
          ))}

          {/* Streaming response */}
          {streamingResponse && (
            <div className="flex justify-start">
              <div className="flex items-start space-x-2 max-w-[80%]">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <div className="px-4 py-3 rounded-2xl bg-white border border-slate-200 shadow-sm text-slate-800">
                  {streamingResponse.trim().startsWith('{') ? (
                    (() => {
                      try {
                        // Try to parse partial JSON by closing braces (rudimentary)
                        let attempt = streamingResponse;
                        if (!attempt.endsWith('}')) {
                          // Simple check for if we are in a string or not
                          // This is very basic but works for simple flat objects
                          const openCount = (attempt.match(/{/g) || []).length;
                          const closeCount = (attempt.match(/}/g) || []).length;
                          if (openCount > closeCount) {
                            attempt += '}'.repeat(openCount - closeCount);
                          }
                        }
                        const parsed = JSON.parse(attempt);
                        return <StructuredData data={parsed} />;
                      } catch (e) {
                        return <p className="text-sm font-mono whitespace-pre-wrap">{streamingResponse}</p>;
                      }
                    })()
                  ) : (
                    <p className="text-sm">{streamingResponse}</p>
                  )}
                  <div className="flex items-center mt-2 border-t border-slate-50 pt-1">
                    <span className="text-xs text-slate-400">
                      {new Date().toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </span>
                    <span className="ml-2 flex items-center text-xs text-indigo-500">
                      <Volume2 className="w-3 h-3 mr-1 animate-pulse" />
                      Streaming Extraction...
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Controls Area */}
      <div className="bg-white border-t border-slate-200 px-6 py-6">
        <div className="max-w-4xl mx-auto">
          {/* Recording indicator */}
          {recordingState === 'recording' && (
            <div className="flex items-center justify-center mb-4">
              <div className="flex items-center space-x-2 bg-red-50 px-4 py-2 rounded-full">
                <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-red-600">
                  Recording... Click end to send
                </span>
              </div>
            </div>
          )}

          {recordingState === 'processing' && (
            <div className="flex items-center justify-center mb-4">
              <div className="flex items-center space-x-2 bg-indigo-50 px-4 py-2 rounded-full">
                <div className="w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                <span className="text-sm font-medium text-indigo-600">
                  Processing...
                </span>
              </div>
            </div>
          )}

          {/* Control buttons */}
          <div className="flex items-center justify-center space-x-6">
            {/* Mute button (shown during recording) */}
            {recordingState === 'recording' && (
              <button
                className="p-4 bg-slate-100 hover:bg-slate-200 rounded-full transition-colors"
                title="Mute"
                disabled
              >
                <MicOff className="w-6 h-6 text-slate-500" />
              </button>
            )}

            {/* Main recording button */}
            {recordingState === 'idle' ? (
              <button
                onClick={handleMicClick}
                className="p-6 bg-gradient-to-br from-violet-500 to-indigo-600 hover:from-violet-600 hover:to-indigo-700 rounded-full shadow-lg shadow-indigo-200 transition-all transform hover:scale-105 active:scale-95"
                title="Start recording"
              >
                <Mic className="w-8 h-8 text-white" />
              </button>
            ) : recordingState === 'recording' ? (
              <button
                onClick={handleEndClick}
                className="p-6 bg-gradient-to-br from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 rounded-full shadow-lg shadow-red-200 transition-all transform hover:scale-105 active:scale-95"
                title="Stop recording and send"
              >
                <Square className="w-8 h-8 text-white fill-current" />
              </button>
            ) : (
              <div className="p-6 bg-slate-100 rounded-full">
                <div className="w-8 h-8 border-2 border-slate-400 border-t-transparent rounded-full animate-spin"></div>
              </div>
            )}

            {/* Send button (shown during recording as end button) */}
            {recordingState === 'recording' && (
              <button
                onClick={handleEndClick}
                className="p-4 bg-gradient-to-br from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 rounded-full shadow-lg shadow-green-200 transition-colors"
                title="Send recording"
              >
                <Send className="w-6 h-6 text-white" />
              </button>
            )}
          </div>

          {/* Instructions */}
          <div className="flex flex-col items-center mt-4 space-y-2">
            {recordingState === 'recording' && (
              <div className="flex flex-col items-center space-y-3 w-64">
                <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
                  <div
                    className={`h-full transition-all duration-75 ${currentVolume > silenceThreshold ? 'bg-indigo-500' : 'bg-slate-300'
                      }`}
                    style={{ width: `${Math.min(100, (currentVolume / 200) * 100)}%` }}
                  ></div>
                </div>

                <div className="flex flex-col items-center w-full space-y-1">
                  <div className="flex justify-between w-full px-1">
                    <span className="text-[10px] font-bold text-slate-400">QUIET</span>
                    <span className="text-[10px] font-bold text-slate-400">THRESHOLD: {silenceThreshold}</span>
                    <span className="text-[10px] font-bold text-slate-400">LOUD</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="150"
                    value={silenceThreshold}
                    onChange={(e) => setSilenceThreshold(Number(e.target.value))}
                    className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                  />
                  <p className="text-[10px] text-slate-400 mt-1 uppercase font-bold tracking-tighter">
                    Current Level: {currentVolume.toFixed(0)}
                  </p>
                </div>
              </div>
            )}
            <p className="text-center text-sm text-slate-500">
              {recordingState === 'idle'
                ? 'Click the microphone to start recording'
                : recordingState === 'recording'
                  ? 'The bot will automatically send audio when you pause'
                  : 'Processing your audio...'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
