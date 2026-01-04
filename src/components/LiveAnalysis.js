import React, { useState, useRef, useEffect } from 'react';
import './LiveAnalysis.css';

const LiveAnalysis = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [ws, setWs] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    return () => {
      if (ws) {
        ws.close();
      }
      stopCamera();
    };
  }, [ws]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          inlineSize: { ideal: 640 },
          blockSize: { ideal: 480 },
          facingMode: 'user'
        } 
      });
      videoRef.current.srcObject = stream;
      setCameraActive(true);
      setError(null);
    } catch (error) {
      console.error('Error accessing camera:', error);
      setError('Cannot access camera. Please check permissions and make sure no other app is using the camera.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  };

  const drawImageToCanvas = (imgSrc) => {
    if (!canvasRef.current) {
      console.error('Canvas reference is null');
      return;
    }

    const img = new Image();
    img.onload = () => {
      if (!canvasRef.current) return;
      
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = imgSrc;
  };

  const displayNoPoseMessage = () => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('No pose detected', canvas.width / 2, canvas.height / 2);
    ctx.font = '16px Arial';
    ctx.fillText('Ensure your entire body is visible', canvas.width / 2, canvas.height / 2 + 30);
  };

  const startAnalysis = () => {
    if (!cameraActive) {
      setError('Please start camera first');
      return;
    }

    if (!canvasRef.current) {
      setError('Canvas not available. Please refresh the page.');
      return;
    }

    const websocket = new WebSocket('ws://localhost:8000/ws/live-analysis');
    
    websocket.onopen = () => {
      console.log('WebSocket connected');
      setIsRecording(true);
      setWs(websocket);
      setError(null);
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'analysis') {
          setAnalysis(data.analysis);
          drawImageToCanvas(data.frame);
        } else if (data.type === 'no_pose_detected') {
          setAnalysis(null);
          displayNoPoseMessage();
        } else if (data.type === 'error') {
          setError(data.message);
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setIsRecording(false);
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection failed');
      setIsRecording(false);
    };

    const interval = setInterval(() => {
      if (websocket.readyState === WebSocket.OPEN && 
          videoRef.current && 
          videoRef.current.videoWidth > 0) {
        
        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0);
        
        const imageData = canvas.toDataURL('image/jpeg', 0.7);
        websocket.send(imageData);
      }
    }, 100);

    return () => {
      clearInterval(interval);
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.close();
      }
    };
  };

  const stopAnalysis = () => {
    if (ws) {
      ws.close();
    }
    setIsRecording(false);
    setAnalysis(null);
    
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 90) return '#4CAF50';
    if (score >= 70) return '#FFC107';
    if (score >= 50) return '#FF9800';
    return '#F44336';
  };

  return (
    <div className="live-analysis">
      <div className="live-container">
        <div className="controls-section">
          <div className="controls-card">
            <h3>Live Pose Analysis</h3>
            
            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}
            
            <div className="control-buttons">
              {!cameraActive ? (
                <button onClick={startCamera} className="control-button start-camera">
                  📷 Start Camera
                </button>
              ) : (
                <button onClick={stopCamera} className="control-button stop-camera">
                  📷 Stop Camera
                </button>
              )}

              {!isRecording ? (
                <button 
                  onClick={startAnalysis} 
                  disabled={!cameraActive}
                  className="control-button start-analysis"
                >
                  🎬 Start Live Analysis
                </button>
              ) : (
                <button onClick={stopAnalysis} className="control-button stop-analysis">
                  ⏹️ Stop Analysis
                </button>
              )}
            </div>

            {cameraActive && (
              <div className="camera-status">
                <span className="status-indicator active"></span>
                Camera Active - Position yourself in the frame
              </div>
            )}
          </div>
        </div>

        <div className="video-section">
          <div className="video-card">
            <h4>Live Camera Feed</h4>
            <div className="video-container">
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                muted
                className="camera-feed"
              />
              <canvas 
                ref={canvasRef} 
                width={640} 
                height={480} 
                className="annotated-feed"
              />
              {!cameraActive && (
                <div className="camera-placeholder">
                  <div className="placeholder-content">
                    <span className="placeholder-icon">📷</span>
                    <p>Camera not active</p>
                    <p>Click "Start Camera" to begin</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {analysis && (
          <div className="live-feedback-section">
            <div className="feedback-card">
              <h4>Real-time Feedback</h4>
              
              <div className="live-feedback-content">
                <div className="live-score">
                  <div className="score-header">
                    <span>Current Score</span>
                    <div 
                      className="score-badge"
                      style={{ backgroundColor: getScoreColor(analysis.score) }}
                    >
                      {analysis.score}/100
                    </div>
                  </div>
                  <p className="pose-name">{analysis.pose_name}</p>
                </div>

                <div className="live-feedback-list">
                  <h5>Feedback & Corrections:</h5>
                  {analysis.feedback && analysis.feedback.length > 0 ? (
                    <div className="feedback-items">
                      {analysis.feedback.map((item, index) => (
                        <div key={index} className="live-feedback-item">
                          <span className="feedback-icon">💡</span>
                          <span className="feedback-text">{item}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="perfect-live-pose">
                      <span className="perfect-icon">🎉</span>
                      <p>Perfect form! Maintain this pose.</p>
                    </div>
                  )}
                </div>

                {analysis.detailed_metrics && (
                  <div className="live-metrics">
                    <h5>Detailed Metrics:</h5>
                    <div className="metrics-grid">
                      <div className="metric-item">
                        <span>Alignment</span>
                        <span>{analysis.detailed_metrics.alignment_score}/100</span>
                      </div>
                      <div className="metric-item">
                        <span>Balance</span>
                        <span>{analysis.detailed_metrics.balance_score}/100</span>
                      </div>
                      <div className="metric-item">
                        <span>Form</span>
                        <span>{analysis.detailed_metrics.form_score}/100</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveAnalysis;