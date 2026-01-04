import React, { useState } from 'react';
import './PhotoAnalysis.css';

const PhotoAnalysis = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [poseType, setPoseType] = useState('mountain');
  const [supportedPoses, setSupportedPoses] = useState([]);
  const [error, setError] = useState(null);

  React.useEffect(() => {
    fetch('http://localhost:8000/supported-poses')
      .then(response => {
        if (!response.ok) throw new Error('Failed to fetch poses');
        return response.json();
      })
      .then(data => setSupportedPoses(data.poses))
      .catch(error => {
        console.error('Error fetching poses:', error);
        setError('Failed to load pose types. Please check if backend server is running.');
      });
  }, []);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setAnalysis(null);
        setError(null);
      } else {
        setError('Please select an image file (JPEG, PNG, etc.)');
      }
    }
  };

  const analyzePose = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('pose_type', poseType);

    try {
      const response = await fetch(`http://localhost:8000/analyze-photo?pose_type=${poseType}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.status === 'success') {
        setAnalysis(result);
      } else {
        setError(result.message || 'Error analyzing pose');
      }
    } catch (error) {
      console.error('Error analyzing pose:', error);
      setError(`Error analyzing pose: ${error.message}. Make sure backend is running on port 8000.`);
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 90) return '#4CAF50';
    if (score >= 70) return '#FFC107';
    if (score >= 50) return '#FF9800';
    return '#F44336';
  };

  return (
    <div className="photo-analysis">
      <div className="analysis-container">
        <div className="upload-section">
          <div className="upload-card">
            <h3>Upload Your Yoga Pose</h3>
            
            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}
            
            <div className="file-input-container">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="file-input"
                id="pose-photo"
              />
              <label htmlFor="pose-photo" className="file-input-label">
                📁 Choose Photo
              </label>
              {selectedFile && (
                <div className="file-info">
                  Selected: {selectedFile.name}
                </div>
              )}
            </div>

            <div className="pose-selection">
              <label>Select Yoga Pose:</label>
              <select 
                value={poseType} 
                onChange={(e) => setPoseType(e.target.value)}
                className="pose-select"
                disabled={loading}
              >
                {supportedPoses.length > 0 ? (
                  supportedPoses.map(pose => (
                    <option key={pose.id} value={pose.id}>
                      {pose.name}
                    </option>
                  ))
                ) : (
                  <option value="mountain">Mountain Pose</option>
                )}
              </select>
            </div>

            {previewUrl && (
              <div className="preview-section">
                <div className="preview-header">
                  <h4>Preview</h4>
                </div>
                <img src={previewUrl} alt="Pose preview" className="preview-image" />
                <button 
                  onClick={analyzePose} 
                  disabled={loading}
                  className="analyze-button"
                >
                  {loading ? (
                    <>
                      <div className="spinner"></div>
                      Analyzing...
                    </>
                  ) : (
                    '🔍 Analyze Pose'
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        {analysis && (
          <div className="results-section">
            <div className="results-card">
              <h3>Analysis Results</h3>
              
              <div className="results-content">
                {analysis.annotated_image && (
                  <div className="annotated-image-section">
                    <h4>Annotated Pose</h4>
                    <div className="image-container">
                      <img 
                        src={`data:image/jpeg;base64,${analysis.annotated_image}`} 
                        alt="Annotated pose analysis"
                        className="annotated-image"
                      />
                    </div>
                  </div>
                )}

                {analysis.analysis && (
                  <div className="feedback-section">
                    <div className="score-display">
                      <h4>Pose Score</h4>
                      <div 
                        className="score-circle"
                        style={{ 
                          background: `conic-gradient(${getScoreColor(analysis.analysis.score)} ${analysis.analysis.score * 3.6}deg, #eee 0deg)` 
                        }}
                      >
                        <div className="score-inner">
                          <span className="score-value">{analysis.analysis.score}</span>
                          <span className="score-max">/100</span>
                        </div>
                      </div>
                      <p className="pose-name">{analysis.analysis.pose_name}</p>
                    </div>
                    
                    <div className="feedback-list">
                      <h4>Feedback & Corrections</h4>
                      {analysis.analysis.feedback && analysis.analysis.feedback.length > 0 ? (
                        <ul>
                          {analysis.analysis.feedback.map((item, index) => (
                            <li key={index} className="feedback-item">
                              <span className="feedback-icon">💡</span>
                              {item}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <div className="perfect-pose">
                          <span className="perfect-icon">🎉</span>
                          <p>Perfect pose! Excellent alignment and form!</p>
                        </div>
                      )}
                    </div>

                    {analysis.analysis.detailed_metrics && (
                      <div className="metrics-section">
                        <h4>Detailed Metrics</h4>
                        <div className="metrics-grid">
                          <div className="metric-item">
                            <span className="metric-label">Alignment Score</span>
                            <span className="metric-value">
                              {analysis.analysis.detailed_metrics.alignment_score}/100
                            </span>
                          </div>
                          <div className="metric-item">
                            <span className="metric-label">Balance Score</span>
                            <span className="metric-value">
                              {analysis.analysis.detailed_metrics.balance_score}/100
                            </span>
                          </div>
                          <div className="metric-item">
                            <span className="metric-label">Form Score</span>
                            <span className="metric-value">
                              {analysis.analysis.detailed_metrics.form_score}/100
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
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

export default PhotoAnalysis;