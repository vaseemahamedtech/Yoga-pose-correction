import React, { useState } from 'react';
import './App.css';
import PhotoAnalysis from './components/PhotoAnalysis';
import LiveAnalysis from './components/LiveAnalysis';

function App() {
  const [activeTab, setActiveTab] = useState('photo');

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>🧘 Yoga Pose Correction AI</h1>
          <p>Get instant feedback on your yoga poses with AI analysis</p>
        </div>
      </header>

      <div className="tab-navigation">
        <button 
          className={`tab-button ${activeTab === 'photo' ? 'active' : ''}`}
          onClick={() => setActiveTab('photo')}
        >
          📸 Photo Analysis
        </button>
        <button 
          className={`tab-button ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          📹 Live Analysis
        </button>
      </div>

      <main className="main-content">
        {activeTab === 'photo' && <PhotoAnalysis />}
        {activeTab === 'live' && <LiveAnalysis />}
      </main>

      <footer className="app-footer">
        <p>Powered by MediaPipe Pose Detection & Advanced AI Analysis</p>
      </footer>
    </div>
  );
}

export default App;