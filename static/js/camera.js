// static/js/camera.js - ENHANCED VERSION WITH ANTI-SPOOFING LIVENESS DETECTION
class BeautifulCamera {
  constructor() {
    this.isMarking = false;
    this.lastRecognizedStudent = null;
    this.lastRecognitionTime = 0;
    this.RECOGNITION_COOLDOWN = 3000;
    this.markInterval = null;
    
    // Liveness detection properties
    this.livenessEnabled = true; // Enable liveness detection by default
    this.livenessCheckInterval = null;
    this.currentLivenessData = null;
    this.livenessUI = null;
    this.livenessCheckPassed = false;
    this.REQUIRED_LIVENESS_SCORE = 80; // Increased for better security
    this.REQUIRED_REAL_PERSON_SCORE = 70;
    this.MIN_BLINKS_REQUIRED = 2; // Require at least 2 blinks
    this.REQUIRE_HEAD_MOVEMENT = true;
    this.REQUIRE_MOUTH_MOVEMENT = true;
    this.spoofingAlertShown = false;
    this.consecutiveFailedLivenessChecks = 0;
    this.MAX_FAILED_CHECKS = 3;
  }

  // Initialize camera with beautiful UI
  async initCamera(videoId = 'video') {
    try {
      const video = document.getElementById(videoId);
      if (!video) {
        this.showNotification('Camera element not found', 'error');
        return false;
      }

      // Add loading state
      video.innerHTML = `
        <div class="flex items-center justify-center h-full">
          <div class="text-center">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-3"></div>
            <p class="text-gray-600">Initializing camera with anti-spoofing...</p>
          </div>
        </div>
      `;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });

      video.srcObject = stream;
      await video.play();

      // Initialize liveness detection if enabled
      if (this.livenessEnabled) {
        await this.initializeLivenessDetection(video);
      }

      // Add success animation
      video.style.border = '3px solid #10b981';
      setTimeout(() => {
        video.style.border = '3px solid #e5e7eb';
      }, 2000);

      return true;
    } catch (error) {
      console.error('Camera error:', error);
      this.showNotification('Cannot access camera. Please check permissions.', 'error');
      return false;
    }
  }

  // Initialize liveness detection
  async initializeLivenessDetection(videoElement) {
    try {
      if (typeof livenessDetector === 'undefined') {
        // Load liveness detection script
        await this.loadLivenessScript();
      }

      const initialized = await window.livenessDetector.initialize();
      if (initialized) {
        this.createLivenessUI();
        this.showNotification('🎯 Anti-spoofing liveness detection activated', 'success');
      } else {
        this.showNotification('⚠️ Liveness detection unavailable', 'warning');
        this.livenessEnabled = false;
      }
    } catch (error) {
      console.error('Failed to initialize liveness detection:', error);
      this.livenessEnabled = false;
    }
  }

  loadLivenessScript() {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = '/static/js/liveness.js';
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  createLivenessUI() {
    // Create liveness status overlay
    this.livenessUI = document.createElement('div');
    this.livenessUI.id = 'livenessOverlay';
    this.livenessUI.className = 'liveness-overlay';
    this.livenessUI.innerHTML = `
      <div class="liveness-container">
        <h4 class="liveness-title">
          <i class="fas fa-shield-alt mr-2"></i>Anti-Spoofing Verification
        </h4>
        <div class="liveness-stats">
          <div class="liveness-stat">
            <span class="liveness-label">Status:</span>
            <span id="livenessStatus" class="liveness-value pending">Checking...</span>
          </div>
          <div class="liveness-stat">
            <span class="liveness-label">Blinks:</span>
            <span id="livenessBlinks" class="liveness-value">0/2</span>
          </div>
          <div class="liveness-stat">
            <span class="liveness-label">Head Movement:</span>
            <span id="livenessMovement" class="liveness-value">Checking</span>
          </div>
          <div class="liveness-stat">
            <span class="liveness-label">Anti-Spoofing:</span>
            <span id="livenessAntiSpoofing" class="liveness-value">Checking</span>
          </div>
          <div class="liveness-stat">
            <span class="liveness-label">Real Person Score:</span>
            <span id="livenessRealPerson" class="liveness-value">0%</span>
          </div>
          <div class="liveness-progress">
            <div class="progress-bar-container">
              <div id="livenessProgressBar" class="progress-bar-fill" style="width: 0%"></div>
            </div>
          </div>
          <div class="liveness-instructions">
            <p><i class="fas fa-lightbulb"></i> Please blink naturally, move your head, and speak briefly</p>
            <p class="text-xs mt-1"><i class="fas fa-exclamation-triangle"></i> Photos and screens are automatically rejected</p>
          </div>
        </div>
      </div>
    `;

    // Add to camera container
    const cameraContainer = document.querySelector('.camera-wrapper');
    if (cameraContainer) {
      cameraContainer.appendChild(this.livenessUI);
    }
  }

  updateLivenessUI(data) {
    if (!this.livenessUI) return;

    const statusElement = document.getElementById('livenessStatus');
    const blinksElement = document.getElementById('livenessBlinks');
    const movementElement = document.getElementById('livenessMovement');
    const antiSpoofingElement = document.getElementById('livenessAntiSpoofing');
    const realPersonElement = document.getElementById('livenessRealPerson');
    const progressBar = document.getElementById('livenessProgressBar');

    if (statusElement) {
      let statusText = data.status;
      let statusClass = data.status;
      
      // Convert status to readable text
      switch(data.status) {
        case 'photo_detected':
          statusText = 'Photo Detected ❌';
          statusClass = 'photo_detected';
          break;
        case 'screen_detected':
          statusText = 'Screen Detected ❌';
          statusClass = 'screen_detected';
          break;
        case 'need_blinks':
          statusText = 'Blink Required 👁️';
          statusClass = 'need_action';
          break;
        case 'need_movement':
          statusText = 'Move Head ↔️';
          statusClass = 'need_action';
          break;
        case 'suspicious':
          statusText = 'Suspicious Activity ⚠️';
          statusClass = 'suspicious';
          break;
        case 'active':
          statusText = 'Verified ✅';
          statusClass = 'active';
          break;
        case 'inactive':
          statusText = 'Verifying...';
          statusClass = 'inactive';
          break;
      }
      
      statusElement.textContent = statusText;
      statusElement.className = `liveness-value ${statusClass}`;
    }

    if (blinksElement) {
      blinksElement.textContent = `${data.eyeBlinkCount}/${this.MIN_BLINKS_REQUIRED}`;
      blinksElement.className = `liveness-value ${data.eyeBlinkCount >= this.MIN_BLINKS_REQUIRED ? 'good' : 'bad'}`;
    }

    if (movementElement) {
      movementElement.textContent = data.headMovementDetected ? 'Detected ✅' : 'Required ↔️';
      movementElement.className = `liveness-value ${data.headMovementDetected ? 'good' : 'bad'}`;
    }

    if (antiSpoofingElement) {
      const score = data.antiSpoofingScore;
      antiSpoofingElement.textContent = `${Math.round(score)}%`;
      antiSpoofingElement.className = `liveness-value ${score >= 70 ? 'good' : score >= 50 ? 'warning' : 'bad'}`;
    }

    if (realPersonElement) {
      realPersonElement.textContent = `${Math.round(data.realPersonScore)}%`;
      realPersonElement.className = `liveness-value ${data.realPersonScore >= this.REQUIRED_REAL_PERSON_SCORE ? 'good' : 'bad'}`;
    }

    if (progressBar) {
      progressBar.style.width = `${data.livenessScore}%`;
      
      // Color coding based on score
      if (data.livenessScore >= 80) {
        progressBar.style.background = 'linear-gradient(90deg, #10b981, #059669)';
      } else if (data.livenessScore >= 60) {
        progressBar.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
      } else {
        progressBar.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
      }
    }

    this.currentLivenessData = data;
    
    // Check if liveness requirements are met
    this.livenessCheckPassed = data.isActive && 
                              !data.spoofingDetected &&
                              data.realPersonScore >= this.REQUIRED_REAL_PERSON_SCORE;
    
    // Handle spoofing detection
    if (data.spoofingDetected && !this.spoofingAlertShown) {
      this.handleSpoofingDetection(data);
    }
  }

  handleSpoofingDetection(data) {
    this.spoofingAlertShown = true;
    
    let spoofingType = 'Photo';
    if (data.screenSpoofingDetected) spoofingType = 'Screen';
    
    this.showNotification(`⚠️ ${spoofingType} spoofing detected! Please use a real person.`, 'error', 5000);
    
    // Add to log
    this.addToLog(`❌ ${spoofingType} spoofing detected - Attendance blocked`, 'error', 'fas fa-ban', 'red');
    
    // Reset after 10 seconds
    setTimeout(() => {
      this.spoofingAlertShown = false;
    }, 10000);
  }

  startLivenessChecking(videoElement) {
    if (!this.livenessEnabled || !window.livenessDetector) return;

    // Reset liveness detector
    window.livenessDetector.reset();
    this.consecutiveFailedLivenessChecks = 0;

    this.livenessCheckInterval = setInterval(async () => {
      try {
        const livenessData = await window.livenessDetector.detectLiveness(videoElement);
        this.updateLivenessUI(livenessData);
        
        // Track consecutive failed checks
        if (livenessData.isActive) {
          this.consecutiveFailedLivenessChecks = 0;
        } else {
          this.consecutiveFailedLivenessChecks++;
          
          // Show instructional messages
          if (this.consecutiveFailedLivenessChecks >= this.MAX_FAILED_CHECKS) {
            this.showInstructionalMessage(livenessData);
          }
        }
      } catch (error) {
        console.error('Liveness check error:', error);
        this.consecutiveFailedLivenessChecks++;
      }
    }, 500); // Check every 500ms
  }

  showInstructionalMessage(livenessData) {
    let message = '';
    
    switch(livenessData.status) {
      case 'need_blinks':
        message = 'Please blink naturally 2-3 times';
        break;
      case 'need_movement':
        message = 'Please move your head slightly side to side';
        break;
      case 'photo_detected':
        message = 'Photo detected. Please use a real person';
        break;
      case 'screen_detected':
        message = 'Screen detected. Please use a real person';
        break;
      default:
        message = 'Please blink and move your head for verification';
    }
    
    if (message) {
      this.showNotification(`💡 ${message}`, 'info', 3000);
    }
  }

  stopLivenessChecking() {
    if (this.livenessCheckInterval) {
      clearInterval(this.livenessCheckInterval);
      this.livenessCheckInterval = null;
    }
  }

  // Enhanced notification system - FIXED Z-INDEX
  showNotification(message, type = 'info', duration = 4000) {
    // Remove existing notifications
    document.querySelectorAll('.beautiful-notification').forEach(notif => notif.remove());

    const notification = document.createElement('div');
    notification.className = `beautiful-notification ${type} show`;
    
    // Ensure proper z-index and positioning
    notification.style.zIndex = '9999';
    notification.style.position = 'fixed';
    notification.style.top = '90px'; // Below navbar
    notification.style.right = '20px';
    
    const icons = {
      success: 'fas fa-check-circle',
      error: 'fas fa-exclamation-circle',
      warning: 'fas fa-exclamation-triangle',
      info: 'fas fa-info-circle'
    };

    notification.innerHTML = `
      <div class="flex items-center">
        <i class="${icons[type]} mr-3 text-lg"></i>
        <span>${message}</span>
      </div>
      <button onclick="this.parentElement.remove()" class="ml-4 text-white opacity-70 hover:opacity-100">
        <i class="fas fa-times"></i>
      </button>
    `;

    document.body.appendChild(notification);

    // Auto remove after duration
    setTimeout(() => {
      if (notification.parentNode) {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => notification.remove(), 300);
      }
    }, duration);
  }

  // Enhanced attendance marking with anti-spoofing
  startMarking(session_id, subject_id) {
    if (this.isMarking) {
      this.showNotification('Already marking attendance', 'info');
      return;
    }

    this.isMarking = true;
    this.showNotification('🎯 Starting attendance marking with anti-spoofing verification...', 'info');

    // Update UI
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;

    // Start liveness checking
    const video = document.getElementById('video');
    if (video && this.livenessEnabled) {
      this.startLivenessChecking(video);
    }

    this.markInterval = setInterval(async () => {
      try {
        // Check liveness before proceeding
        if (this.livenessEnabled) {
          if (!this.livenessCheckPassed) {
            this.addToLog('❌ Liveness verification failed. Please blink and move your head.', 'error', 'fas fa-exclamation-triangle', 'red');
            return;
          }
          
          if (this.currentLivenessData && this.currentLivenessData.spoofingDetected) {
            this.addToLog('❌ Spoofing detected! Attendance blocked.', 'error', 'fas fa-ban', 'red');
            return;
          }
          
          if (this.currentLivenessData && this.currentLivenessData.realPersonScore < this.REQUIRED_REAL_PERSON_SCORE) {
            this.addToLog('❌ Real person verification failed. Please ensure you are a real person.', 'error', 'fas fa-user-slash', 'red');
            return;
          }
        }

        const video = document.getElementById('video');
        const dataUrl = this.getImageDataUrl(video);
        
        const resp = await fetch(`/prof/${subject_id}/recognize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            image: dataUrl, 
            session_id: session_id,
            liveness_data: this.currentLivenessData // Send liveness data for server verification
          })
        });

        if (!resp.ok) throw new Error('Network error');

        const data = await resp.json();
        if (data && data.results) {
          data.results.forEach(result => {
            this.processRecognitionResult(result);
          });
        }
      } catch (error) {
        console.error('Recognition error:', error);
        this.showNotification('Recognition error occurred', 'error');
      }
    }, 2000);
  }

  processRecognitionResult(result) {
    const now = Date.now();
    
    // Skip if same student was recognized recently
    if (result.roll === this.lastRecognizedStudent && 
        (now - this.lastRecognitionTime) < this.RECOGNITION_COOLDOWN) {
      return;
    }

    let message, type, icon, color;

    if (result.warning === 'multiple_faces') {
      message = "Multiple faces detected! Please ensure only one student is in frame.";
      type = 'warning';
      icon = 'fas fa-users';
      color = 'yellow';
    } else if (result.status === 'marked') {
      message = `${result.name} (${result.roll}) - Attendance marked with anti-spoofing ✅`;
      type = 'success';
      icon = 'fas fa-check-circle';
      color = 'green';
      this.lastRecognizedStudent = result.roll;
      this.lastRecognitionTime = now;
    } else if (result.status === 'already_marked_today' || result.status === 'already_marked') {
      message = `${result.name} - Already marked today`;
      type = 'info';
      icon = 'fas fa-info-circle';
      color = 'blue';
      this.lastRecognizedStudent = result.roll;
      this.lastRecognitionTime = now;
    } else if (result.status === 'unknown') {
      message = `Unknown face - Not in database`;
      type = 'error';
      icon = 'fas fa-question-circle';
      color = 'red';
      this.lastRecognizedStudent = null;
    } else if (result.status === 'low_confidence') {
      message = `${result.name} - Low confidence match (${(result.confidence * 100).toFixed(1)}%)`;
      type = 'warning';
      icon = 'fas fa-exclamation-triangle';
      color = 'orange';
      this.lastRecognizedStudent = null;
    } else if (result.status === 'no_face') {
      message = `No face detected - Please position face in frame`;
      type = 'error';
      icon = 'fas fa-user-slash';
      color = 'red';
      this.lastRecognizedStudent = null;
    } else if (result.status === 'spoofing_detected') {
      message = `Spoofing detected! Attendance blocked.`;
      type = 'error';
      icon = 'fas fa-ban';
      color = 'red';
      this.lastRecognizedStudent = null;
    } else {
      message = `Recognition result: ${JSON.stringify(result)}`;
      type = 'info';
      icon = 'fas fa-info-circle';
      color = 'gray';
      this.lastRecognizedStudent = null;
    }

    this.addToLog(message, type, icon, color);
    
    // Show notification only for important events
    if (type !== 'info' || result.status === 'marked') {
      this.showNotification(message, type);
    }
  }

  addToLog(message, type, icon, color) {
    const log = document.getElementById('log');
    if (!log) return;

    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type} animate-fade-in`;
    logEntry.innerHTML = `
      <div class="flex items-center p-3 rounded-lg border-l-4 bg-${color}-50 border-${color}-500">
        <i class="${icon} text-${color}-500 mr-3 text-lg"></i>
        <div class="flex-1">
          <p class="text-sm font-medium text-gray-800">${message}</p>
          <p class="text-xs text-gray-500">${new Date().toLocaleTimeString()}</p>
        </div>
      </div>
    `;

    log.prepend(logEntry);

    // Limit log entries
    const entries = log.querySelectorAll('.log-entry');
    if (entries.length > 10) {
      entries[entries.length - 1].remove();
    }

    // Add animation
    setTimeout(() => {
      logEntry.style.opacity = '1';
      logEntry.style.transform = 'translateY(0)';
    }, 10);
  }

  stopMarking() {
    if (this.markInterval) {
      clearInterval(this.markInterval);
      this.markInterval = null;
      this.isMarking = false;
      this.lastRecognizedStudent = null;
      this.lastRecognitionTime = 0;
      this.spoofingAlertShown = false;
      this.consecutiveFailedLivenessChecks = 0;

      // Stop liveness checking
      this.stopLivenessChecking();

      // Update UI
      const startBtn = document.getElementById('startBtn');
      const stopBtn = document.getElementById('stopBtn');
      if (startBtn) startBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = true;

      this.showNotification('Attendance marking stopped', 'info');
    }
  }

  getImageDataUrl(video) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
  }
}

// Initialize global camera instance
window.beautifulCamera = new BeautifulCamera();

// Legacy function compatibility
async function initCamera(videoId = 'video') {
  return window.beautifulCamera.initCamera(videoId);
}

function startMarking(session_id, subject_id) {
  window.beautifulCamera.startMarking(session_id, subject_id);
}

function stopMarking() {
  window.beautifulCamera.stopMarking();
}

function showNotification(message, type = 'info') {
  window.beautifulCamera.showNotification(message, type);
}

function getImageDataUrl(video) {
  return window.beautifulCamera.getImageDataUrl(video);
}