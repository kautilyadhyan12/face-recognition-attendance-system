
class LivenessDetector {
    constructor() {
        this.faceLandmarks = null;
        this.eyeAspectRatio = 0;
        this.blinkCount = 0;
        this.headMovement = { x: 0, y: 0, z: 0 };
        this.livenessStatus = 'pending';
        this.eyeClosedFrames = 0;
        this.eyeOpenFrames = 0;
        this.blinkDetected = false;
        this.headMovementDetected = false;
        this.mouthOpenFrames = 0;
        this.mouthClosedFrames = 0;
        this.mouthOpenDetected = false;
        this.antiSpoofingScore = 0;
        this.livenessScore = 0;
        this.isLivenessActive = false;
        this.frameHistory = [];
        this.maxHistorySize = 10;
        this.photoSpoofingDetected = false;
        this.screenSpoofingDetected = false;
        this.realPersonScore = 0;
        
        // Constants
        this.EYE_AR_THRESHOLD = 0.25;
        this.EYE_AR_CONSEC_FRAMES = 3;
        this.MOUTH_AR_THRESHOLD = 0.35;
        this.MOUTH_AR_CONSEC_FRAMES = 5;
        this.MIN_BLINKS_REQUIRED = 2;
        this.MIN_MOUTH_MOVEMENTS = 1;
        this.MIN_LIVENESS_SCORE = 80; 
        this.ANTI_SPOOFING_THRESHOLD = 70;
        this.HEAD_MOVEMENT_THRESHOLD = 5;
        this.PHOTO_SPOOFING_THRESHOLD = 0.1; 
    }

    async initialize() {
        try {
            // Load TensorFlow.js and face-api.js from CDN
            await this.loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.11.0/dist/tf.min.js');
            await this.loadScript('https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js');
            
            // Load models
            await faceapi.nets.tinyFaceDetector.loadFromUri('/static/models');
            await faceapi.nets.faceLandmark68Net.loadFromUri('/static/models');
            await faceapi.nets.faceExpressionNet.loadFromUri('/static/models');
            
            console.log(' Face detection models loaded for anti-spoofing');
            return true;
        } catch (error) {
            console.error(' Failed to load face detection models:', error);
            return false;
        }
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    async detectLiveness(videoElement) {
        if (!videoElement || !faceapi) {
            return {
                status: 'pending',
                eyeBlinkCount: 0,
                mouthMovements: 0,
                headMovementDetected: false,
                antiSpoofingScore: 0,
                livenessScore: 0,
                isActive: false,
                spoofingDetected: false,
                realPersonScore: 0
            };
        }

        try {
            const detections = await faceapi.detectAllFaces(
                videoElement, 
                new faceapi.TinyFaceDetectorOptions()
            ).withFaceLandmarks().withFaceExpressions();

            if (detections.length === 0) {
                this.livenessStatus = 'no_face';
                return this.getLivenessData();
            }

            if (detections.length > 1) {
                this.livenessStatus = 'multiple_faces';
                return this.getLivenessData();
            }

            const detection = detections[0];
            this.faceLandmarks = detection.landmarks;

            // Calculate Eye Aspect Ratio (EAR)
            this.calculateEyeAspectRatio();

            // Calculate Mouth Aspect Ratio (MAR)
            this.calculateMouthAspectRatio();

            // Detect blinks
            this.detectBlinks();

            // Detect mouth movements (speaking/opening)
            this.detectMouthMovements();

            // Detect head movement
            this.detectHeadMovement();

            // Detect photo/screen spoofing
            this.detectSpoofing();

            // Calculate anti-spoofing score
            this.calculateAntiSpoofingScore();

            // Calculate liveness score
            this.calculateLivenessScore();

            return this.getLivenessData();

        } catch (error) {
            console.error('Liveness detection error:', error);
            return {
                status: 'error',
                eyeBlinkCount: 0,
                mouthMovements: 0,
                headMovementDetected: false,
                antiSpoofingScore: 0,
                livenessScore: 0,
                isActive: false,
                spoofingDetected: true,
                realPersonScore: 0
            };
        }
    }

    calculateEyeAspectRatio() {
        if (!this.faceLandmarks) return;

        // Get eye landmarks
        const leftEye = this.faceLandmarks.getLeftEye();
        const rightEye = this.faceLandmarks.getRightEye();

        // Calculate EAR for both eyes
        const leftEAR = this.getEAR(leftEye);
        const rightEAR = this.getEAR(rightEye);
        
        this.eyeAspectRatio = (leftEAR + rightEAR) / 2;
        
        // Detect eye closure
        if (this.eyeAspectRatio < this.EYE_AR_THRESHOLD) {
            this.eyeClosedFrames++;
            this.eyeOpenFrames = 0;
        } else {
            this.eyeClosedFrames = 0;
            this.eyeOpenFrames++;
        }
    }

    calculateMouthAspectRatio() {
        if (!this.faceLandmarks) return;

        const mouth = this.faceLandmarks.getMouth();
        
        // Calculate vertical distances
        const A = this.distance(mouth[13], mouth[19]); 
        const B = this.distance(mouth[14], mouth[18]);
        const C = this.distance(mouth[15], mouth[17]);
        
        // Calculate horizontal distance
        const D = this.distance(mouth[12], mouth[16]);
        
        // Mouth Aspect Ratio
        const mouthAR = (A + B + C) / (3 * D);
        
        // Detect mouth opening
        if (mouthAR > this.MOUTH_AR_THRESHOLD) {
            this.mouthOpenFrames++;
            this.mouthClosedFrames = 0;
        } else {
            this.mouthOpenFrames = 0;
            this.mouthClosedFrames++;
        }
    }

    getEAR(eyePoints) {
        // Calculate vertical distances
        const A = this.distance(eyePoints[1], eyePoints[5]);
        const B = this.distance(eyePoints[2], eyePoints[4]);
        
        // Calculate horizontal distance
        const C = this.distance(eyePoints[0], eyePoints[3]);
        
        // Eye Aspect Ratio
        return (A + B) / (2.0 * C);
    }

    distance(point1, point2) {
        return Math.sqrt(
            Math.pow(point2.x - point1.x, 2) + 
            Math.pow(point2.y - point1.y, 2)
        );
    }

    detectBlinks() {
        if (this.eyeClosedFrames >= this.EYE_AR_CONSEC_FRAMES && !this.blinkDetected) {
            this.blinkCount++;
            this.blinkDetected = true;
            console.log(` Blink detected! Total: ${this.blinkCount}`);
        }
        
        if (this.eyeOpenFrames >= this.EYE_AR_CONSEC_FRAMES) {
            this.blinkDetected = false;
        }
    }

    detectMouthMovements() {
        if (this.mouthOpenFrames >= this.MOUTH_AR_CONSEC_FRAMES && !this.mouthOpenDetected) {
            this.mouthOpenDetected = true;
            console.log(' Mouth movement detected!');
        }
        
        if (this.mouthClosedFrames >= this.MOUTH_AR_CONSEC_FRAMES) {
            this.mouthOpenDetected = false;
        }
    }

    detectHeadMovement() {
        if (!this.faceLandmarks) return;
        
        // Get nose position
        const nose = this.faceLandmarks.getNose();
        if (nose && nose.length > 0) {
            const nosePoint = nose[0];
            
            // Store current position
            this.frameHistory.push({
                x: nosePoint.x,
                y: nosePoint.y,
                timestamp: Date.now()
            });
            
            // Keep only recent history
            if (this.frameHistory.length > this.maxHistorySize) {
                this.frameHistory.shift();
            }
            
            // Calculate movement variation
            if (this.frameHistory.length >= 3) {
                let totalMovement = 0;
                let movements = 0;
                
                for (let i = 1; i < this.frameHistory.length; i++) {
                    const prev = this.frameHistory[i-1];
                    const curr = this.frameHistory[i];
                    
                    const movementX = Math.abs(curr.x - prev.x);
                    const movementY = Math.abs(curr.y - prev.y);
                    totalMovement += movementX + movementY;
                    movements++;
                    
                    if (movementX > this.HEAD_MOVEMENT_THRESHOLD || movementY > this.HEAD_MOVEMENT_THRESHOLD) {
                        this.headMovementDetected = true;
                    }
                }
                
                const avgMovement = totalMovement / movements;
                
                // Detect photo spoofing 
                if (avgMovement < this.PHOTO_SPOOFING_THRESHOLD && this.frameHistory.length >= 5) {
                    this.photoSpoofingDetected = true;
                    console.log('Photo spoofing detected (very low movement)');
                }
            }
        }
    }

    detectSpoofing() {
        // Check for screen reflection patterns (simplified)
        // Real faces have more variation in facial expressions
        const expressionVariation = this.calculateExpressionVariation();
        
        // Screen spoofing detection based on perfect stillness
        if (this.frameHistory.length >= 8) {
            const recentMovements = this.calculateMovementVariation();
            if (recentMovements < 0.05) { // Extremely low movement variation
                this.screenSpoofingDetected = true;
                console.log('⚠️ Possible screen spoofing detected');
            }
        }
    }

    calculateExpressionVariation() {
        
        
        return Math.random() * 0.5 + 0.5; 
    }

    calculateMovementVariation() {
        if (this.frameHistory.length < 2) return 0;
        
        let totalVariation = 0;
        let count = 0;
        
        for (let i = 1; i < this.frameHistory.length; i++) {
            const prev = this.frameHistory[i-1];
            const curr = this.frameHistory[i];
            
            const variation = Math.sqrt(
                Math.pow(curr.x - prev.x, 2) + 
                Math.pow(curr.y - prev.y, 2)
            );
            totalVariation += variation;
            count++;
        }
        
        return count > 0 ? totalVariation / count : 0;
    }

    calculateAntiSpoofingScore() {
        let score = 100; 
        
        // Deduct points for spoofing indicators
        if (this.photoSpoofingDetected) score -= 50;
        if (this.screenSpoofingDetected) score -= 50;
        
        // Deduct points for lack of movement
        const movementVariation = this.calculateMovementVariation();
        if (movementVariation < 0.1) score -= 30;
        if (movementVariation < 0.05) score -= 20;
        
        this.antiSpoofingScore = Math.max(0, Math.min(100, score));
    }

    calculateLivenessScore() {
        let score = 0;
        
        // Blink detection (40 points max)
        const blinkScore = Math.min(this.blinkCount * 20, 40);
        score += blinkScore;
        
        // Mouth movement (20 points max)
        if (this.mouthOpenDetected) score += 20;
        
        // Head movement (20 points max)
        if (this.headMovementDetected) score += 20;
        
        // Anti-spoofing score (20 points max)
        const antiSpoofingPoints = this.antiSpoofingScore * 0.2;
        score += antiSpoofingPoints;
        
        this.livenessScore = Math.min(100, Math.max(0, score));
        this.realPersonScore = this.calculateRealPersonScore();
        
        // Require multiple factors for true liveness
        const hasRequiredBlinks = this.blinkCount >= this.MIN_BLINKS_REQUIRED;
        const hasHeadMovement = this.headMovementDetected;
        const hasGoodAntiSpoofing = this.antiSpoofingScore >= this.ANTI_SPOOFING_THRESHOLD;
        
        this.isLivenessActive = this.livenessScore >= this.MIN_LIVENESS_SCORE &&
                              hasRequiredBlinks &&
                              hasHeadMovement &&
                              hasGoodAntiSpoofing &&
                              !this.photoSpoofingDetected &&
                              !this.screenSpoofingDetected;
        
        this.livenessStatus = this.isLivenessActive ? 'active' : this.getDetailedStatus();
    }

    calculateRealPersonScore() {
        // Calculate a score that indicates real person vs photo
        let score = 0;
        
        // Multiple blinks indicate real person
        if (this.blinkCount >= 2) score += 40;
        if (this.blinkCount >= 3) score += 10;
        
        // Head movement
        if (this.headMovementDetected) score += 30;
        
        // Mouth movement
        if (this.mouthOpenDetected) score += 20;
        
        // Anti-spoofing confidence
        score += this.antiSpoofingScore * 0.1;
        
        return Math.min(100, score);
    }

    getDetailedStatus() {
        if (this.photoSpoofingDetected) return 'photo_detected';
        if (this.screenSpoofingDetected) return 'screen_detected';
        if (this.blinkCount < this.MIN_BLINKS_REQUIRED) return 'need_blinks';
        if (!this.headMovementDetected) return 'need_movement';
        if (this.antiSpoofingScore < this.ANTI_SPOOFING_THRESHOLD) return 'suspicious';
        return 'inactive';
    }

    getLivenessData() {
        return {
            status: this.livenessStatus,
            eyeBlinkCount: this.blinkCount,
            mouthMovements: this.mouthOpenDetected ? 1 : 0,
            headMovementDetected: this.headMovementDetected,
            antiSpoofingScore: this.antiSpoofingScore,
            livenessScore: this.livenessScore,
            isActive: this.isLivenessActive,
            blinkDetected: this.blinkDetected,
            mouthOpenDetected: this.mouthOpenDetected,
            spoofingDetected: this.photoSpoofingDetected || this.screenSpoofingDetected,
            realPersonScore: this.realPersonScore,
            photoSpoofingDetected: this.photoSpoofingDetected,
            screenSpoofingDetected: this.screenSpoofingDetected
        };
    }

    reset() {
        this.blinkCount = 0;
        this.livenessScore = 0;
        this.isLivenessActive = false;
        this.blinkDetected = false;
        this.mouthOpenDetected = false;
        this.headMovementDetected = false;
        this.photoSpoofingDetected = false;
        this.screenSpoofingDetected = false;
        this.eyeClosedFrames = 0;
        this.eyeOpenFrames = 0;
        this.mouthOpenFrames = 0;
        this.mouthClosedFrames = 0;
        this.antiSpoofingScore = 0;
        this.realPersonScore = 0;
        this.livenessStatus = 'pending';
        this.frameHistory = [];
    }
}

// Initialize global liveness detector
window.livenessDetector = new LivenessDetector();
