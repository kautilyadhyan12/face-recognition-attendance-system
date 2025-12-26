
import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import time

class LivenessDetector:
    def __init__(self):
        # Eye aspect ratio thresholds
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 3
        
        # Mouth aspect ratio thresholds for yawning detection
        self.MAR_THRESH = 0.5
        
        # Head movement thresholds
        self.HEAD_MOVEMENT_THRESH = 10  # pixels
        self.MIN_FRAMES_FOR_HEAD_MOVEMENT = 10
        
        # Face landmarks indices 
        self.FACIAL_LANDMARKS_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17))
        ])
        
        # Initialize counters and trackers
        self.eye_blink_count = 0
        self.yawn_count = 0
        self.consecutive_frames_eye_closed = 0
        self.consecutive_frames_mouth_open = 0
        self.head_movement_detected = False
        self.liveness_score = 0
        self.face_positions = []
        self.last_face_position = None
        self.start_time = None
        self.is_live = False
        
        # Performance metrics
        self.total_frames_processed = 0
        self.liveness_check_started = False
        
    def start_liveness_check(self):
        """Start a new liveness check session"""
        self.reset()
        self.liveness_check_started = True
        self.start_time = time.time()
        
    def reset(self):
        """Reset all counters and states"""
        self.eye_blink_count = 0
        self.yawn_count = 0
        self.consecutive_frames_eye_closed = 0
        self.consecutive_frames_mouth_open = 0
        self.head_movement_detected = False
        self.liveness_score = 0
        self.face_positions = []
        self.last_face_position = None
        self.is_live = False
        self.total_frames_processed = 0
        
    def eye_aspect_ratio(self, eye):
        """Calculate the eye aspect ratio"""
        
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        
        C = dist.euclidean(eye[0], eye[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        """Calculate the mouth aspect ratio for yawning detection"""
       
        # vertical mouth landmarks
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        
       
        # mouth landmark
        D = dist.euclidean(mouth[12], mouth[16])
        
        # the mouth aspect ratio
        mar = (A + B + C) / (3.0 * D)
        return mar
    
    def detect_eye_blink(self, left_eye, right_eye):
        """Detect eye blink based on eye aspect ratio"""
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        
        # Average the eye aspect ratio 
        ear = (left_ear + right_ear) / 2.0
        
        # Check if eye aspect ratio is below the blink threshold
        if ear < self.EYE_AR_THRESH:
            self.consecutive_frames_eye_closed += 1
        else:
            # If eyes were closed long enough, count as a blink
            if self.consecutive_frames_eye_closed >= self.EYE_AR_CONSEC_FRAMES:
                self.eye_blink_count += 1
            
            # Reset the eye frame counter
            self.consecutive_frames_eye_closed = 0
        
        return ear
    
    def detect_yawning(self, mouth):
        """Detect yawning based on mouth aspect ratio"""
        mar = self.mouth_aspect_ratio(mouth)
        
        # Check if mouth aspect ratio is above the yawn threshold
        if mar > self.MAR_THRESH:
            self.consecutive_frames_mouth_open += 1
            
            # If mouth has been open for 10 consecutive frames, count as a yawn
            if self.consecutive_frames_mouth_open >= 10:
                self.yawn_count += 1
                self.consecutive_frames_mouth_open = 0  
        else:
            self.consecutive_frames_mouth_open = 0
        
        return mar
    
    def track_head_movement(self, face_center):
        """Track head movement by monitoring face position changes"""
        if face_center is not None:
            self.face_positions.append(face_center)
            
            # Keep only last 30 positions
            if len(self.face_positions) > 30:
                self.face_positions.pop(0)
            
            # Check for head movement after we have enough frames
            if len(self.face_positions) >= self.MIN_FRAMES_FOR_HEAD_MOVEMENT:
                # Calculate movement variance
                positions_array = np.array(self.face_positions)
                variance = np.var(positions_array, axis=0)
                movement_score = np.mean(variance)
                
                # If movement score exceeds threshold, mark as movement detected
                if movement_score > self.HEAD_MOVEMENT_THRESH and not self.head_movement_detected:
                    self.head_movement_detected = True
                
                return movement_score
        
        return 0
    
    def calculate_liveness_score(self):
        """Calculate overall liveness score"""
        score = 0
        
        # Check for eye blinks 
        score += min(self.eye_blink_count, 2)
        
        # Check for yawning 
        if self.yawn_count > 0:
            score += 1
        
        # Check for head movement 
        if self.head_movement_detected:
            score += 2
        
        # Check if we've processed enough frames 
        if self.total_frames_processed >= 15:
            score += 1
        
        self.liveness_score = score
        return score
    
    def is_live_face(self):
        """Determine if face is live based on liveness score"""
        score = self.calculate_liveness_score()
        
        # Require at least 4 points out of possible 6 to be considered live
        self.is_live = score >= 4
        
        # Additional requirement: must have at least one blink
        if self.eye_blink_count < 1:
            self.is_live = False
        
        return self.is_live
    
    def get_liveness_status(self):
        """Get detailed liveness status"""
        return {
            'is_live': self.is_live,
            'liveness_score': self.liveness_score,
            'eye_blink_count': self.eye_blink_count,
            'yawn_count': self.yawn_count,
            'head_movement_detected': self.head_movement_detected,
            'total_frames_processed': self.total_frames_processed,
            'requires_action': self.get_required_action()
        }
    
    def get_required_action(self):
        """Get what action user needs to perform for liveness verification"""
        requirements = []
        
        if self.eye_blink_count < 1:
            requirements.append("Blink your eyes")
        
        if not self.head_movement_detected:
            requirements.append("Move your head slightly")
        
        if self.total_frames_processed < 15:
            requirements.append("Keep face in frame longer")
        
        if requirements:
            return " | ".join(requirements)
        return "All checks passed"
    
    def draw_liveness_info(self, frame, face_rect=None):
        """Draw liveness information on the frame"""
        height, width = frame.shape[:2]
        
        
        overlay = frame.copy()
        
        # Draw semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (width - 10, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw liveness status with color coding
        status_color = (0, 255, 0) if self.is_live else (0, 0, 255)
        status_text = "LIVE FACE DETECTED" if self.is_live else "VERIFYING LIVENESS..."
        
        cv2.putText(frame, f"Status: {status_text}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw liveness metrics
        cv2.putText(frame, f"Eye Blinks: {self.eye_blink_count}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Head Movement: {'YES' if self.head_movement_detected else 'NO'}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Liveness Score: {self.liveness_score}/6", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Required: {self.get_required_action()}", (20, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        # Draw face rectangle with liveness status color
        if face_rect:
            x, y, w, h = face_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
            
          
            text_y = max(10, y - 10)
            cv2.putText(frame, "LIVE" if self.is_live else "VERIFYING", 
                       (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return frame

# Face landmark detector using dlib 
class FaceLandmarkDetector:
    def __init__(self):
        self.detector = None
        self.predictor = None
        self.initialized = False
        
        try:
            import dlib
           
            self.detector = dlib.get_frontal_face_detector()
            
           
            shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
            
           
            if os.path.exists(shape_predictor_path):
                self.predictor = dlib.shape_predictor(shape_predictor_path)
                self.initialized = True
                print(" Face landmark detector initialized with dlib")
            else:
                print(" Shape predictor file not found. Using simplified detection.")
        except ImportError:
            print(" dlib not installed. Using simplified face detection.")
        
    def detect_landmarks(self, frame, face_rect):
        """Detect facial landmarks for a given face rectangle"""
        if not self.initialized or self.predictor is None:
            return self.simplified_landmarks(face_rect)
        
        try:
            import dlib
            # Convert face rectangle to dlib rectangle
            dlib_rect = dlib.rectangle(face_rect[0], face_rect[1], 
                                      face_rect[0] + face_rect[2], 
                                      face_rect[1] + face_rect[3])
            
            # Detect landmarks
            landmarks = self.predictor(frame, dlib_rect)
            
            # Convert landmarks to numpy array
            landmarks_np = np.zeros((68, 2), dtype=int)
            for i in range(68):
                landmarks_np[i] = (landmarks.part(i).x, landmarks.part(i).y)
            
            return landmarks_np
        except Exception as e:
            print(f"Landmark detection error: {e}")
            return self.simplified_landmarks(face_rect)
    
    def simplified_landmarks(self, face_rect):
        """Generate simplified landmarks based on face rectangle"""
        x, y, w, h = face_rect
        
        # Create approximate landmarks
        landmarks = np.zeros((68, 2), dtype=int)
        
        # Jaw line
        for i in range(17):
            landmarks[i] = (x + int(w * i / 16), y + h)
        
        # Left eyebrow 
        for i in range(5):
            landmarks[17 + i] = (x + int(w * (i + 1) / 6), y + int(h * 0.2))
        
        # Right eyebrow 
        for i in range(5):
            landmarks[22 + i] = (x + int(w * (i + 4) / 6), y + int(h * 0.2))
        
        # Nose 
        nose_points = [(0.35, 0.4), (0.5, 0.3), (0.65, 0.4), 
                      (0.5, 0.6), (0.5, 0.7), (0.5, 0.8),
                      (0.4, 0.9), (0.5, 0.95), (0.6, 0.9)]
        
        for i in range(9):
            landmarks[27 + i] = (x + int(w * nose_points[i][0]), 
                                y + int(h * nose_points[i][1]))
        
        # Left eye 
        eye_points = [(0.3, 0.35), (0.35, 0.3), (0.4, 0.35),
                     (0.35, 0.4), (0.3, 0.4), (0.35, 0.45)]
        
        for i in range(6):
            landmarks[36 + i] = (x + int(w * eye_points[i][0]), 
                                y + int(h * eye_points[i][1]))
        
        # Right eye 
        for i in range(6):
            landmarks[42 + i] = (x + int(w * (eye_points[i][0] + 0.3)), 
                                y + int(h * eye_points[i][1]))
        
        # Mouth 
        mouth_points = [(0.3, 0.7), (0.35, 0.65), (0.4, 0.7), (0.45, 0.65), 
                       (0.5, 0.7), (0.55, 0.65), (0.6, 0.7), (0.65, 0.65),
                       (0.7, 0.7), (0.65, 0.75), (0.6, 0.8), (0.55, 0.75),
                       (0.5, 0.8), (0.45, 0.75), (0.4, 0.8), (0.35, 0.75),
                       (0.3, 0.8), (0.35, 0.85), (0.5, 0.85), (0.65, 0.85)]
        
        for i in range(20):
            landmarks[48 + i] = (x + int(w * mouth_points[i][0]), 
                                y + int(h * mouth_points[i][1]))
        
        return landmarks

# Main Liveness Processor
class LivenessProcessor:
    def __init__(self):
        self.liveness_detector = LivenessDetector()
        self.landmark_detector = FaceLandmarkDetector()
        self.face_cascade = None
        
        # Initialize OpenCV face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def detect_face(self, frame):
        """Detect face in frame using OpenCV cascade"""
        if self.face_cascade is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            return faces[0]  
        return None
    
    def process_frame(self, frame):
        """Process a single frame for liveness detection"""
        self.liveness_detector.total_frames_processed += 1
        
        # Detect face
        face_rect = self.detect_face(frame)
        
        if face_rect is None:
            
            return frame, None, {"error": "No face detected"}
        
        x, y, w, h = face_rect
        face_center = (x + w//2, y + h//2)
        
        # Track head movement
        movement_score = self.liveness_detector.track_head_movement(face_center)
        
        # Detect facial landmarks
        landmarks = self.landmark_detector.dect_landmarks(frame, face_rect)
        
        if landmarks is not None and len(landmarks) >= 68:
            # Extract eye and mouth regions
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            mouth = landmarks[48:68]
            
            # Detect eye blink
            ear = self.liveness_detector.detect_eye_blink(left_eye, right_eye)
            
            # Detect yawning
            mar = self.liveness_detector.detect_yawning(mouth)
            
            # Draw landmarks for visualization
            for (x_lm, y_lm) in landmarks:
                cv2.circle(frame, (x_lm, y_lm), 1, (0, 255, 255), -1)
            
            # Draw eye and mouth outlines
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [mouth], True, (255, 0, 0), 1)
            
            # Display EAR and MAR values
            cv2.putText(frame, f"EAR: {ear:.2f}", (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (x, y + h + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Calculate liveness score
        self.liveness_detector.calculate_liveness_score()
        self.liveness_detector.is_live_face()
        
        # Draw liveness information on frame
        frame = self.liveness_detector.draw_liveness_info(frame, face_rect)
        
        # Get liveness status
        liveness_status = self.liveness_detector.get_liveness_status()
        
        return frame, face_rect, liveness_status
    
    def reset_liveness_check(self):
        """Reset liveness check"""
        self.liveness_detector.reset()
    
    def start_liveness_check(self):
        """Start a new liveness check"""
        self.liveness_detector.start_liveness_check()
    
    def get_liveness_status(self):
        """Get current liveness status"""
        return self.liveness_detector.get_liveness_status()
    
    def is_face_live(self):
        """Check if face is live"""
        return self.liveness_detector.is_live
