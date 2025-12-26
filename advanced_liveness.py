
import cv2
import numpy as np
import time
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import os

class AdvancedLivenessDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Eye landmarks indices
        self.LEFT_EYE_START, self.LEFT_EYE_END = 42, 48
        self.RIGHT_EYE_START, self.RIGHT_EYE_END = 36, 42
        
        # Thresholds
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 3
        self.BLINK_THRESHOLD = 2
        
        # Counters
        self.blink_counter = 0
        self.total_blinks = 0
        self.frame_count = 0
        self.start_time = None
    
    def eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio for blink detection"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def check_liveness(self, frame):
        """Advanced liveness detection using facial landmarks"""
        self.frame_count += 1
        
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed_time = time.time() - self.start_time
        
        # Resize for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (640, new_height))
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        rects = self.detector(gray, 0)
        
        if len(rects) == 0:
            return {
                'live': False,
                'confidence': 0.0,
                'message': 'No face detected',
                'blinks': self.total_blinks,
                'frame_count': self.frame_count,
                'method': 'advanced'
            }
        
        # Get facial landmarks
        shape = self.predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        
        # Extract eye regions
        left_eye = shape[self.LEFT_EYE_START:self.LEFT_EYE_END]
        right_eye = shape[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
        
        # Calculate eye aspect ratios
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Detect blinks
        if ear < self.EYE_AR_THRESH:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.EYE_AR_CONSEC_FRAMES:
                self.total_blinks += 1
            self.blink_counter = 0
        
        # Calculate confidence
        confidence = 0.0
        
        #  Blink detection (60%)
        blink_score = min(self.total_blinks / 2.0, 1.0) * 0.6
        confidence += blink_score
        
        # Time-based (40%)
        time_score = min(elapsed_time / 2.0, 0.4)
        confidence += time_score
        
        # Determine if live
        is_live = confidence >= 0.5
        
        # Generate message
        if is_live:
            message = " Live person verified"
        elif self.total_blinks < 2:
            message = " Please blink naturally"
        else:
            message = " Verifying liveness..."
        
        return {
            'live': is_live,
            'confidence': round(confidence, 2),
            'message': message,
            'blinks': self.total_blinks,
            'eye_aspect_ratio': round(ear, 3),
            'frame_count': self.frame_count,
            'method': 'advanced'
        }
    
    def reset(self):
        """Reset detector"""
        self.blink_counter = 0
        self.total_blinks = 0
        self.frame_count = 0
        self.start_time = None
