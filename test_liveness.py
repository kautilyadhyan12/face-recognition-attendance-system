# test_liveness.py
import cv2
import base64
from liveness_detector import get_detector

def test_with_camera():
    """Test liveness detection with webcam"""
    detector = get_detector()
    cap = cv2.VideoCapture(0)
    
    print("🎥 Starting liveness detection test...")
    print("Press 'q' to quit")
    print("Move your head and blink naturally")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Test liveness detection
        result = detector.check_liveness(frame)
        
        # Display results on frame
        cv2.putText(frame, f"Live: {result['live']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if result['live'] else (0, 0, 255), 2)
        cv2.putText(frame, f"Confidence: {result['confidence']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Message: {result['message']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {result['blinks']}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        # Show frame
        cv2.imshow('Liveness Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_with_camera()