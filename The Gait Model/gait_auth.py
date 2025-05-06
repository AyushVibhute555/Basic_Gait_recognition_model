import cv2
from extract_features import extract_gait_features
from utils import load_dataset

def draw_box(frame, name, score, color):
    h, w, _ = frame.shape
    top_left = (50, 50)
    bottom_right = (w - 50, h - 50)
    cv2.rectangle(frame, top_left, bottom_right, color, 3)

    if name != "Unknown":
        confidence = min(max(score * 100, 0), 100)  # Clamp between 0-100
        label = f"{name} ({confidence:.1f}%)"
    else:
        label = "Unknown (no match)"
    
    # Add additional text about confidence level
    cv2.putText(frame, label, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if name != "Unknown":
        confidence_text = f"Confidence: {score:.2f}"
        cv2.putText(frame, confidence_text, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def process_video(video_path):
    print(f"[INFO] Extracting gait features from: {video_path}")
    features = extract_gait_features(video_path)

    if features is None:
        print("❌ Could not extract valid gait features. Try a clearer video.")
        return

    recognizer = load_dataset()
    name, score = recognizer.match_user(features)
    
    # Set color based on confidence
    if name != "Unknown":
        if score > 0.9:
            color = (0, 255, 0)  # High confidence - green
        elif score > 0.75:
            color = (0, 255, 255)  # Medium confidence - yellow
        else:
            color = (0, 165, 255)  # Low confidence - orange
    else:
        color = (0, 0, 255)  # Unknown - red

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open test video: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        draw_box(frame, name, score, color)
        cv2.imshow('Gait Authentication Result', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = input("Enter path to test video: ").strip('"')
    process_video(video_path)import cv2
from extract_features import extract_gait_features
from utils import load_dataset

def draw_box(frame, name, score, color):
    h, w, _ = frame.shape
    top_left = (50, 50)
    bottom_right = (w - 50, h - 50)
    cv2.rectangle(frame, top_left, bottom_right, color, 3)

    if name != "Unknown":
        confidence = min(max(score * 100, 0), 100)  # Clamp between 0-100
        label = f"{name} ({confidence:.1f}%)"
    else:
        label = "Unknown (no match)"
    
    # Add additional text about confidence level
    cv2.putText(frame, label, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if name != "Unknown":
        confidence_text = f"Confidence: {score:.2f}"
        cv2.putText(frame, confidence_text, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def process_video(video_path):
    print(f"[INFO] Extracting gait features from: {video_path}")
    features = extract_gait_features(video_path)

    if features is None:
        print("❌ Could not extract valid gait features. Try a clearer video.")
        return

    recognizer = load_dataset()
    name, score = recognizer.match_user(features)
    
    # Set color based on confidence
    if name != "Unknown":
        if score > 0.9:
            color = (0, 255, 0)  # High confidence - green
        elif score > 0.75:
            color = (0, 255, 255)  # Medium confidence - yellow
        else:
            color = (0, 165, 255)  # Low confidence - orange
    else:
        color = (0, 0, 255)  # Unknown - red

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open test video: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        draw_box(frame, name, score, color)
        cv2.imshow('Gait Authentication Result', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = input("Enter path to test video: ").strip('"')
    process_video(video_path)