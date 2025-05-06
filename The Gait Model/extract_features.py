import cv2
import mediapipe as mp
import numpy as np
from scipy import signal

def extract_gait_features(video_path, sample_rate=5):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return None

    features = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_frames = int(fps * 2)  # At least 2 seconds of video
    
    if total_frames < min_frames:
        print(f"[ERROR] Video too short ({total_frames/fps:.1f}s). Need at least 2s.")
        return None

    # For temporal features
    temporal_features = []
    prev_landmarks = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % sample_rate != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            frame_feats = []
            # Get keypoints for major joints (hips, knees, ankles)
            keypoints = [23, 24, 25, 26, 27, 28]  # MediaPipe indices for lower body
            for idx in keypoints:
                lm = results.pose_landmarks.landmark[idx]
                frame_feats.extend([lm.x, lm.y, lm.visibility])
            
            # Calculate velocity if previous frame exists
            if prev_landmarks is not None:
                velocity = []
                for idx in keypoints:
                    curr = results.pose_landmarks.landmark[idx]
                    prev = prev_landmarks.landmark[idx]
                    dx = curr.x - prev.x
                    dy = curr.y - prev.y
                    velocity.extend([dx, dy])
                frame_feats.extend(velocity)
            
            prev_landmarks = results.pose_landmarks
            temporal_features.append(frame_feats)
        else:
            print(f"[WARNING] No keypoints in frame {frame_count}")

    cap.release()
    pose.close()

    if len(temporal_features) < 10:  # Need minimum frames for analysis
        print(f"[ERROR] Insufficient keypoints detected in video: {video_path}")
        return None

    # Convert to numpy array and normalize
    temporal_features = np.array(temporal_features)
    
    # Extract both spatial and temporal features
    spatial_features = np.mean(temporal_features[:, :18], axis=0)  # First 18 elements are x,y,visibility
    velocity_features = np.mean(temporal_features[:, 18:], axis=0)  # Velocity components
    
    # Add frequency domain features
    fft_features = []
    for i in range(temporal_features.shape[1]):
        sig = temporal_features[:, i]
        fft = np.abs(np.fft.fft(sig)[:len(sig)//2])  # Take magnitude of FFT
        fft_features.extend([np.mean(fft), np.std(fft), np.max(fft)])
    
    # Combine all features
    combined_features = np.concatenate([
        spatial_features,
        velocity_features,
        fft_features
    ])
    
    # Handle NaN values
    if np.any(np.isnan(combined_features)):
        print(f"[ERROR] NaN detected in features.")
        return None

    return combined_features