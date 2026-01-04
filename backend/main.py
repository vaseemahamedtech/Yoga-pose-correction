from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io
import base64
from datetime import datetime
import asyncio
import random


app = FastAPI(title="Yoga Pose Correction API", version="1.0.0")


# CORS configuration - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


print("✅ MediaPipe Pose Detection initialized successfully!")


class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"🔌 WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)


manager = ConnectionManager()


def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def analyze_pose_landmarks(landmarks, pose_type):
    """Analyze pose landmarks and provide specific feedback"""
    feedback = []
    score = 85  # Start with high score
    
    points = {}
    for idx, landmark in enumerate(landmarks.landmark):
        points[idx] = (landmark.x, landmark.y, landmark.z)
    
    # Basic alignment checks (applicable to all poses)
    left_shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
    
    if shoulder_diff > 0.02:
        score -= 5
        if left_shoulder[1] > right_shoulder[1]:
            feedback.append("Lower your left shoulder slightly")
        else:
            feedback.append("Lower your right shoulder slightly")
    
    # Hip alignment
    left_hip = points[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = points[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_diff = abs(left_hip[1] - right_hip[1])
    
    if hip_diff > 0.02:
        score -= 5
        if left_hip[1] > right_hip[1]:
            feedback.append("Shift weight to your right leg slightly")
        else:
            feedback.append("Shift weight to your left leg slightly")
    
    # Pose-specific analysis
    if pose_type == "mountain":
        # Check arm position
        left_elbow_angle = calculate_angle(
            points[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            points[mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        
        if left_elbow_angle < 160:
            score -= 8
            feedback.append("Straighten your arms more by your sides")
        
        # Check knee alignment
        left_knee_angle = calculate_angle(
            points[mp_pose.PoseLandmark.LEFT_HIP.value],
            points[mp_pose.PoseLandmark.LEFT_KNEE.value],
            points[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        
        if left_knee_angle < 170:
            score -= 7
            feedback.append("Straighten your legs more")
    
    elif pose_type == "warrior2":
        # Front knee angle
        front_knee_angle = calculate_angle(
            points[mp_pose.PoseLandmark.LEFT_HIP.value],
            points[mp_pose.PoseLandmark.LEFT_KNEE.value],
            points[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        
        if front_knee_angle < 85:
            score -= 6
            feedback.append("Bend your front knee more deeply - aim for 90 degrees")
        elif front_knee_angle > 95:
            score -= 6
            feedback.append("Reduce bend in your front knee - aim for 90 degrees")
        
        # Check if front knee is over ankle
        knee_x = points[mp_pose.PoseLandmark.LEFT_KNEE.value][0]
        ankle_x = points[mp_pose.PoseLandmark.LEFT_ANKLE.value][0]
        if abs(knee_x - ankle_x) > 0.1:
            score -= 8
            if knee_x > ankle_x:
                feedback.append("Move your front knee slightly to the right over your ankle")
            else:
                feedback.append("Move your front knee slightly to the left over your ankle")
    
    elif pose_type == "tree":
        # Standing leg straightness
        standing_knee_angle = calculate_angle(
            points[mp_pose.PoseLandmark.RIGHT_HIP.value],
            points[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            points[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        
        if standing_knee_angle < 170:
            score -= 10
            feedback.append("Straighten your standing leg more")
        
        # Balance check
        left_ankle = points[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = points[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        com_x = (left_ankle[0] + right_ankle[0]) / 2
        
        if abs(com_x - right_ankle[0]) > 0.05:
            score -= 7
            if com_x > right_ankle[0]:
                feedback.append("Shift your weight slightly to the left for better balance")
            else:
                feedback.append("Shift your weight slightly to the right for better balance")
    
    elif pose_type == "downward_dog":
        # Spine alignment
        shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = points[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = points[mp_pose.PoseLandmark.LEFT_KNEE.value]
        
        spine_angle = calculate_angle(shoulder, hip, knee)
        if spine_angle < 160:
            score -= 12
            feedback.append("Straighten your back more - aim for a straight line from hands to hips")
        
        # Heel position
        heel = points[mp_pose.PoseLandmark.LEFT_HEEL.value]
        ankle = points[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        if heel[1] - ankle[1] > 0.1:
            score -= 5
            feedback.append("Press your heels down toward the floor")
    
    # Add general improvements if few specific feedbacks
    if len(feedback) < 2:
        general_tips = [
            "Engage your core muscles more",
            "Breathe deeply and steadily",
            "Relax your facial muscles",
            "Maintain steady gaze forward",
            "Keep your neck aligned with spine"
        ]
        feedback.extend(random.sample(general_tips, 2 - len(feedback)))
    
    # Ensure score is reasonable
    #score = max(60, min(95, score))
    
    return {
        "feedback": feedback,
        "score": score,
        "pose_type": pose_type,
        "pose_name": pose_type.replace("2", " II").title().replace("Dog", " Dog"),
        "detailed_metrics": {
            "alignment_score": max(70, score - random.randint(0, 10)),
            "balance_score": max(65, score - random.randint(0, 15)),
            "form_score": score,
            "stability_score": max(75, score - random.randint(0, 8))
        }
    }


@app.post("/analyze-photo/")
async def analyze_photo(file: UploadFile = File(...), pose_type: str = "mountain"):
    try:
        print(f"🔍 Analyzing photo for {pose_type} pose...")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Convert to RGB for MediaPipe
        if len(image_np.shape) == 2:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            print("✅ Pose detected - analyzing...")
            
            # Analyze pose with real landmark data
            analysis = analyze_pose_landmarks(results.pose_landmarks, pose_type)
            
            # Create annotated image - ensure it's in BGR format with 3 channels
            if len(image_np.shape) == 2:
                # Grayscale image
                annotated_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            elif image_np.shape[2] == 4:
                # RGBA image - convert to BGR
                annotated_image = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            elif image_np.shape[2] == 3:
                # Already 3 channels, but might be RGB - convert to BGR
                annotated_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                annotated_image = image_np.copy()
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Add analysis text to image
            y_offset = 30
            cv2.putText(
                annotated_image,
                f"Score: {analysis['score']}/100 - {analysis['pose_name']}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            for i, feedback in enumerate(analysis["feedback"][:3]):
                y_offset += 25
                cv2.putText(
                    annotated_image,
                    feedback,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            annotated_image_str = base64.b64encode(buffer).decode()
            
            response_data = {
                "status": "success",
                "analysis": analysis,
                "annotated_image": annotated_image_str,
                "landmarks_detected": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            print(f"✅ Analysis complete! Score: {analysis['score']}")
            return response_data
            
        else:
            print("❌ No pose detected in image")
            return {
                "status": "error",
                "message": "No human pose detected. Please ensure:\n• Entire body is visible\n• Good lighting\n• Clear background\n• Front-facing position",
                "landmarks_detected": False
            }
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error processing image: {str(e)}"}
        )


@app.websocket("/ws/live-analysis")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 WebSocket connected - Live analysis started")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                # Decode image from base64
                image_data = base64.b64decode(data.split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to decode image"
                    })
                    continue
               
                # Process frame with MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # Analyze pose in real-time (using mountain as default for live)
                    analysis = analyze_pose_landmarks(results.pose_landmarks, "mountain")
                    
                    # Draw landmarks on frame
                    mp_drawing.draw_landmarks(
                        frame, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Add real-time feedback to frame
                    cv2.putText(
                        frame,
                        f"Live Score: {analysis['score']}/100",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                    
                    cv2.putText(
                        frame,
                        f"Pose: {analysis['pose_name']}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA
                    )
                    
                    y_offset = 90
                    for i, feedback in enumerate(analysis["feedback"][:2]):
                        cv2.putText(
                            frame,
                            feedback,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA
                        )
                        y_offset += 25
                    
                    # Encode frame back to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_encoded = base64.b64encode(buffer).decode()
                    
                    await websocket.send_json({
                        "type": "analysis",
                        "analysis": analysis,
                        "frame": f"data:image/jpeg;base64,{frame_encoded}",
                        "timestamp": datetime.utcnow().isoformat(),
                        "landmarks_detected": True
                    })
                else:
                    await websocket.send_json({
                        "type": "no_pose_detected",
                        "message": "Move into frame and ensure full body visibility",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except Exception as e:
                print(f"WebSocket processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


@app.get("/supported-poses")
async def get_supported_poses():
    return {
        "poses": [
            {"id": "mountain", "name": "Mountain Pose", "description": "Standing straight with arms at sides"},
            {"id": "warrior2", "name": "Warrior II", "description": "Front knee bent, arms extended"},
            {"id": "tree", "name": "Tree Pose", "description": "Standing on one leg, other foot on inner thigh"},
            {"id": "downward_dog", "name": "Downward Facing Dog", "description": "Inverted V shape with hands and feet"}
        ]
    }


@app.get("/")
async def root():
    return {
        "message": "Yoga Pose Correction API", 
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "photo_analysis": "POST /analyze-photo/",
            "live_analysis": "WebSocket /ws/live-analysis",
            "supported_poses": "GET /supported-poses",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "service": "yoga-pose-correction",
        "mediapipe": "active"
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Yoga Pose Correction API...")
    print("✅ MediaPipe Pose Detection: ACTIVE")
    print("📸 Photo analysis: POST http://localhost:8000/analyze-photo/")
    print("📹 Live analysis: WebSocket ws://localhost:8000/ws/live-analysis")
    print("📍 Supported poses: GET http://localhost:8000/supported-poses")
    print("❤️  Health check: GET http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
