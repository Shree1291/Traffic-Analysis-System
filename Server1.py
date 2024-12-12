from flask import Flask, request, jsonify
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import os
import tempfile
from werkzeug.utils import secure_filename
import multiprocessing as mp
import time
import psutil
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# Custom filter to exclude WARNING logs
class NoWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to include INFO, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('server1.log'),
        logging.StreamHandler()
    ]
)

# Apply the NoWarningFilter to all handlers
logger = logging.getLogger("Server1")
for handler in logger.handlers:
    handler.addFilter(NoWarningFilter())

app = Flask(name)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

@dataclass
class VehicleDetectionConfig:
    """Configuration for vehicle detection parameters."""
    resize_width: int = 640
    resize_height: int = 480
    frame_skip: int = 2
    history: int = 200
    var_threshold: int = 40
    detect_shadows: bool = False
    learning_rate: float = 0.001
    min_contour_area: int = 500
    max_contour_area: int = 15000
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 4.0
    distance_threshold: float = 50.0
    min_speed: float = 0.0
    max_speed: float = 200.0
    distance_scale_factor: float = 0.1
    min_tracking_time: int = 3  # minimum frames to track a vehicle

class VehicleTracker:
    """Track individual vehicles across frames."""
    def init(self, bbox: Tuple[int, int, int, int], track_id: int):
        self.track_id = track_id
        self.bbox = bbox
        self.center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
        self.positions = [(self.center, time.time())]
        self.lost_frames = 0
        self.total_frames = 1
        self.speeds = []
        self.active = True

    def update(self, bbox: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Update tracker with new detection."""
        current_time = time.time()
        
        if bbox is not None:
            self.bbox = bbox
            self.center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
            self.lost_frames = 0
        else:
            self.lost_frames += 1
            
        self.positions.append((self.center, current_time))
        self.total_frames += 1
        
        # Calculate speed if possible
        if len(self.positions) >= 2:
            self.calculate_speed()

    def calculate_speed(self) -> None:
        """Calculate current speed in pixels per second."""
        pos1, t1 = self.positions[-2]
        pos2, t2 = self.positions[-1]
        
        dt = t2 - t1
        if dt > 0:
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            speed = np.sqrt(dx*dx + dy*dy) / dt
            self.speeds.append(speed)

    def get_average_speed(self) -> float:
        """Get average speed over tracker lifetime."""
        if not self.speeds:
            return 0.0
        return sum(self.speeds) / len(self.speeds)

class VehicleDetector:
    """Enhanced vehicle detection and tracking."""
    def init(self, config: VehicleDetectionConfig):
        self.config = config
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=config.detect_shadows
        )
        self.trackers: Dict[int, VehicleTracker] = {}
        self.next_id = 0
        
        # Initialize additional detection methods
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced frame preprocessing."""
        # Resize frame
        frame = cv2.resize(frame, (self.config.resize_width, self.config.resize_height))
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        return blurred

    def detect_vehicles(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect vehicles in frame."""
        # Get foreground mask
        fg_mask = self.background_subtractor.apply(
            frame,
            learningRate=self.config.learning_rate
        )
        
        # Enhance mask
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter valid contours
        valid_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_contour_area < area < self.config.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                
                if (self.config.min_aspect_ratio < aspect_ratio < 
                    self.config.max_aspect_ratio):
                    valid_detections.append((x, y, w, h))
        
        return valid_detections

    def update_trackers(self, detections: List[Tuple[int, int, int, int]]) -> None:
        """Update vehicle trackers with new detections."""
        # Match detections to existing trackers
        matched_trackers = {}
        unmatched_detections = []
        
        for bbox in detections:
            center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
            matched = False
            
            for track_id, tracker in self.trackers.items():
                if not tracker.active:
                    continue
                    
                dist = np.sqrt(
                    (center[0] - tracker.center[0])**2 +
                    (center[1] - tracker.center[1])**2
                )
                
                if dist < self.config.distance_threshold:
                    matched_trackers[track_id] = bbox
                    matched = True
                    break
            
            if not matched:
                unmatched_detections.append(bbox)
        
        # Update matched trackers
        for track_id, bbox in matched_trackers.items():
            self.trackers[track_id].update(bbox)
        
        # Create new trackers for unmatched detections
        for bbox in unmatched_detections:
            self.trackers[self.next_id] = VehicleTracker(bbox, self.next_id)
            self.next_id += 1
        
        # Update unmatched trackers and remove inactive ones
        for track_id, tracker in list(self.trackers.items()):
            if track_id not in matched_trackers:
                tracker.update(None)
                if tracker.lost_frames > 10:  # Maximum frames to keep lost tracker
                    tracker.active = False

def process_video_worker(video_path: str, video_name: str) -> Optional[Dict]:
    """Process single video file."""
    try:
        start_time = time.time()  # Define start_time here
        config = VehicleDetectionConfig()
        detector = VehicleDetector(config)
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        active_tracks = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % config.frame_skip != 0:
                    frame_count += 1
                    continue

                # Process frame
                processed_frame = detector.preprocess_frame(frame)
                detections = detector.detect_vehicles(processed_frame)
                detector.update_trackers(detections)
                
                frame_count += 1

        finally:
            cap.release()

        # Analyze tracking results
        valid_tracks = [
            tracker for tracker in detector.trackers.values()
            if (tracker.total_frames >= config.min_tracking_time and
                config.min_speed <= tracker.get_average_speed() <= config.max_speed)
        ]

        if valid_tracks:
            average_speed = sum(t.get_average_speed() for t in valid_tracks) / len(valid_tracks)
            processing_time = time.time() - start_time
            processing_fps = round(frame_count / processing_time, 2) if processing_time > 0 else 0
            return {
                "video": video_name,
                "vehicle_count": len(valid_tracks),
                "average_speed": round(average_speed * config.distance_scale_factor * 3.6, 2),  # Convert to km/h
                "processed_frames": frame_count,
                "processed_by": "Server1",
                "processing_fps": processing_fps
            }
        
        return None

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with system metrics."""
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return jsonify({
            "status": "healthy",
            "system_metrics": {
                "cpu_usage": cpu_percent,
                "memory_used": memory.percent,
                "memory_available": memory.available // (1024 * 1024),  # MB
                "disk_free": disk.free // (1024 * 1024 * 1024),  # GB
                "disk_percent": disk.percent
            },
            "server_type": "vehicle_detection",
            "capabilities": {
                "processing_power": 1.0,
                "specialized_for": "vehicle_tracking",
                "max_concurrent_videos": mp.cpu_count()
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_videos():
    """Process multiple videos with enhanced error handling and monitoring."""
    start_time = time.time()
    
    try:
        if 'videos' not in request.files:
            return jsonify({"error": "No video files provided"}), 400

        videos = request.files.getlist('videos')
        if not videos:
            return jsonify({"error": "No video files provided"}), 400

        # Create processing pool
        pool = mp.Pool(mp.cpu_count())
        jobs = []
        results = []

        # Process videos
        for video in videos:
            try:
                video_name = secure_filename(video.filename)
                logger.info(f"Received video: {video_name}")
                
                # Save video to temporary file
                temp_video = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(video_name)[1]
                )
                video.save(temp_video.name)
                temp_video.close()
                
                # Submit processing job
                job = pool.apply_async(
                    process_video_worker,
                    args=(temp_video.name, video_name)
                )
                jobs.append((job, temp_video.name))
                
            except Exception as e:
                logger.error(f"Error queuing video {video.filename}: {str(e)}")

        # Collect results
        for job, temp_file in jobs:
            try:
                result = job.get(timeout=300)  # 5 minutes timeout per video
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {temp_file}: {str(e)}")
            finally:
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.error(f"Error removing temp file {temp_file}: {str(e)}")

        pool.close()
        pool.join()

        processing_time = time.time() - start_time
        return jsonify({
            "status": "success",
            "results": results,
            "videos_processed": len(results),
            "processing_time": round(processing_time, 2),
            "videos_per_second": round(len(results) / processing_time, 2) if processing_time > 0 else 0
        })

    except Exception as e:
        logger.error(f"Error in process_videos: {str(e)}")
        return jsonify({"error": str(e)}), 500

if name == "main":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)