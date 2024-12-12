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
        logging.FileHandler('server2.log'),
        logging.StreamHandler()
    ]
)

# Apply the NoWarningFilter to all handlers
logger = logging.getLogger("Server2")
for handler in logger.handlers:
    handler.addFilter(NoWarningFilter())

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

@dataclass
class EdgeDetectionConfig:
    """Configuration for edge detection parameters."""
    resize_width: int = 640
    resize_height: int = 480
    frame_skip: int = 1
    history: int = 200
    var_threshold: int = 40
    detect_shadows: bool = False
    learning_rate: float = 0.001
    min_edge_area: int = 100
    max_edge_area: int = 10000
    canny_low: int = 50
    canny_high: int = 150
    blur_kernel_size: int = 5
    dilate_iterations: int = 2
    min_line_length: int = 50
    max_line_gap: int = 10
    edge_density_threshold: float = 0.1

class EdgeAnalyzer:
    """Enhanced edge detection and analysis."""
    def __init__(self, config: EdgeDetectionConfig):
        self.config = config
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=config.detect_shadows
        )
        # Initialize Gaussian and bilateral filter kernels
        self.gaussian_kernel = (config.blur_kernel_size, config.blur_kernel_size)
        self.bilateral_params = {
            'd': 9,
            'sigmaColor': 75,
            'sigmaSpace': 75
        }

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced frame preprocessing."""
        # Resize frame
        frame = cv2.resize(frame, (self.config.resize_width, self.config.resize_height))
       
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
       
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(
            enhanced,
            d=self.bilateral_params['d'],
            sigmaColor=self.bilateral_params['sigmaColor'],
            sigmaSpace=self.bilateral_params['sigmaSpace']
        )
       
        return bilateral

    def detect_edges(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Detect and analyze edges in the frame."""
        # Get foreground mask
        fg_mask = self.background_subtractor.apply(
            frame,
            learningRate=self.config.learning_rate
        )
       
        # Apply Canny edge detection
        edges = cv2.Canny(
            frame,
            self.config.canny_low,
            self.config.canny_high
        )
       
        # Enhance edges
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=self.config.dilate_iterations)
       
        # Find contours
        contours, _ = cv2.findContours(
            dilated_edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
       
        # Analyze edge features
        edge_features = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_edge_area < area < self.config.max_edge_area:
                x, y, w, h = cv2.boundingRect(contour)
                perimeter = cv2.arcLength(contour, True)
                complexity = perimeter / (4 * np.sqrt(area)) if area > 0 else 0
               
                edge_features.append({
                    'area': area,
                    'perimeter': perimeter,
                    'complexity': complexity,
                    'bbox': (x, y, w, h)
                })
       
        return edges, edge_features

    def analyze_edge_density(self, edges: np.ndarray) -> Dict:
        """Analyze edge density and distribution."""
        total_pixels = edges.size
        edge_pixels = np.count_nonzero(edges)
        density = edge_pixels / total_pixels
       
        # Analyze edge distribution in regions
        regions = np.zeros(9)  # 3x3 grid
        h, w = edges.shape
        cell_h, cell_w = h // 3, w // 3
       
        for i in range(3):
            for j in range(3):
                region = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                regions[i*3 + j] = np.count_nonzero(region) / region.size
       
        return {
            'density': density,
            'region_distribution': regions.tolist(),
            'edge_pixels': edge_pixels
        }

    def detect_lines(self, edges: np.ndarray) -> List[Dict]:
        """Detect and analyze lines in the edge image."""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap
        )
       
        if lines is None:
            return []
       
        line_features = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
           
            line_features.append({
                'length': length,
                'angle': angle,
                'start': (x1, y1),
                'end': (x2, y2)
            })
       
        return line_features

def process_video_worker(video_path: str, video_name: str) -> Optional[Dict]:
    """Process single video file with edge analysis."""
    try:
        start_time = time.time()  # Define start_time here
        config = EdgeDetectionConfig()
        analyzer = EdgeAnalyzer(config)
       
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        frame_count = 0
        total_edge_density = 0
        total_edge_pixels = 0
        edge_complexities = []
        line_orientations = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % config.frame_skip != 0:
                    frame_count += 1
                    continue

                # Process frame
                processed_frame = analyzer.preprocess_frame(frame)
                edges, edge_features = analyzer.detect_edges(processed_frame)
               
                # Analyze edges
                edge_analysis = analyzer.analyze_edge_density(edges)
                line_features = analyzer.detect_lines(edges)
               
                # Collect metrics
                total_edge_density += edge_analysis['density']
                total_edge_pixels += edge_analysis['edge_pixels']
               
                for feature in edge_features:
                    edge_complexities.append(feature['complexity'])
               
                for line in line_features:
                    line_orientations.append(line['angle'])
               
                frame_count += 1

        finally:
            cap.release()

        if frame_count > 0:
            avg_edge_density = total_edge_density / frame_count
            avg_edge_pixels = total_edge_pixels / frame_count
            avg_complexity = np.mean(edge_complexities) if edge_complexities else 0
           
            # Analyze line orientations
            orientation_hist = np.histogram(
                line_orientations,
                bins=8,
                range=(-180, 180)
            )[0].tolist() if line_orientations else []

            processing_time = time.time() - start_time
            processing_fps = round(frame_count / processing_time, 2) if processing_time > 0 else 0

            return {
                "video": video_name,
                "average_edge_density": round(avg_edge_density, 4),
                "average_edge_pixels": round(avg_edge_pixels, 2),
                "average_complexity": round(avg_complexity, 4),
                "orientation_distribution": orientation_hist,
                "processed_frames": frame_count,
                "processed_by": "Server2",
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
        cpu_percent = psutil.cpu_percent(interval=1)
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
            "server_type": "edge_detection",
            "capabilities": {
                "processing_power": 1.2,
                "specialized_for": "edge_analysis",
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
        videos_per_second = round(len(results) / processing_time, 2) if processing_time > 0 else 0

        return jsonify({
            "status": "success",
            "results": results,
            "videos_processed": len(results),
            "processing_time": round(processing_time, 2),
            "videos_per_second": videos_per_second
        })

    except Exception as e:
        logger.error(f"Error in process_videos: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)