from flask import Flask, request, jsonify
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
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
from scipy import stats
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('server3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Server3")


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

@dataclass
class IntensityAnalysisConfig:
    """Configuration for intensity analysis parameters."""
    resize_width: int = 640
    resize_height: int = 480
    frame_skip: int = 3
    history: int = 200
    var_threshold: int = 40
    detect_shadows: bool = False
    learning_rate: float = 0.001
    min_region_size: int = 50
    grid_size: int = 4  # Divides frame into 4x4 grid
    intensity_bins: int = 256
    temporal_window: int = 10  # Frames to consider for temporal analysis
    motion_threshold: float = 0.1
    blur_kernel_size: int = 5

class IntensityAnalyzer:
    """Enhanced intensity and standard deviation analysis."""
    def __init__(self, config: IntensityAnalysisConfig):
        self.config = config
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=config.detect_shadows
        )
        self.temporal_buffer = []
        self.frame_metrics_history = []

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced frame preprocessing."""
        # Resize frame
        frame = cv2.resize(frame, (self.config.resize_width, self.config.resize_height))
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            enhanced,
            (self.config.blur_kernel_size, self.config.blur_kernel_size),
            0
        )
        
        return enhanced, blurred

    def analyze_intensity_distribution(self, frame: np.ndarray) -> Dict:
        """Analyze intensity distribution in the frame."""
        # Calculate histogram
        hist = cv2.calcHist([frame], [0], None, [self.config.intensity_bins], [0, 256])
        hist = hist.flatten() / frame.size
        
        # Calculate statistical measures
        mean_intensity = np.mean(frame)
        std_dev = np.std(frame)
        skewness = stats.skew(frame.flatten())
        kurtosis = stats.kurtosis(frame.flatten())
        
        # Calculate percentiles
        percentiles = np.percentile(frame, [25, 50, 75])
        
        return {
            'mean_intensity': mean_intensity,
            'std_dev': std_dev,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'histogram': hist.tolist(),
            'percentiles': {
                'q1': percentiles[0],
                'median': percentiles[1],
                'q3': percentiles[2]
            }
        }

    def analyze_spatial_distribution(self, frame: np.ndarray) -> Dict:
        """Analyze spatial distribution of intensities."""
        h, w = frame.shape
        cell_h, cell_w = h // self.config.grid_size, w // self.config.grid_size
        grid_stats = []
        
        for i in range(self.config.grid_size):
            row_stats = []
            for j in range(self.config.grid_size):
                cell = frame[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_stats = {
                    'mean': np.mean(cell),
                    'std_dev': np.std(cell),
                    'min': np.min(cell),
                    'max': np.max(cell)
                }
                row_stats.append(cell_stats)
            grid_stats.append(row_stats)
        
        return {
            'grid_stats': grid_stats,
            'horizontal_gradient': np.mean(np.diff(frame, axis=1)),
            'vertical_gradient': np.mean(np.diff(frame, axis=0))
        }

    def analyze_temporal_changes(self, frame: np.ndarray) -> Dict:
        """Analyze temporal changes in intensity."""
        self.temporal_buffer.append(frame)
        if len(self.temporal_buffer) > self.config.temporal_window:
            self.temporal_buffer.pop(0)
        
        if len(self.temporal_buffer) < 2:
            return {
                'temporal_std_dev': 0,
                'intensity_change_rate': 0,
                'motion_intensity': 0
            }
        
        # Calculate temporal statistics
        temporal_stack = np.stack(self.temporal_buffer)
        temporal_std = np.std(temporal_stack, axis=0)
        intensity_changes = np.diff(temporal_stack, axis=0)
        
        return {
            'temporal_std_dev': np.mean(temporal_std),
            'intensity_change_rate': np.mean(np.abs(intensity_changes)),
            'motion_intensity': np.sum(np.abs(intensity_changes) > self.config.motion_threshold) / frame.size
        }

    def detect_anomalies(self, metrics: Dict) -> List[Dict]:
        """Detect anomalies in intensity patterns."""
        self.frame_metrics_history.append(metrics)
        if len(self.frame_metrics_history) > 100:  # Keep last 100 frames
            self.frame_metrics_history.pop(0)
        
        anomalies = []
        if len(self.frame_metrics_history) > 10:
            # Calculate baseline statistics
            baseline_means = np.mean([m['mean_intensity'] for m in self.frame_metrics_history[:-1]])
            baseline_std = np.std([m['std_dev'] for m in self.frame_metrics_history[:-1]])
            
            current_metrics = self.frame_metrics_history[-1]
            
            # Check for anomalies
            if abs(current_metrics['mean_intensity'] - baseline_means) > 2 * baseline_std:
                anomalies.append({
                    'type': 'intensity_anomaly',
                    'severity': abs(current_metrics['mean_intensity'] - baseline_means) / baseline_std
                })
            
            if current_metrics['motion_intensity'] > np.mean([m['motion_intensity'] for m in self.frame_metrics_history[:-1]]) * 2:
                anomalies.append({
                    'type': 'motion_anomaly',
                    'severity': current_metrics['motion_intensity']
                })
        
        return anomalies

def summarize_anomalies(anomalies: List[Dict]) -> Dict:
    """Summarize detected anomalies."""
    summary = defaultdict(int)
    max_severity = defaultdict(float)
    
    for anomaly in anomalies:
        anomaly_type = anomaly['type']
        severity = anomaly['severity']
        summary[anomaly_type] += 1
        max_severity[anomaly_type] = max(max_severity[anomaly_type], severity)
    
    return {
        'counts': dict(summary),
        'max_severity': dict(max_severity)
    }

def process_video_worker(video_path: str, video_name: str) -> Optional[Dict]:
    """Process single video file with intensity analysis."""
    try:
        start_time = time.time()  # Define start_time here
        config = IntensityAnalysisConfig()
        analyzer = IntensityAnalyzer(config)
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        frame_count = 0
        intensity_metrics = []
        spatial_metrics = []
        temporal_metrics = []
        detected_anomalies = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % config.frame_skip != 0:
                    frame_count += 1
                    continue

                # Process frame
                enhanced, blurred = analyzer.preprocess_frame(frame)
                
                # Analyze frame
                intensity_dist = analyzer.analyze_intensity_distribution(enhanced)
                spatial_dist = analyzer.analyze_spatial_distribution(enhanced)
                temporal_changes = analyzer.analyze_temporal_changes(blurred)
                
                # Collect metrics
                intensity_metrics.append(intensity_dist)
                spatial_metrics.append(spatial_dist)
                temporal_metrics.append(temporal_changes)
                
                # Detect anomalies
                anomalies = analyzer.detect_anomalies({
                    'mean_intensity': intensity_dist['mean_intensity'],
                    'std_dev': intensity_dist['std_dev'],
                    'motion_intensity': temporal_changes['motion_intensity']
                })
                detected_anomalies.extend(anomalies)
                
                frame_count += 1

        finally:
            cap.release()

        if frame_count > 0:
            # Calculate aggregate metrics
            avg_intensity = np.mean([m['mean_intensity'] for m in intensity_metrics])
            avg_std_dev = np.mean([m['std_dev'] for m in intensity_metrics])
            avg_temporal_std = np.mean([m['temporal_std_dev'] for m in temporal_metrics])
            
            # Calculate intensity stability
            intensity_stability = 1.0 - (np.std([m['mean_intensity'] for m in intensity_metrics]) / avg_intensity)
            
            processing_time = time.time() - start_time
            processing_fps = round(frame_count / processing_time, 2) if processing_time > 0 else 0

            return {
                "video": video_name,
                "average_intensity": round(avg_intensity, 2),
                "average_std_dev": round(avg_std_dev, 2),
                "temporal_std_dev": round(avg_temporal_std, 2),
                "intensity_stability": round(intensity_stability, 4),
                "anomalies_detected": len(detected_anomalies),
                "anomaly_summary": summarize_anomalies(detected_anomalies),
                "processed_frames": frame_count,
                "processed_by": "Server3",
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
            "server_type": "intensity_analysis",
            "capabilities": {
                "processing_power": 0.8,
                "specialized_for": "intensity_analysis",
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
                logger.error(f"Error queuing video {video_name}: {str(e)}")

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
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)

