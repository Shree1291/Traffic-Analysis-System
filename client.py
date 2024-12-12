import os
import requests
import logging
from typing import Dict, List, Optional, Tuple
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import tempfile
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/traffic_analysis.log'),  # Moved to 'logs' folder
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Client")

# Server Configuration
SERVERS = [
    {
        "url": "http://10.0.3.237:5000/process",  # Vehicle Detection Server
        "type": "vehicle_detection",
        "health_timeout": 5,
        "process_timeout": 300
    },
    {
        "url": "http://10.0.5.52:5001/process",  # Edge Detection Server
        "type": "edge_detection",
        "health_timeout": 5,
        "process_timeout": 300
    },
    {
        "url": "http://10.0.5.141:5002/process",  # Intensity Analysis Server
        "type": "intensity_analysis",
        "health_timeout": 5,
        "process_timeout": 300
    }
]

class VideoProcessor:
    def __init__(self, servers: List[Dict]):
        self.servers = servers
        self.active_servers = []
        self.processing_stats = {
            "total_videos": 0,
            "successful_videos": 0,
            "failed_videos": 0,
            "processing_time":0
            }

    def check_server_health(self, server: Dict) -> bool:
        """Check if a server is healthy."""
        try:
            health_url = f"{server['url'].rsplit('/', 1)[0]}/health"
            response = requests.get(
                health_url,
                timeout=server['health_timeout']
            )
            if response.status_code == 200:
                logger.info(f"Server {server['url']} is active")
                return True
            logger.warning(f"Server {server['url']} health check failed with status {response.status_code}")
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Server {server['url']} is not responding: {str(e)}")
            return False

    def get_active_servers(self) -> List[Dict]:
        """Get list of active servers."""
        self.active_servers = []
        with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
            future_to_server = {
                executor.submit(self.check_server_health, server): server
                for server in self.servers
            }
            for future in as_completed(future_to_server):
                server = future_to_server[future]
                try:
                    if future.result():
                        self.active_servers.append(server)
                except Exception as e:
                    logger.error(f"Error checking server {server['url']}: {str(e)}")
        return self.active_servers

    def send_videos_to_server(self, server: Dict, videos: List[str]) -> List[Dict]:
        """Send videos to a specific server for processing."""
        try:
            logger.info(f"Sending {len(videos)} videos to {server['url']}")
            start_time = time.time()
            
            # Send videos in batches
            batch_size = 10
            all_results = []
            
            for i in range(0, len(videos), batch_size):
                batch = videos[i:i + batch_size]
                files = []
                try:
                    for video in batch:
                        f = open(video, 'rb')
                        files.append(('videos', (os.path.basename(video), f, 'video/avi')))
                    
                    response = requests.post(
                        server['url'],
                        files=files,
                        timeout=server['process_timeout']
                    )
                    
                    if response.status_code == 200:
                        results = response.json().get('results', [])
                        all_results.extend(results)
                        self.processing_stats["successful_videos"] += len(results)
                        logger.info(f"Received {len(results)} results from {server['url']}")
                    else:
                        logger.error(f"Error from server {server['url']}: {response.text}")
                        self.processing_stats["failed_videos"] += len(batch)
                        
                except Exception as e:
                    logger.error(f"Error processing batch on {server['url']}: {str(e)}")
                    self.processing_stats["failed_videos"] += len(batch)
                finally:
                    # Close file handlers in all cases
                    for _, f in files:
                        f[1].close()
            
            processing_time = time.time() - start_time
            self.processing_stats["processing_time"] += processing_time
            
            return all_results
            
        except Exception as e:
            logger.error(f"Failed to connect to {server['url']}: {str(e)}")
            return []

    def distribute_videos(self, video_paths: List[str]) -> List[Dict]:
        """Distribute videos across active servers."""
        active_servers = self.get_active_servers()
        
        if not active_servers:
            raise ConnectionError("No servers are available")

        logger.info(f"Found {len(active_servers)} active servers")
        self.processing_stats["total_videos"] = len(video_paths)
        
        # Divide videos among active servers
        videos_per_server = len(video_paths) // len(active_servers)
        extra_videos = len(video_paths) % len(active_servers)
        
        current_index = 0
        all_results = []
        
        with ThreadPoolExecutor(max_workers=len(active_servers)) as executor:
            future_to_server = {}
            
            for i, server in enumerate(active_servers):
                # Calculate number of videos for this server
                num_videos = videos_per_server + (1 if i < extra_videos else 0)
                server_videos = video_paths[current_index:current_index + num_videos]
                current_index += num_videos
                
                # Submit processing job
                future = executor.submit(self.send_videos_to_server, server, server_videos)
                future_to_server[future] = server
            
            # Collect results
            for future in as_completed(future_to_server):
                server = future_to_server[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error processing videos on {server['url']}: {str(e)}")
        
        return all_results

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze processing results and generate statistics."""
    if not results:
        return {}

    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Basic statistics
    stats = {
        "total_videos_processed": len(df),
        "total_vehicles": df['vehicle_count'].sum() if 'vehicle_count' in df.columns else 0,
        "average_vehicles_per_video": df['vehicle_count'].mean() if 'vehicle_count' in df.columns else 0,
        "max_vehicles_in_video": df['vehicle_count'].max() if 'vehicle_count' in df.columns else 0,
        "min_vehicles_in_video": df['vehicle_count'].min() if 'vehicle_count' in df.columns else 0,
        "average_speed": df['average_speed'].mean() if 'average_speed' in df.columns else 0,
        "max_speed": df['average_speed'].max() if 'average_speed' in df.columns else 0,
        "min_speed": df['average_speed'].min() if 'average_speed' in df.columns else 0
    }

    if 'average_edge_pixels' in df.columns:
        stats.update({
            "average_edge_pixels": df['average_edge_pixels'].mean(),
            "max_edge_pixels": df['average_edge_pixels'].max(),
            "min_edge_pixels": df['average_edge_pixels'].min()
        })

    if 'average_std_dev' in df.columns:
        stats.update({
            "average_std_dev": df['average_std_dev'].mean(),
            "max_std_dev": df['average_std_dev'].max(),
            "min_std_dev": df['average_std_dev'].min()
        })

    return stats

def generate_visualizations(results: List[Dict], output_dir: str = "visualizations"):
    """Generate detailed visualizations of the results."""
    if not results:
        logger.warning("No results to visualize")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Set Seaborn style
    sns.set()
    
    # 1. Vehicle Count Distribution
    if 'vehicle_count' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='vehicle_count', bins=30, kde=True)
        plt.title('Distribution of Vehicle Counts per Video')
        plt.xlabel('Number of Vehicles')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'vehicle_count_distribution.png'))
        plt.close()

        # Box plot of vehicle counts
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=df['vehicle_count'])
        plt.title('Vehicle Count Box Plot')
        plt.ylabel('Number of Vehicles')
        plt.savefig(os.path.join(output_dir, 'vehicle_count_boxplot.png'))
        plt.close()

    # 2. Speed Distribution
    if 'average_speed' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='average_speed', bins=30, kde=True)
        plt.title('Distribution of Average Speeds')
        plt.xlabel('Speed (km/h)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'speed_distribution.png'))
        plt.close()

    # 3. Edge Pixels Distribution
    if 'average_edge_pixels' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='average_edge_pixels', bins=30, kde=True)
        plt.title('Distribution of Average Edge Pixels')
        plt.xlabel('Edge Pixels')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'edge_pixels_distribution.png'))
        plt.close()

    # 4. Standard Deviation Distribution
    if 'average_std_dev' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='average_std_dev', bins=30, kde=True)
        plt.title('Distribution of Average Standard Deviation')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'std_dev_distribution.png'))
        plt.close()

    # 5. Correlation Matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numeric Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()

def save_analysis_report(stats: Dict, processing_stats: Dict, output_dir: str = "analysis"):
    """Save analysis results to a detailed report."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(output_dir, f'analysis_report_{timestamp}.txt')
    
    with open(report_file, 'w') as f:
        f.write("Traffic Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Processing Statistics
        f.write("Processing Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Videos: {processing_stats['total_videos']}\n")
        f.write(f"Successfully Processed: {processing_stats['successful_videos']}\n")
        f.write(f"Failed: {processing_stats['failed_videos']}\n")
        f.write(f"Total Processing Time: {processing_stats['processing_time']:.2f} seconds\n\n")
        
        # Analysis Results
        f.write("Analysis Results:\n")
        f.write("-" * 30 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.2f}\n")
        
        # Key Insights
        f.write("\nKey Insights:\n")
        f.write("-" * 30 + "\n")
        
        # Traffic density insights
        if 'average_vehicles_per_video' in stats:
            avg_vehicles = stats['average_vehicles_per_video']
            if avg_vehicles > 20:
                f.write("- High traffic density detected\n")
            elif avg_vehicles > 10:
                f.write("- Moderate traffic density\n")
            else:
                f.write("- Low traffic density\n")
        
        # Speed insights
        if 'average_speed' in stats:
            avg_speed = stats['average_speed']
            if avg_speed > 80:
                f.write("- High average vehicle speed detected\n")
            elif avg_speed > 50:
                f.write("- Moderate average vehicle speed\n")
            else:
                f.write("- Low average vehicle speed\n")

    logger.info(f"Analysis report saved to {report_file}")

def main():
    try:
        # Initialize video processor
        processor = VideoProcessor(SERVERS)
        
        # Get video paths
        video_folder = "./traffic_videos/video"
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f"Video folder not found: {video_folder}")

        video_paths = [
            os.path.join(video_folder, file)
            for file in os.listdir(video_folder)
            if file.endswith(".avi")
        ]

        if not video_paths:
            raise ValueError("No video files found in the specified directory")

        logger.info(f"Found {len(video_paths)} videos to process")
        
        # Process videos
        results = processor.distribute_videos(video_paths)

        if results:
            # Generate analysis
            stats = analyze_results(results)
            
            # Log analysis summary
            logger.info("Analysis Summary:")
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
            
            # Generate visualizations
            generate_visualizations(results)
            # Save analysis report
            save_analysis_report(stats, processor.processing_stats)
            
            # Save raw results
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            results_file = f'raw_results_{timestamp}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Save summary metrics to CSV
            df = pd.DataFrame(results)
            df.to_csv(f'metrics_{timestamp}.csv', index=False)
            
            logger.info(f"Raw results saved to {results_file}")
            logger.info(f"Metrics saved to metrics_{timestamp}.csv")
            logger.info("Processing and analysis completed successfully")
            
            # Print final summary
            logger.info("\nProcessing Summary:")
            logger.info(f"Total videos processed: {processor.processing_stats['total_videos']}")
            logger.info(f"Successfully processed: {processor.processing_stats['successful_videos']}")
            logger.info(f"Failed to process: {processor.processing_stats['failed_videos']}")
            logger.info(f"Total processing time: {processor.processing_stats['processing_time']:.2f} seconds")
            
            # Print analysis summary
            logger.info("\nAnalysis Summary:")
            for key, value in stats.items():
                logger.info(f"{key}: {value:.2f}")
            
            # Generate performance report
            avg_time_per_video = processor.processing_stats['processing_time'] / processor.processing_stats['total_videos'] if processor.processing_stats['total_videos'] > 0 else 0
            success_rate = (processor.processing_stats['successful_videos'] / processor.processing_stats['total_videos']) * 100 if processor.processing_stats['total_videos'] > 0 else 0
            
            logger.info("\nPerformance Metrics:")
            logger.info(f"Average processing time per video: {avg_time_per_video:.2f} seconds")
            logger.info(f"Success rate: {success_rate:.2f}%")
            
        else:
            logger.warning("No results were returned from servers")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        raise
    except ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise
    finally:
        # Clean up any temporary files if they exist
        try:
            temp_dir = tempfile.gettempdir()
            for filename in os.listdir(temp_dir):
                if filename.endswith('.avi'):
                    os.remove(os.path.join(temp_dir, filename))
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

def setup_folders():
    """Create necessary folders for output."""
    folders = ['visualizations', 'analysis', 'logs']  # Added 'logs' folder
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

if __name__ == "__main__":
    # Setup output folders
    setup_folders()
    
    # Set up Seaborn style
    sns.set()
    
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    # Run main program
    main()