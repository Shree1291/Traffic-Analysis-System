Traffic Analysis System - IDPC Project

Overview:
The Traffic Analysis System is a distributed, multi-server-based system developed as part of the IDPC project. It aims to provide real-time traffic monitoring and analysis, leveraging advanced techniques like vehicle detection, edge pattern analysis, and intensity-based anomaly detection. The system includes a central client for task distribution and three specialized processing servers, each responsible for a different aspect of traffic analysis.

This system is built for scalability and efficiency, utilizing multi-server processing for real-time video analysis. The goal is to deliver comprehensive insights into traffic behavior, vehicle dynamics, and road conditions.

Features:
- Vehicle Detection and Tracking: Detects vehicles, counts them, and measures their speed in video footage.
- Edge Pattern Analysis: Identifies edge patterns to assess road conditions and traffic flow.
- Intensity and Anomaly Detection: Analyzes intensity changes and detects anomalies in traffic patterns.
- Real-time Processing: Processes video data in real-time across multiple servers.
- Distributed Architecture: Utilizes a multi-server setup for scalability and optimized performance.
- API Endpoints: Provides endpoints for server health checks and video processing tasks.

System Components:
The system consists of the following components:
- Client (client.py): Manages task distribution, result aggregation, and visualization.
- Vehicle Detection Server (Server 1): Handles vehicle detection, counting, and speed measurement.
- Edge Detection Server (Server 2): Analyzes edge patterns to assess traffic and road conditions.
- Intensity Analysis Server (Server 3): Analyzes intensity patterns and detects anomalies in traffic behavior.

Installation:
1. Clone the repository:
   git clone https://github.com/<your-username>/traffic-analysis-system.git

2. Install the required dependencies:
   pip install -r requirements.txt

3. Configure the system:
   - Set up the server endpoints and processing parameters in the config.py file.
   - Define server details and video file paths.

Usage:
1. Run the Client:
   The client is responsible for distributing video processing tasks to the specialized servers.
   python client.py

2. API Endpoints:
   - Health Check: To check server status:
     GET /health
   - Video Processing: To process video files:
     POST /process
     Upload one or more video files for analysis.

3. Start the Servers:
   Ensure all specialized servers (Vehicle Detection, Edge Detection, and Intensity Analysis) are running before initiating the client.

Data:
The system uses traffic video footage for analysis. The dataset is from the Traffic Database at the University of California, San Diego, which includes labeled videos taken over two days from a stationary camera overlooking I-5 in Seattle, WA.

Example Results:
- Vehicle Count Distribution
- Speed Distribution
- Edge Density
- Anomaly Detection Reports

API Documentation:
The system provides two main API endpoints:
- GET /health: Checks the health and status of the server.
- POST /process: Submits video files for processing and retrieves results.

Error Handling:
Comprehensive error handling mechanisms are implemented, covering common scenarios such as:
- Connection Failures
- Processing Timeouts
- Resource Limitations
- File Handling Errors
- Server Unavailability

Contributing:
This project was a team effort by the following members:
- Avin Saxena (202111015)
- K. Anamithra (202111041)
- Shriram Ashok Birajdar (202111078)
- Vatti Yeshwanth (202111086)
