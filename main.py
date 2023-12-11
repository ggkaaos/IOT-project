import subprocess
import time
import signal
import os


def start_process(command):
    try:
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"Error while starting the process: {e}")
        return None


def stop_process(process):
    try:
        process.terminate()
        process.wait(timeout=5)
        if process.returncode is None:
            os.kill(process.pid, signal.SIGKILL)
    except Exception as e:
        print(f"Error closing process: {e}")


def start_publisher(server_ip_address):
    return subprocess.Popen(["python", "publisher.py", "--server-ip", server_ip_address])


# List of publisher
publishers = []

# Start the publishers with using IPs
server_ips = ["127.0.0.1", "192.168.0.2", "192.168.0.3"]
for server_ip in server_ips:
    publisher_process = start_publisher(server_ip)
    publishers.append(publisher_process)

# Start the processor
processor_process = start_process(
    ["python", "processor.py", "-p", "MobileNetSSD_deploy.prototxt", "-m", "MobileNetSSD_deploy.caffemodel"])

# Wait a few seconds before launching the viewer
time.sleep(5)

# Start the viewer
viewer_process = start_process(["python", "viewer.py", "--montageW", "2", "--montageH", "2"])

try:
    # loop?
    time.sleep(60)
finally:
    # Cleanup: Stop all processes
    for publisher_process in publishers:
        stop_process(publisher_process)
    stop_process(processor_process)
    stop_process(viewer_process)
