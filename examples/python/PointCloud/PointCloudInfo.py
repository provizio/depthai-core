#!/usr/bin/env python3
"""
Simple PointCloud Example

Demonstrates basic usage of the PointCloud node to generate 3D point clouds from stereo depth.
"""
import time
import os
import sys

sys.path.insert(0, '/home/tomas/code/depthai-device-kb/external/depthai-core/build/bindings/python/')

# Set logging level
os.environ["DEPTHAI_LEVEL"] = "info"
os.environ["DEPTHAI_DEVICE_RVC4_FWP"] = "/home/tomas/code/depthai-device-kb/build_docker_arm64_rvc4/RelWithDebInfo/depthai-device-rvc4-fwp.tar.xz"

import depthai as dai

device_ip = "10.11.0.29"
device_info = dai.DeviceInfo(device_ip)
device = dai.Device(device_info)

def main():
    # Create device (auto-discover first available device)
    device = dai.Device()
    
    print(f"Connected to device: {device.getDeviceName()}")
    print(f"MxId: {device.getMxId()}\n")
    
    # Create pipeline
    pipeline = dai.Pipeline(device)
    
    left = pipeline.create(dai.node.Camera)
    right = pipeline.create(dai.node.Camera)
    stereo = pipeline.create(dai.node.StereoDepth)
    pointCloud = pipeline.create(dai.node.PointCloud)
    
    left.build(dai.CameraBoardSocket.CAM_B)
    right.build(dai.CameraBoardSocket.CAM_C)
    
    # stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setSubpixel(True)
    
    pointCloud.setRunOnHost(True)
    pointCloud.setLengthUnit(dai.LengthUnit.METER)
    pointCloud.useCPU()
    
    # Keep organized point cloud (width * height points)
    pointCloud.keepPointCloudOrganized()
    
    # Transform point cloud to target coordinate system
    pointCloud.setTargetCoordinateSystem(dai.HousingCoordinateSystem.VESA_A)

    # Link nodes
    leftOut = left.requestFullResolutionOutput()
    rightOut = right.requestFullResolutionOutput()
    leftOut.link(stereo.left)
    rightOut.link(stereo.right)
    stereo.depth.link(pointCloud.inputDepth)
    stereo_queue = stereo.depth.createOutputQueue()
    
    queue = pointCloud.outputPointCloud.createOutputQueue()
    pipeline.start()
    
    print("PointCloud example started. Press Ctrl+C to stop\n")
    
    frameCount = 0
    
    try:
        while pipeline.isRunning():
            pclData = queue.get()
            if not pclData:
                continue
            depth_frame = stereo_queue.get().getCvFrame()
            print(f"Depth frame shape: {depth_frame.shape}")
            
            points = pclData.getPoints()
            
            print("\n========================================")
            print(f"Frame {frameCount}")
            print("========================================")
            print(f"Total points: {len(points)}")
            print(f"PointCloud organized: {pclData.isOrganized()}")
            print(f"PointCloud color: {pclData.isColor()}")
            print(f"PointCloud width: {pclData.getWidth()}")
            print(f"PointCloud height: {pclData.getHeight()}")
            
            print("\nBounding box (mm):")
            print(f"  X: [{pclData.getMinX()}, {pclData.getMaxX()}]")
            print(f"  Y: [{pclData.getMinY()}, {pclData.getMaxY()}]")
            print(f"  Z: [{pclData.getMinZ()}, {pclData.getMaxZ()}]")
            
            frameCount += 1
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\nException: {e}")
    
    pipeline.stop()
    print("\nStopped.")

if __name__ == "__main__":
    main()