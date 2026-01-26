/**
 * Simple PointCloud Example
 * 
 * Demonstrates basic usage of the PointCloud node to generate 3D point clouds from stereo depth.
 */

#include <iostream>
#include <chrono>
#include <thread>
#include "depthai/depthai.hpp"

int main() {
    using namespace std;
    
    // Create device (auto-discover first available device)
    auto device = std::make_shared<dai::Device>();
    
    cout << "Connected to device: " << device->getDeviceName() << endl;
    cout << "DeviceId: " << device->getDeviceId() << "\n" << endl;
    
    // Create pipeline
    dai::Pipeline pipeline(device);
    
    auto left = pipeline.create<dai::node::Camera>();
    auto right = pipeline.create<dai::node::Camera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    auto pointCloud = pipeline.create<dai::node::PointCloud>();
    
    left->build(dai::CameraBoardSocket::CAM_B);
    right->build(dai::CameraBoardSocket::CAM_C);
    
    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::DEFAULT);
    stereo->setSubpixel(true);
    
    pointCloud->setRunOnHost(true);
    pointCloud->setDepthUnit(dai::DepthUnit::MILLIMETER);
    pointCloud->useCPU();
    
    // Transform point cloud to target coordinate system
    pointCloud->setTargetCoordinateSystem(dai::HousingCoordinateSystem::VESA_A);
    
    // Link nodes
    auto leftOut = left->requestFullResolutionOutput();
    auto rightOut = right->requestFullResolutionOutput();
    leftOut->link(stereo->left);
    rightOut->link(stereo->right);
    stereo->depth.link(pointCloud->inputDepth);
    
    auto queue = pointCloud->outputPointCloud.createOutputQueue();
    pipeline.start();
    
    cout << "PointCloud example started. Press Ctrl+C to stop\n" << endl;
    
    int frameCount = 0;
    
    try {
        while(pipeline.isRunning()) {
            auto pclData = std::dynamic_pointer_cast<dai::PointCloudData>(queue->get());
            if(!pclData) continue;
            
            auto points = pclData->getPoints();
            
            cout << "\n========================================" << endl;
            cout << "Frame " << frameCount++ << endl;
            cout << "========================================" << endl;
            cout << "Total points: " << points.size() << endl;
            
            cout << "\nBounding box (mm):" << endl;
            cout << "  X: [" << pclData->getMinX() << ", " << pclData->getMaxX() << "]" << endl;
            cout << "  Y: [" << pclData->getMinY() << ", " << pclData->getMaxY() << "]" << endl;
            cout << "  Z: [" << pclData->getMinZ() << ", " << pclData->getMaxZ() << "]" << endl;
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } catch(const std::exception& e) {
        cout << "\nException: " << e.what() << endl;
    }
    
    pipeline.stop();
    cout << "\nStopped." << endl;
    
    return 0;
}