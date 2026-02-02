#include "depthai/pipeline/node/PointCloud.hpp"

#include <chrono>
#include <future>
#include <thread>

#include "depthai/pipeline/datatype/ImgFrame.hpp"
#include "depthai/pipeline/datatype/PointCloudData.hpp"
#include "depthai/common/DepthUnit.hpp"
#include "pipeline/ThreadedNodeImpl.hpp"
#include "device/CalibrationHandler.hpp"

#ifdef DEPTHAI_ENABLE_KOMPUTE
    #include "depthai/shaders/depth2pointcloud.hpp"
#endif
#include "utility/PimplImpl.hpp"

namespace dai {
namespace node {

// PointCloud::Impl method implementations
void PointCloud::Impl::setLogger(std::shared_ptr<spdlog::logger> log) {
    logger = log;
}

void PointCloud::Impl::computePointCloudDense(const uint8_t* depthData, std::vector<Point3f>& points) {
    if(!intrinsicsSet) {
        throw std::runtime_error("Intrinsics not set");
    }
    
    points.resize(size);
    
    switch(computeMethod) {
        case ComputeMethod::CPU:
            computePointCloudDenseCPU(depthData, points);
            break;
        case ComputeMethod::CPU_MT:
            computePointCloudDenseCPUMT(depthData, points);
            break;
        case ComputeMethod::GPU:
            computePointCloudDenseGPU(depthData, points);
            break;
    }
}

void PointCloud::Impl::applyTransformation(std::vector<Point3f>& points) {
    if(!hasExtrinsics) {
        if(logger) {
            logger->debug("No extrinsics set, skipping transformation");
        }
        return;
    }
    
    if(logger) {
        logger->info("Applying coordinate system transformation");
    }
    
    switch(computeMethod) {
        case ComputeMethod::CPU:
            transformPointsCPU(points);
            break;
        case ComputeMethod::CPU_MT:
            transformPointsCPU(points);
            break;
        case ComputeMethod::GPU:
            transformPointsCPU(points);
            break;
    }
}

std::vector<Point3f> PointCloud::Impl::filterValidPoints(const std::vector<Point3f>& densePoints) {
    std::vector<Point3f> sparsePoints;
    sparsePoints.reserve(densePoints.size() / 2);
    
    for(const auto& p : densePoints) {
        if(p.z > 0.0f) {
            sparsePoints.push_back(p);
        }
    }
    
    return sparsePoints;
}

void PointCloud::Impl::setLengthUnit(dai::LengthUnit lengthUnit) {
    // Check if unit actually changed
    bool unitChanged = (targetLengthUnit != lengthUnit);
    
    targetLengthUnit = lengthUnit;
    lengthUnitMultiplier = getLengthUnitMultiplier(lengthUnit);
    
    // Depth values from sensor are in millimeters (uint16_t raw values)
    // scaleFactor converts from mm to target unit
    constexpr float MM_MULTIPLIER = getLengthUnitMultiplier(LengthUnit::MILLIMETER);
    scaleFactor = lengthUnitMultiplier / MM_MULTIPLIER;
    
    if(logger) {
        logger->info("Set length unit: multiplier={}, scaleFactor={} (mm->target), unit changed: {}", 
                    lengthUnitMultiplier, scaleFactor, unitChanged);
    }
}

void PointCloud::Impl::useCPU() {
    computeMethod = ComputeMethod::CPU;
}

void PointCloud::Impl::useCPUMT(uint32_t numThreads) {
    threadNum = numThreads;
    computeMethod = ComputeMethod::CPU_MT;
}

void PointCloud::Impl::useGPU(uint32_t device) {
    initializeGPU(device);
}

void PointCloud::Impl::setIntrinsics(float fx, float fy, float cx, float cy, unsigned int width, unsigned int height) {
    this->fx = fx;
    this->fy = fy;
    this->cx = cx;
    this->cy = cy;
    this->width = width;
    this->height = height;
    size = this->width * this->height;
    intrinsicsSet = true;
}

void PointCloud::Impl::setExtrinsics(const std::vector<std::vector<float>>& transformMatrix) {
    if(transformMatrix.size() != 4 || transformMatrix[0].size() != 4) {
        throw std::runtime_error("Transformation matrix must be 4x4");
    }
    extrinsics = transformMatrix;
    hasExtrinsics = true;
    
    if(logger) {
        logger->info("Extrinsics transformation matrix set:");
        for(size_t i = 0; i < 4; i++) {
            logger->info("  [{:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}]", 
                        extrinsics[i][0], extrinsics[i][1], extrinsics[i][2], extrinsics[i][3]);
        }
    }
}

void PointCloud::Impl::initializeGPU(uint32_t device) {
#ifdef DEPTHAI_ENABLE_KOMPUTE
    mgr = std::make_shared<kp::Manager>(device);
    shader = std::vector<uint32_t>(shaders::DEPTH2POINTCLOUD_COMP_SPV.begin(), shaders::DEPTH2POINTCLOUD_COMP_SPV.end());
    computeMethod = ComputeMethod::GPU;
#else
    (void)device;
    throw std::runtime_error("Kompute not enabled in this build");
#endif
}

void PointCloud::Impl::transformPointsCPU(std::vector<Point3f>& points) {
    // Both points and extrinsics translations are in the same unit (target unit)
    // No conversion needed - just apply the transformation directly
    
    if(logger) {
        logger->debug("Applying transformation to {} points", points.size());
    }
    
    size_t transformedCount = 0;
    for(auto& p : points) {
        if(p.z > 0.0f) {
            // Standard 4x4 transformation: R*p + t
            float x = extrinsics[0][0] * p.x + extrinsics[0][1] * p.y + extrinsics[0][2] * p.z + extrinsics[0][3];
            float y = extrinsics[1][0] * p.x + extrinsics[1][1] * p.y + extrinsics[1][2] * p.z + extrinsics[1][3];
            float z = extrinsics[2][0] * p.x + extrinsics[2][1] * p.y + extrinsics[2][2] * p.z + extrinsics[2][3];
            
            p.x = x;
            p.y = y;
            p.z = z;
            transformedCount++;
        }
    }
    
    if(logger) {
        logger->debug("Transformed {} valid points (z > 0)", transformedCount);
    }
}

void PointCloud::Impl::calcPointsChunkDense(const uint8_t* depthData, std::vector<Point3f>& points, unsigned int startRow, unsigned int endRow) {
    float scale = scaleFactor;

    for(unsigned int row = startRow; row < endRow; row++) {
        unsigned int rowStart = row * width;
        for(unsigned int col = 0; col < width; col++) {
            size_t i = rowStart + col;

            uint16_t depthValue = *(reinterpret_cast<const uint16_t*>(depthData + i * 2));
            float z = static_cast<float>(depthValue) * scale;

            float xCoord = 0.0f;
            float yCoord = 0.0f;
            
            if(z > 0.0f) {
                xCoord = (col - cx) * z / fx;
                yCoord = (row - cy) * z / fy;
            }
            
            points[i] = Point3f{xCoord, yCoord, z};
        }
    }
}

void PointCloud::Impl::computePointCloudDenseCPU(const uint8_t* depthData, std::vector<Point3f>& points) {
    calcPointsChunkDense(depthData, points, 0, height);
}

void PointCloud::Impl::computePointCloudDenseCPUMT(const uint8_t* depthData, std::vector<Point3f>& points) {
    unsigned int rowsPerThread = height / threadNum;
    std::vector<std::future<void>> futures;

    auto processRows = [&](unsigned int startRow, unsigned int endRow) {
        calcPointsChunkDense(depthData, points, startRow, endRow);
    };

    for(uint32_t t = 0; t < threadNum; ++t) {
        unsigned int startRow = t * rowsPerThread;
        unsigned int endRow = (t == threadNum - 1) ? height : (startRow + rowsPerThread);
        futures.emplace_back(std::async(std::launch::async, processRows, startRow, endRow));
    }

    for(auto& f : futures) {
        f.get();
    }
}

void PointCloud::Impl::computePointCloudDenseGPU(const uint8_t* depthData, std::vector<Point3f>& points) {
#ifdef DEPTHAI_ENABLE_KOMPUTE
    std::vector<float> xyzOut;
    xyzOut.resize(size * 3);

    float scale = scaleFactor;

    std::vector<float> depthDataFloat(size);
    for(size_t i = 0; i < size; i++) {
        uint16_t depthValue = *(reinterpret_cast<const uint16_t*>(depthData + i * 2));
        depthDataFloat[i] = static_cast<float>(depthValue);
    }

    std::vector<float> intrinsics = {fx, fy, cx, cy, scale, static_cast<float>(width), static_cast<float>(height)};

    if(!tensorsInitialized) {
        depthTensor = mgr->tensor(depthDataFloat);
        intrinsicsTensor = mgr->tensor(intrinsics);
        xyzTensor = mgr->tensor(xyzOut);
        tensorsInitialized = true;
    } else {
        depthTensor->setData(depthDataFloat);
    }
    
    if(!algoInitialized) {
        tensors.emplace_back(depthTensor);
        tensors.emplace_back(intrinsicsTensor);
        tensors.emplace_back(xyzTensor);
        algo = mgr->algorithm(tensors, shader);
        algoInitialized = true;
    }
    
    mgr->sequence()->record<kp::OpSyncDevice>(tensors)->record<kp::OpAlgoDispatch>(algo)->record<kp::OpSyncLocal>(tensors)->eval();
    
    xyzOut = xyzTensor->vector<float>();
    
    for(size_t i = 0; i < size; i++) {
        points[i].x = xyzOut[i * 3 + 0];
        points[i].y = xyzOut[i * 3 + 1];
        points[i].z = xyzOut[i * 3 + 2];
    }
#else
    (void)depthData;
    (void)points;
    throw std::runtime_error("Kompute not enabled in this build");
#endif
}

// PointCloud main class implementations
PointCloud::PointCloud() 
    : pimplPointCloud() {
}

PointCloud::~PointCloud() = default;

PointCloud::Properties& PointCloud::getProperties() {
    properties.initialConfig = *initialConfig;
    return properties;
}

void PointCloud::setNumFramesPool(int numFramesPool) {
    properties.numFramesPool = numFramesPool;
}

void PointCloud::setRunOnHost(bool runOnHost) {
    runOnHostVar = runOnHost;
}

bool PointCloud::runOnHost() const {
    return runOnHostVar;
}

void PointCloud::setLengthUnit(dai::LengthUnit lengthUnit) {
    pimplPointCloud->setLengthUnit(lengthUnit);
    
    // If already initialized, we need to re-fetch extrinsics with the new unit
    if(initialized) {
        if(pimpl->logger) {
            pimpl->logger->warn("Length unit changed after initialization - extrinsics will be re-fetched");
        }
        initialized = false;  // Force re-initialization on next frame
    }
}

void PointCloud::useCPU() {
    pimplPointCloud->useCPU();
}

void PointCloud::useCPUMT(uint32_t numThreads) {
    pimplPointCloud->useCPUMT(numThreads);
}

void PointCloud::useGPU(uint32_t device) {
    pimplPointCloud->useGPU(device);
}

void PointCloud::setTargetCoordinateSystem(CameraBoardSocket targetCamera, bool useSpecTranslation) {
    coordSystemType = CoordinateSystemType::CAMERA_SOCKET;
    targetCameraSocket = targetCamera;
    this->useSpecTranslation = useSpecTranslation;
}

void PointCloud::setTargetCoordinateSystem(HousingCoordinateSystem housingCS, bool useSpecTranslation) {
    coordSystemType = CoordinateSystemType::HOUSING;
    targetHousingCS = housingCS;
    this->useSpecTranslation = useSpecTranslation;
}

void PointCloud::keepPointCloudOrganized() {
    keepOrganized = true;
}

void PointCloud::initialize(std::shared_ptr<ImgFrame> depthFrame) {
    pimpl->logger->debug("PointCloud::initialize() called");
    pimplPointCloud->setLogger(pimpl->logger);
    
    auto width = depthFrame->getWidth();
    auto height = depthFrame->getHeight();
    
    // Get camera intrinsics from the depth frame
    auto intrinsics = depthFrame->transformation.getIntrinsicMatrix();
    float fx = intrinsics[0][0];
    float fy = intrinsics[1][1];
    float cx = intrinsics[0][2];
    float cy = intrinsics[1][2];
    
    pimpl->logger->debug("Setting intrinsics: fx={}, fy={}, cx={}, cy={}, size={}x{}", 
                        fx, fy, cx, cy, width, height);
    
    pimplPointCloud->setIntrinsics(fx, fy, cx, cy, width, height);
    
    // Get source camera socket from depth frame
    auto srcCamera = static_cast<CameraBoardSocket>(depthFrame->getInstanceNum());
    
    pimpl->logger->debug("Source camera: {}, Coord system type: {}", 
                        toString(srcCamera), (int)coordSystemType);
    
    // Get calibration handler and setup extrinsics if needed
    auto calibHandler = device->getCalibration();
    std::vector<std::vector<float>> transformMatrix;
    
    // Use the same unit as the point cloud for extrinsics
    // This way both points and translations are in the same unit
    auto extrinsicsUnit = pimplPointCloud->targetLengthUnit;
    
    switch(coordSystemType) {
        case CoordinateSystemType::CAMERA_SOCKET:
            pimpl->logger->info("Using CAMERA_SOCKET transformation from {} to {}, unit: {}", 
                                toString(srcCamera), toString(targetCameraSocket), static_cast<int>(extrinsicsUnit));
            transformMatrix = calibHandler.getCameraExtrinsics(srcCamera, targetCameraSocket, useSpecTranslation, extrinsicsUnit);
            pimplPointCloud->setExtrinsics(transformMatrix);
            break;
            
        case CoordinateSystemType::HOUSING:
            pimpl->logger->info("Using HOUSING transformation from {} to housing {}, unit: {}", 
                                toString(srcCamera), static_cast<int>(targetHousingCS), static_cast<int>(extrinsicsUnit));
            transformMatrix = calibHandler.getHousingCalibration(srcCamera, targetHousingCS, useSpecTranslation, extrinsicsUnit);
            
            pimpl->logger->info("Retrieved housing transformation matrix:");
            for(size_t i = 0; i < transformMatrix.size(); i++) {
                pimpl->logger->info("  [{:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}]", 
                            transformMatrix[i][0], transformMatrix[i][1], transformMatrix[i][2], transformMatrix[i][3]);
            }
            
            pimplPointCloud->setExtrinsics(transformMatrix);
            break;
            
        case CoordinateSystemType::NONE:
            pimpl->logger->info("PointCloud: No coordinate system transformation applied (identity)");
            break;
    }
    
    initialized = true;
    pimpl->logger->info("PointCloud::initialize() completed");
}

void PointCloud::run() {
    pimpl->logger->info("PointCloud node started");

    while(mainLoop()) {
        if(!outputPointCloud.getQueueConnections().empty() || !outputPointCloud.getConnections().empty()) {
            
            std::shared_ptr<ImgFrame> depthFrame = nullptr;
            
            {
                auto blockEvent = this->inputBlockEvent();
                depthFrame = inputDepth.get<ImgFrame>();
            }
            
            if(!initialized) {
                initialize(depthFrame);
            }
            
            auto cameraInstanceNum = depthFrame->getInstanceNum();
            auto width = depthFrame->getWidth();
            auto height = depthFrame->getHeight();
            
            // Step 1: Compute dense point cloud directly into final storage
            // Allocate once for the maximum size needed
            std::vector<Point3f> points;
            const auto* depthData = depthFrame->getData().data();
            
            if(keepOrganized) {
                // For organized output, compute directly into final storage
                pimplPointCloud->computePointCloudDense(depthData, points);
                pimplPointCloud->applyTransformation(points);
            } else {
                // For sparse output, we need to filter, so compute dense first
                std::vector<Point3f> densePoints;
                pimplPointCloud->computePointCloudDense(depthData, densePoints);
                pimplPointCloud->applyTransformation(densePoints);
                points = pimplPointCloud->filterValidPoints(densePoints);
            }
            
            // Calculate bounding box (skip z=0 points)
            float minX = 0.0f, minY = 0.0f, minZ = 0.0f;
            float maxX = 0.0f, maxY = 0.0f, maxZ = 0.0f;
            
            bool foundValidPoint = false;
            for(const auto& p : points) {
                if(p.z > 0.0f) {
                    if(!foundValidPoint) {
                        minX = maxX = p.x;
                        minY = maxY = p.y;
                        minZ = maxZ = p.z;
                        foundValidPoint = true;
                    } else {
                        minX = std::min(minX, p.x);
                        minY = std::min(minY, p.y);
                        minZ = std::min(minZ, p.z);
                        maxX = std::max(maxX, p.x);
                        maxY = std::max(maxY, p.y);
                        maxZ = std::max(maxZ, p.z);
                    }
                }
            }
            
            // Create PointCloudData and move points
            auto pc = std::make_shared<PointCloudData>();
            pc->setTimestamp(depthFrame->getTimestamp());
            pc->setTimestampDevice(depthFrame->getTimestampDevice());
            pc->setSequenceNum(depthFrame->getSequenceNum());
            pc->setInstanceNum(cameraInstanceNum);
            pc->setMinX(minX);
            pc->setMinY(minY);
            pc->setMinZ(minZ);
            pc->setMaxX(maxX);
            pc->setMaxY(maxY);
            pc->setMaxZ(maxZ);
            
            // Move points instead of copying
            pc->setPoints(std::move(points));
            
            // Set width and height based on organization
            if(keepOrganized) {
                pc->setWidth(width);
                pc->setHeight(height);
            } else {
                pc->setWidth(static_cast<unsigned int>(pc->getPoints().size()));
                pc->setHeight(1);
            }
            
            {
                auto blockEvent = this->outputBlockEvent();
                outputPointCloud.send(pc);
                
                if(!passthroughDepth.getQueueConnections().empty() || !passthroughDepth.getConnections().empty()) {
                    passthroughDepth.send(depthFrame);
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    pimpl->logger->info("PointCloud node stopped");
}

}  // namespace node
}  // namespace dai

// Explicit template instantiation for Pimpl
template class dai::Pimpl<dai::node::PointCloud::Impl>;
