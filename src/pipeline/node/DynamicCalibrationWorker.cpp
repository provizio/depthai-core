#include "depthai/pipeline/node/DynamicCalibrationWorker.hpp"

#include <pipeline/ThreadedNodeImpl.hpp>

#include "depthai/pipeline/InputQueue.hpp"

namespace dai {
namespace node {

DynamicCalibrationWorker::~DynamicCalibrationWorker() = default;

DynamicCalibrationWorker::Properties& DynamicCalibrationWorker::getProperties() {
    return properties;
}

void DynamicCalibrationWorker::setRunOnHost(bool runOnHost) {
    runOnHostVar = runOnHost;
}

std::shared_ptr<DynamicCalibrationWorker> DynamicCalibrationWorker::build(const std::shared_ptr<Camera> cameraLeft, const std::shared_ptr<Camera> cameraRight) {
    sync->setRunOnHost(false);
    gate->setRunOnHost(false);
    auto outputCameraLeft = cameraLeft->requestIspOutput(20);
    auto outputCameraRight = cameraRight->requestIspOutput(20);
    outputCameraLeft->link(left);
    outputCameraRight->link(right);
    return std::static_pointer_cast<DynamicCalibrationWorker>(shared_from_this());
}

/**
 * Check if the node is set to run on host
 */
bool DynamicCalibrationWorker::runOnHost() const {
    return runOnHostVar;
}

void DynamicCalibrationWorker::loadData(unsigned int numImages) {
    for(unsigned int i = 0; i < numImages; i++) {
        dynamicCalibrationCommandQueue->send(DCC::loadImage());
        coverageQueue->get<dai::CoverageData>();  // wait until the data are loaded
    }
}

std::shared_ptr<dai::CalibrationMetrics> DynamicCalibrationWorker::getMetrics(std::shared_ptr<dai::CalibrationHandler> calibration) {
    dynamicCalibrationCommandQueue->send(DCC::computeCalibrationMetrics(*calibration));
    return metricsQueue->get<dai::CalibrationMetrics>();
}

void DynamicCalibrationWorker::buildInternalQueues() {
    // TODO check left right inputs
    dynamicCalibrationQueue = dynamicCalibration->calibrationOutput.createOutputQueue();
    coverageQueue = dynamicCalibration->coverageOutput.createOutputQueue();
    metricsQueue = dynamicCalibration->metricsOutput.createOutputQueue();
    dynamicCalibrationCommandQueue = dynamicCalibration->inputControl.createInputQueue();
    gateControlQueue = gate->inputControl.createInputQueue();
    sync->out.link(gate->input);
    gate->output.link(dynamicCalibration->syncInput);
}

void DynamicCalibrationWorker::buildInternal() {
    logger = pimpl->logger;
}

std::shared_ptr<dai::CalibrationHandler> DynamicCalibrationWorker::getNewCalibration(unsigned int maxNumIteration) {
    gateControlQueue->send(dai::GateControl::openGate());
    for(unsigned int i = 0; i < maxNumIteration; i++) {
        dynamicCalibrationCommandQueue->send(DCC::startCalibration());
        bool dataCollected = false;
        while(!dataCollected) {
            auto dynCalibrationResult = dynamicCalibrationQueue->get<dai::DynamicCalibrationResult>();
            if(dynCalibrationResult->calibrationData) {
                dataCollected = true;
                if(dynCalibrationResult->calibrationData.value().dataQuality > initialConfig->dataQualityThreshold) {
                    return std::make_shared<dai::CalibrationHandler>(dynCalibrationResult->calibrationData.value().newCalibration);
                }
            }
        }
        dynamicCalibrationCommandQueue->send(DCC::resetData());
    }
    gateControlQueue->send(dai::GateControl::closeGate());
    return nullptr;
}

bool DynamicCalibrationWorker::recalibrate(unsigned int& numIterations, std::shared_ptr<dai::CalibrationHandler> calibration) {
    if(numIterations > initialConfig->maxIterations) return false;
    dynamicCalibrationCommandQueue->send(DCC::resetData());
    loadData(5);
    auto metrics = getMetrics(calibration);
    logger->info("Iteration = {}", numIterations);
    logger->info("          dataQuality = {}", metrics->dataQuality);
    logger->info("          calibrationConfidence = {}", metrics->calibrationConfidence);
    if(metrics->dataQuality > initialConfig->dataQualityThreshold) {
        if(metrics->calibrationConfidence > initialConfig->calibrationConfidenceThreshold) {
            device->flashCalibration(*calibration);
            return true;
        } else {
            auto newCalibration = getNewCalibration(initialConfig->maxIterations);
            return recalibrate(++numIterations, newCalibration);
        }
    } else {
        return recalibrate(++numIterations, calibration);
    }
}

bool DynamicCalibrationWorker::updateCalibration() {
    auto calibration = std::make_shared<dai::CalibrationHandler>(device->getCalibration());
    unsigned int numIterations = 0;
    return recalibrate(numIterations, calibration);
}

void DynamicCalibrationWorker::runContinuousMode() {
    while(isRunning()) {
        updateCalibration();
        // trigger also by a low fillrate?
        std::this_thread::sleep_for(std::chrono::seconds(initialConfig->sleepingTime));
    }
}

void DynamicCalibrationWorker::runOnStartMode() {
    updateCalibration();
}

void DynamicCalibrationWorker::run() {
    logger->info("DynamicCalibrationWorker started to work!");
    switch(initialConfig->mode) {
        case DynamicCalibrationWorkerConfig::Mode::CONTINUOUS:
            logger->info("Running continuous mode.");
            runContinuousMode();
            break;
        case DynamicCalibrationWorkerConfig::Mode::ON_START:
            logger->info("Running on-start mode.");
            runOnStartMode();
            logger->info("On-start mode finished.");
            // properly destroy the Node ??
            break;
    }
}

}  // namespace node
}  // namespace dai
