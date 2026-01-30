#pragma once

#include <depthai/pipeline/DeviceNode.hpp>
#include <depthai/pipeline/Subnode.hpp>
#include <depthai/pipeline/datatype/DynamicCalibrationControl.hpp>
#include <depthai/pipeline/datatype/DynamicCalibrationResults.hpp>
#include <depthai/pipeline/node/Camera.hpp>
#include <depthai/pipeline/node/DynamicCalibrationNode.hpp>
#include <depthai/pipeline/node/Gate.hpp>
#include <depthai/properties/DynamicCalibrationWorkerProperties.hpp>

namespace spdlog {
class async_logger;
}
namespace dai {
namespace node {

class DynamicCalibrationWorker : public DeviceNodeCRTP<DeviceNode, DynamicCalibrationWorker, DynamicCalibrationWorkerProperties>, public HostRunnable {
   private:
    Subnode<node::DynamicCalibration> dynamicCalibration{*this, "dynamicCalibration"};

    Subnode<node::Sync> sync{*this, "sync"};

    Subnode<node::Gate> gate{*this, "gate"};

    InputMap& inputs = sync->inputs;

    std::string leftInputName = "left";
    std::string rightInputName = "right";

   public:
    using DCC = dai::DynamicCalibrationControl;

    std::shared_ptr<DynamicCalibrationWorker> build(const std::shared_ptr<Camera> cameraLeft, const std::shared_ptr<Camera> cameraRight);

    constexpr static const char* NAME = "DynamicCalibrationWorker";
    using DeviceNodeCRTP::DeviceNodeCRTP;

    ~DynamicCalibrationWorker() override;

    std::shared_ptr<DynamicCalibrationWorkerConfig> initialConfig = std::make_shared<DynamicCalibrationWorkerConfig>();

    void run() override;

    void setRunOnHost(bool runOnHost);

    bool runOnHost() const override;

    void buildInternal() override;

   protected:
    Properties& getProperties() override;

   private:
#ifndef DEPTHAI_INTERNAL_DEVICE_BUILD_RVC4
    /**
     * Input left image
     */
    Input& left = inputs[leftInputName];

    /**
     * Input right image
     */
    Input& right = inputs[rightInputName];

    InputMap& syncedInput = sync->inputs;
#endif

    void buildInternalQueues() override;

    bool checkCalibration();

    void updateCalibration();

    void runContinuousMode();

    void runOnStartMode();

    bool isNewCalibrationOK(std::shared_ptr<dai::DynamicCalibrationResult> calibrationResult);

    DynamicCalibrationWorkerProperties properties;

    std::shared_ptr<dai::InputQueue> gateControlQueue;
    std::shared_ptr<dai::InputQueue> dynamicCalibrationCommandQueue;
    std::shared_ptr<dai::MessageQueue> dynamicCalibrationCalibrationQueue;

    bool runOnHostVar = true;

    std::shared_ptr<::spdlog::async_logger> logger;
};

}  // namespace node
}  // namespace dai
