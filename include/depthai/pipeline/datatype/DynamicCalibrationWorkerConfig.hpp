#pragma once

#include "depthai/pipeline/datatype/Buffer.hpp"

namespace dai {

struct DynamicCalibrationWorkerConfig : public Buffer {
    DynamicCalibrationWorkerConfig() = default;
    virtual ~DynamicCalibrationWorkerConfig();

    enum Mode : int {
        ON_START = 1,
        CONTINUOUS = 2,
    };

    Mode mode = Mode::ON_START;

    int sleepingTime = 600;  // 10 minutes

    float calibrationConfidenceThreshold = 0.5;

    float dataQualityThreshold = 0.5;

    unsigned int maxIterations = 3;

    void serialize(std::vector<std::uint8_t>& metadata, DatatypeEnum& datatype) const override;

    DEPTHAI_SERIALIZE(DynamicCalibrationWorkerConfig, mode, sleepingTime, calibrationConfidenceThreshold, dataQualityThreshold, maxIterations);
};

}  // namespace dai
