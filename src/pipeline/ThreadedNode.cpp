#include "depthai/pipeline/ThreadedNode.hpp"

#include <spdlog/spdlog.h>

#include <memory>
#include <optional>
#include <thread>

#include "depthai/utility/PipelineEventDispatcher.hpp"
#include "pipeline/ThreadedNodeImpl.hpp"
#include "utility/Environment.hpp"
#include "utility/ErrorMacros.hpp"
#include "utility/Logging.hpp"
#include "utility/Platform.hpp"

namespace dai {

ThreadedNode::ThreadedNode() {
    pimpl = std::make_unique<Impl>();
    auto level = spdlog::level::warn;
    auto envLevel = utility::getEnvAs<std::string>("DEPTHAI_LEVEL", "");
    if(!envLevel.empty()) {
        level = Logging::parseLevel(envLevel);
    }
    pimpl->logger->set_level(level);
    pipelineEventDispatcher = std::make_unique<utility::PipelineEventDispatcher>(this->id, &pipelineEventOutput);
}

ThreadedNode::~ThreadedNode() = default;

void ThreadedNode::start() {
    // A node should not be started if it is already running
    // We would be creating multiple threads for the same node
    DAI_CHECK_V(!isRunning(), "Node with id {} is already running. Cannot start it again. Node name: {}", id, getName());

    onStart();
    // Start the thread
    running = true;
    thread = std::thread([this]() {
        try {
            run();
        } catch(const MessageQueue::QueueException& ex) {
            // catch the exception and stop the node
            auto expStr = fmt::format("Node stopped with a queue exception: {}", ex.what());
            if(pimpl->logger) {
                pimpl->logger->trace(expStr);
            } else {
                spdlog::trace(expStr);
            }
            running = false;
        } catch(const std::runtime_error& ex) {
            auto expStr = fmt::format("Node threw exception, stopping the node. Exception message: {}", ex.what());
            if(pimpl->logger) {
                pimpl->logger->error(expStr);
            } else {
                spdlog::error(expStr);
            }
            running = false;
            stopPipeline();
        }
    });
    platform::setThreadName(thread, fmt::format("{}({})", getName(), id));
}

void ThreadedNode::wait() {
    // The node's own thread can get here: when run() throws, the catch handler calls
    // stopPipeline(), and if the temporary Pipeline it holds is the last owner,
    // ~PipelineImpl -> wait() runs on this very thread. Joining the current thread
    // would throw std::system_error (resource deadlock avoided) and terminate the
    // process, so only join from other threads; JoiningThread detaches on
    // self-destruction to keep the teardown safe.
    if(thread.joinable() && thread.get_id() != std::this_thread::get_id()) thread.join();
}

void ThreadedNode::stop() {
    onStop();
    // TBD
    // Sets running to false
    running = false;
    // closes all the queueus, then waits for the thread to join
    for(auto& in : getInputRefs()) {
        in->close();
    }
    // for(auto& rout : getOutputRefs()) {
    // }
    // wait();
}

void ThreadedNode::setLogLevel(dai::LogLevel level) {
    pimpl->logger->set_level(logLevelToSpdlogLevel(level, spdlog::level::warn));
}

dai::LogLevel ThreadedNode::getLogLevel() const {
    return spdlogLevelToLogLevel(pimpl->logger->level(), LogLevel::WARN);
}

bool ThreadedNode::isRunning() const {
    return running;
}

bool ThreadedNode::mainLoop() {
    this->pipelineEventDispatcher->pingMainLoopEvent();
    return isRunning();
}

utility::PipelineEventDispatcherInterface::BlockPipelineEvent ThreadedNode::blockEvent(PipelineEvent::Type type, const std::string& source, bool startNow) {
    return pipelineEventDispatcher->blockEvent(type, source, std::nullopt, startNow);
}
utility::PipelineEventDispatcherInterface::BlockPipelineEvent ThreadedNode::inputBlockEvent(bool startNow) {
    return pipelineEventDispatcher->inputBlockEvent(startNow);
}
utility::PipelineEventDispatcherInterface::BlockPipelineEvent ThreadedNode::outputBlockEvent(bool startNow) {
    return pipelineEventDispatcher->outputBlockEvent(startNow);
}

}  // namespace dai
