#pragma once

#include <thread>

namespace dai {

class JoiningThread : private std::thread {
   public:
    using std::thread::thread;
    JoiningThread() = default;
    // Create an empty copy constructor
    JoiningThread(const JoiningThread&) : JoiningThread() {}
    JoiningThread(JoiningThread&&) = default;
    JoiningThread& operator=(JoiningThread&& thr) {
        joinUnlessSelf();
        swap(thr);
        return *this;
    };
    ~JoiningThread() {
        joinUnlessSelf();
    }
    JoiningThread(std::thread t) : std::thread(std::move(t)) {}

    using std::thread::detach;
    using std::thread::get_id;
    using std::thread::hardware_concurrency;
    using std::thread::join;
    using std::thread::joinable;
    using std::thread::native_handle;

    void swap(JoiningThread& x) {
        std::thread::swap(x);
    }

   private:
    void joinUnlessSelf() {
        if(!joinable()) return;
        if(get_id() == std::this_thread::get_id()) {
            // Joining the current thread would throw std::system_error (resource
            // deadlock avoided) and terminate the process. This happens when the
            // owning object is destroyed on this very thread, e.g. a ThreadedNode
            // whose run() threw: its stopPipeline() call may hold the last
            // Pipeline reference, so ~PipelineImpl destroys the node on the
            // node's own thread. Detach instead - past this point the thread
            // only unwinds its stack and touches no freed state.
            detach();
        } else {
            join();
        }
    }
};

inline void swap(JoiningThread& x, JoiningThread& y) {
    x.swap(y);
}

}  // namespace dai
