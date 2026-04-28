#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <chrono>
#include <cstring>
#include <thread>

#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace rtp_llm {

namespace {

std::string makeUdsPath(const std::string& base, int rank) {
    return base + "_" + std::to_string(rank) + ".sock";
}

// Loop until `nbytes` written or fatal error. Returns -1 on error.
ssize_t writeAll(int fd, const void* buf, std::size_t nbytes) {
    const char* p    = static_cast<const char*>(buf);
    std::size_t left = nbytes;
    while (left > 0) {
        ssize_t n = ::write(fd, p, left);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (n == 0) {
            return -1;
        }
        p += n;
        left -= n;
    }
    return static_cast<ssize_t>(nbytes);
}

ssize_t readAll(int fd, void* buf, std::size_t nbytes) {
    char*       p    = static_cast<char*>(buf);
    std::size_t left = nbytes;
    while (left > 0) {
        ssize_t n = ::read(fd, p, left);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (n == 0) {
            // peer closed prematurely
            return -1;
        }
        p += n;
        left -= n;
    }
    return static_cast<ssize_t>(nbytes);
}

}  // namespace

CpuTpBroadcaster& CpuTpBroadcaster::instance() {
    static CpuTpBroadcaster i;
    return i;
}

CpuTpBroadcaster::~CpuTpBroadcaster() {
    for (int fd : peer_fds_) {
        if (fd >= 0) {
            ::close(fd);
        }
    }
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
    }
    if (!my_uds_path_.empty()) {
        ::unlink(my_uds_path_.c_str());
    }
}

void CpuTpBroadcaster::initialize(int tp_rank, int tp_size, const std::string& base_path) {
    std::lock_guard<std::mutex> lock(mu_);

    if (initialized_.load(std::memory_order_acquire)) {
        if (tp_rank_ == tp_rank && tp_size_ == tp_size && base_path_ == base_path) {
            return;
        }
        RTP_LLM_FAIL("CpuTpBroadcaster re-init mismatch: was rank=%d size=%d path=%s, "
                     "now rank=%d size=%d path=%s",
                     tp_rank_,
                     tp_size_,
                     base_path_.c_str(),
                     tp_rank,
                     tp_size,
                     base_path.c_str());
    }

    if (tp_size <= 1) {
        // Single-rank no-op; broadcast() short-circuits.
        tp_rank_ = tp_rank;
        tp_size_ = tp_size;
        initialized_.store(true, std::memory_order_release);
        return;
    }

    tp_rank_   = tp_rank;
    tp_size_   = tp_size;
    base_path_ = base_path;
    peer_fds_.assign(tp_size, -1);

    if (tp_rank == 0) {
        const std::string path = makeUdsPath(base_path, 0);
        // Remove any stale socket left by a previous crashed run.
        ::unlink(path.c_str());

        listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        RTP_LLM_CHECK_WITH_INFO(listen_fd_ >= 0, "CpuTpBroadcaster socket: %s", std::strerror(errno));

        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        RTP_LLM_CHECK_WITH_INFO(
            path.size() < sizeof(addr.sun_path), "CpuTpBroadcaster UDS path too long: %s", path.c_str());
        std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

        int rc = ::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
        RTP_LLM_CHECK_WITH_INFO(rc == 0, "CpuTpBroadcaster bind(%s): %s", path.c_str(), std::strerror(errno));
        my_uds_path_ = path;

        rc = ::listen(listen_fd_, tp_size - 1);
        RTP_LLM_CHECK_WITH_INFO(rc == 0, "CpuTpBroadcaster listen: %s", std::strerror(errno));

        // Accept tp_size-1 peers; learn each peer's rank from its handshake byte stream.
        for (int i = 1; i < tp_size; ++i) {
            int fd = ::accept(listen_fd_, nullptr, nullptr);
            RTP_LLM_CHECK_WITH_INFO(fd >= 0, "CpuTpBroadcaster accept: %s", std::strerror(errno));
            int     peer_rank = -1;
            ssize_t n         = readAll(fd, &peer_rank, sizeof(peer_rank));
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(peer_rank)),
                                    "CpuTpBroadcaster handshake read failed");
            RTP_LLM_CHECK_WITH_INFO(peer_rank > 0 && peer_rank < tp_size,
                                    "CpuTpBroadcaster bad peer_rank: %d (tp_size=%d)",
                                    peer_rank,
                                    tp_size);
            RTP_LLM_CHECK_WITH_INFO(peer_fds_[peer_rank] < 0, "CpuTpBroadcaster duplicate peer rank: %d", peer_rank);
            peer_fds_[peer_rank] = fd;
        }
        RTP_LLM_LOG_INFO("CpuTpBroadcaster rank 0: accepted %d peer(s) on %s", tp_size - 1, path.c_str());
    } else {
        const std::string  path = makeUdsPath(base_path, 0);
        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        RTP_LLM_CHECK_WITH_INFO(
            path.size() < sizeof(addr.sun_path), "CpuTpBroadcaster UDS path too long: %s", path.c_str());
        std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

        // Retry connect for up to ~30s — rank 0 may take time to listen, since
        // both ranks reach _create_process_groups in the same window but rank 0
        // does NCCL group creation first.
        constexpr int kMaxAttempts = 600;
        constexpr int kSleepMs     = 50;
        int           fd           = -1;
        for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
            fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
            RTP_LLM_CHECK_WITH_INFO(fd >= 0, "CpuTpBroadcaster socket: %s", std::strerror(errno));
            int rc = ::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
            if (rc == 0) {
                break;
            }
            int saved = errno;
            ::close(fd);
            fd = -1;
            if (attempt + 1 == kMaxAttempts) {
                RTP_LLM_FAIL("CpuTpBroadcaster connect(%s) failed after %d attempts: %s",
                             path.c_str(),
                             kMaxAttempts,
                             std::strerror(saved));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMs));
        }

        // Send our rank so the server can index peer_fds_ correctly.
        int     my_rank = tp_rank;
        ssize_t n       = writeAll(fd, &my_rank, sizeof(my_rank));
        RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(my_rank)), "CpuTpBroadcaster handshake write failed");

        peer_fds_[0] = fd;
        RTP_LLM_LOG_INFO("CpuTpBroadcaster rank %d: connected to rank 0 at %s", tp_rank, path.c_str());
    }

    initialized_.store(true, std::memory_order_release);
}

void CpuTpBroadcaster::broadcast(void* buf, std::size_t nbytes, int root) {
    RTP_LLM_CHECK_WITH_INFO(initialized_.load(std::memory_order_acquire),
                            "CpuTpBroadcaster::broadcast called before initialize");
    if (tp_size_ <= 1 || nbytes == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(root == 0, "CpuTpBroadcaster supports only root=0 (star topology); got %d", root);

    if (tp_rank_ == 0) {
        for (int k = 1; k < tp_size_; ++k) {
            ssize_t n = writeAll(peer_fds_[k], buf, nbytes);
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(nbytes),
                                    "CpuTpBroadcaster write to rank %d (%zu bytes) failed: %s",
                                    k,
                                    nbytes,
                                    std::strerror(errno));
        }
    } else {
        ssize_t n = readAll(peer_fds_[0], buf, nbytes);
        RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(nbytes),
                                "CpuTpBroadcaster read from rank 0 (%zu bytes) failed: %s",
                                nbytes,
                                std::strerror(errno));
    }
}

}  // namespace rtp_llm
