#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImplContext.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void TcpCacheStoreServiceImplContext::loadBlockOnTcp(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    if (done_run_) {
        // already done run, most likely timeout, no need load
        return;
    }

    if (!ok) {
        // request been canceled in cache store, just failed
        runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
        return;
    }

    for (auto& block : blocks) {
        auto unloaded_block_info = getAndEraseUnLoadedBlock(block->key);
        if (unloaded_block_info == nullptr) {
            // block already loaded
            continue;
        }

        CacheStoreBlockPartition partition;
        if (!resolveCacheStoreBlockPartition(
                block->len, unloaded_block_info->len(), partition_count_, partition_id_, partition)) {
            RTP_LLM_LOG_WARNING("cache store service load block not match expect block len, key: %s, len %d, "
                                "partition count %d, partition id %d, peer len %d, peer is %s",
                                block->key.c_str(),
                                block->len,
                                partition_count_,
                                partition_id_,
                                unloaded_block_info->len(),
                                peer_ip_.c_str());
            runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ);
            return;
        }

        if (!writeResponseBlock(block, unloaded_block_info)) {
            runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL);
            return;
        }
        ++write_cnt_;
    }

    if (write_cnt_ == total_block_count_) {
        runSuccess(false);
    }
}

bool TcpCacheStoreServiceImplContext::writeResponseBlock(const std::shared_ptr<BlockBuffer>&     block,
                                                         const std::shared_ptr<BlockBufferInfo>& peer_block) {
    std::lock_guard<std::mutex> lock(response_mutex_);
    if (response_ == nullptr) {
        // try write response while already done
        return false;
    }

    CacheStoreBlockPartition partition;
    if (!resolveCacheStoreBlockPartition(block->len, peer_block->len(), partition_count_, partition_id_, partition)) {
        RTP_LLM_LOG_WARNING(
            "cache store service load block not match expect block len, key: %s, len %d, partition count %d, "
            "partition id %d, peer len %d, peer is %s",
            block->key.c_str(),
            block->len,
            partition_count_,
            partition_id_,
            peer_block->len(),
            peer_ip_.c_str());
        return false;
    }

    auto* block_info = response_->add_blocks();
    block_info->set_key(block->key);
    block_info->set_len(partition.len);
    auto block_content = block_info->mutable_content();
    block_content->assign(
        std::shared_ptr<const char>(
            block->addr,
            reinterpret_cast<const char*>((int64_t)(block->addr.get()) + partition.len * partition.partition_id)),
        size_t(partition.len));
    return true;
}

}  // namespace rtp_llm
