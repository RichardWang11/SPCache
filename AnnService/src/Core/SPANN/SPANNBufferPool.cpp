//
// Created by Junli on 25/07/09.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names
#include "inc/Core/SPANN/SPANNBufferPool.h"
#include "inc/Helper/CommonHelper.h"
#include <cstring>
#include <algorithm>
#include <cstdlib>

namespace SPTAG {
namespace SPANN {

SPANNBufferPool::SPANNBufferPool(size_t maxSizeBytes) : 
    maxSize(maxSizeBytes), 
    currentSize(0), 
    totalAccess(0), 
    cacheHits(0),
    lruHead(nullptr),
    lruTail(nullptr) 
{
    LOG(Helper::LogLevel::LL_Info, "Initialized SPANN Buffer Pool with max size: %zu bytes\n", maxSize);
}

SPANNBufferPool::~SPANNBufferPool() {
    try {
        std::unique_lock<std::shared_mutex> lock(mutex);
        
        
// 清理所有缓存条目
        for (auto& pair : cache) {
            PostingListEntry* entry = pair.second;
            if (entry && entry->data) {
                free(entry->data);
                entry->data = nullptr;
            }
            delete entry;
        }
        cache.clear();
        
    } catch (...) {
    }
}

void* SPANNBufferPool::get(int64_t listID, size_t& size) {
    // std::shared_lock<std::shared_mutex> lock(mutex);
    // 下面方法写锁，线程安全
    std::unique_lock<std::shared_mutex> lock(mutex);
    totalAccess++;
    
    auto it = cache.find(listID);
    if (it == cache.end()) {
        return nullptr; // 缓存未命中
    }
    
    // 缓存命中
    cacheHits++;
    PostingListEntry* entry = it->second;
    entry->lastAccessTime = std::chrono::steady_clock::now().time_since_epoch().count();
    entry->accessCount++;
    
    moveToHead(entry);
    
    size = entry->size;
    return entry->data;
}

void SPANNBufferPool::put(int64_t listID, void* data, size_t size) {
    if (size > maxSize || size == 0 || data == nullptr) {
        // 单个条目大于缓存总大小或无效数据，不缓存
        LOG(Helper::LogLevel::LL_Debug, "Skipping cache put for listID %lld: invalid size %zu or data\n", listID, size);
        return;
    }
    
    std::unique_lock<std::shared_mutex> lock(mutex);
    
 // 检查是否已存在
    auto it = cache.find(listID);
    if (it != cache.end()) {
        // 更新现有条目
        PostingListEntry* entry = it->second;
        
        // 验证新数据与旧数据的大小
        if (entry->size != size) {
            LOG(Helper::LogLevel::LL_Warning, 
                "Updating cache entry %lld with different size: old=%zu, new=%zu\n", 
                listID, entry->size, size);
        }
        
        currentSize -= entry->size;
        
        if (entry->data) {
            free(entry->data);
        }
        
        entry->data = malloc(size);
        if (entry->data == nullptr) {
            LOG(Helper::LogLevel::LL_Error, "Failed to re-allocate memory for listID %lld\n", listID);
            // 从缓存中移除此条目，避免内存泄漏
            removeFromList(entry);
            cache.erase(it);
            delete entry;
            return;
        }
        
        memcpy(entry->data, data, size);
        entry->size = size;
        currentSize += size;
        
        entry->lastAccessTime = std::chrono::steady_clock::now().time_since_epoch().count();
        entry->accessCount++;
        
        moveToHead(entry);
        LOG(Helper::LogLevel::LL_Debug, "Updated posting list %lld in buffer pool, size: %zu\n", listID, size);
    } else {
        // 腾出空间
        while (currentSize + size > maxSize && !cache.empty()) {
            evict();
        }
        
        // 如果腾出空间后仍然无法容纳，则不缓存
        if (currentSize + size > maxSize) {
            LOG(Helper::LogLevel::LL_Warning, "Cannot add to buffer pool: entry too large (%zu bytes) after eviction\n", size);
            return;
        }
        
        // 创建新条目
        PostingListEntry* entry = new PostingListEntry();
        entry->listID = listID;

        entry->data = malloc(size);
        if (entry->data == nullptr) {
            LOG(Helper::LogLevel::LL_Error, "Failed to allocate memory for buffer pool entry\n");
            delete entry;
            return;
        }
        
        memcpy(entry->data, data, size);
        entry->size = size;
        entry->lastAccessTime = std::chrono::steady_clock::now().time_since_epoch().count();
        entry->accessCount = 1;
        
        cache[listID] = entry;
        currentSize += size;
        
        addToHead(entry);
        LOG(Helper::LogLevel::LL_Debug, "Added new posting list %lld to buffer pool, size: %zu\n", listID, size);
    }
}

void SPANNBufferPool::evict() {
    if (lruTail == nullptr) return;
    
    PostingListEntry* toEvict = lruTail;
    
    // 从LRU链表中移除
    removeFromList(toEvict);
    
    // 从哈希表中移除
    cache.erase(toEvict->listID);
    currentSize -= toEvict->size;
    
    // 释放内存
    if (toEvict->data) {
        free(toEvict->data);
    }
    delete toEvict;
}

void SPANNBufferPool::moveToHead(PostingListEntry* entry) {
    if (entry == lruHead) return;
    
    removeFromList(entry);
    addToHead(entry);
}

void SPANNBufferPool::removeFromList(PostingListEntry* entry) {
    if (entry->prev) {
        entry->prev->next = entry->next;
    } else {
        lruHead = entry->next;
    }
    
    if (entry->next) {
        entry->next->prev = entry->prev;
    } else {
        lruTail = entry->prev;
    }
    
    entry->prev = entry->next = nullptr;
}

void SPANNBufferPool::addToHead(PostingListEntry* entry) {
    entry->prev = nullptr;
    entry->next = lruHead;
    
    if (lruHead) {
        lruHead->prev = entry;
    }
    lruHead = entry;
    
    if (lruTail == nullptr) {
        lruTail = entry;
    }
}

void SPANNBufferPool::prefetch(const std::vector<int64_t>& listIDs) {
    if (listIDs.empty()) return;
    
    LOG(Helper::LogLevel::LL_Debug, "Prefetching %zu posting lists\n", listIDs.size());
    
    // 使用独占锁，避免锁升级问题
    std::unique_lock<std::shared_mutex> lock(mutex);
    
    size_t hitCount = 0;
    for (int64_t listID : listIDs) {
        auto it = cache.find(listID);
        if (it != cache.end()) {
            // 如果已经在缓存中，提升其优先级
            hitCount++;
            PostingListEntry* entry = it->second;
            entry->accessCount += 0.1; // 轻微提升访问计数
            moveToHead(entry);  // 移到LRU链表前端
        }
    }
    
    LOG(Helper::LogLevel::LL_Debug, "Prefetch found %zu/%zu lists already in cache\n", 
        hitCount, listIDs.size());
}

float SPANNBufferPool::getHitRatio() const {
    if (totalAccess == 0) return 0.0f;
    return static_cast<float>(cacheHits) / totalAccess;
}

void SPANNBufferPool::resetStats() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    totalAccess = 0;
    cacheHits = 0;
    LOG(Helper::LogLevel::LL_Info, "SPANN Buffer Pool statistics reset\n");
}

size_t SPANNBufferPool::getTotalAccess() const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return totalAccess;
}

size_t SPANNBufferPool::getHits() const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return cacheHits;
}

} // namespace SPANN
} // namespace SPTAG