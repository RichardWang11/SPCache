#ifndef _SPTAG_SPANN_BUFFERPOOL_H_
#define _SPTAG_SPANN_BUFFERPOOL_H_

#include "inc/Core/Common.h"
#include <unordered_map>
#include <shared_mutex>
#include <chrono>
#include <vector>

namespace SPTAG {
namespace SPANN {

struct PostingListEntry {
    int64_t listID;
    void* data;
    size_t size;
    uint64_t lastAccessTime;
    uint64_t accessCount;
    PostingListEntry* prev;
    PostingListEntry* next;
    
    PostingListEntry() : listID(-1), data(nullptr), size(0), 
                        lastAccessTime(0), accessCount(0), 
                        prev(nullptr), next(nullptr) {}
};

class SPANNBufferPool {
private:
    std::unordered_map<int64_t, PostingListEntry*> cache;
    PostingListEntry* lruHead;
    PostingListEntry* lruTail;
    size_t currentSize;
    size_t maxSize;
    uint64_t totalAccess;
    uint64_t cacheHits;
    mutable std::shared_mutex mutex;
    
    void evict();
    void moveToHead(PostingListEntry* entry);
    void removeFromList(PostingListEntry* entry);
    void addToHead(PostingListEntry* entry);

public:
    SPANNBufferPool(size_t maxSizeBytes);
    ~SPANNBufferPool();
    
    void* get(int64_t listID, size_t& size);
    void put(int64_t listID, void* data, size_t size);
    void prefetch(const std::vector<int64_t>& listIDs);
    float getHitRatio() const;
    void resetStats();
    size_t getTotalAccess() const;
    size_t getHits() const; 
    size_t getCurrentSize() const { return currentSize; }
    size_t getMaxSize() const { return maxSize; }
};

} // namespace SPANN
} // namespace SPTAG

#endif // _SPTAG_SPANN_BUFFERPOOL_H_