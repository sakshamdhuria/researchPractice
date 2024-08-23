#include <iostream>
#include <vector>
#include <atomic>
#include <memory>
#include <algorithm>

template<typename T>
class LockFreeVector {
private:
    struct Block {
        T* data;
        size_t start;
        size_t end;
    };
    
    std::vector<Block> lookupVector; 
    std::atomic<size_t> totalSize;   
    size_t initialBlockSize;       
    
    Block allocateBlock(size_t size, size_t startIdx) {
        Block block;
        block.data = new T[size];
        block.start = startIdx;
        block.end = startIdx + size - 1;
        return block;
    }
    
public:
    LockFreeVector(size_t initialSize = 16) : totalSize(0), initialBlockSize(initialSize) {
        lookupVector.push_back(allocateBlock(initialSize, 0));
    }
    
    ~LockFreeVector() {
        for (auto& block : lookupVector) {
            delete[] block.data;
        }
    }
    
    void push_back(const T& value) {
        size_t idx = totalSize.fetch_add(1);
        
        //finding the index to insert to 
        auto it = std::lower_bound(lookupVector.begin(), lookupVector.end(), idx,
                                   [](const Block& block, size_t idx) {
                                       return block.end < idx;
                                   });
        
        if (it != lookupVector.end() && it->start <= idx && idx <= it->end) {
            it->data[idx - it->start] = value;
        } else {
            size_t newBlockSize = initialBlockSize * (1ULL << lookupVector.size());
            Block newBlock = allocateBlock(newBlockSize, it->end + 1);
            lookupVector.push_back(newBlock);
            newBlock.data[0] = value;
        }
    }
    
    T& operator[](size_t idx) {
        auto it = std::lower_bound(lookupVector.begin(), lookupVector.end(), idx,
                                   [](const Block& block, size_t idx) {
                                       return block.end < idx;
                                   });
        
        if (it != lookupVector.end() && it->start <= idx && idx <= it->end) {
            return it->data[idx - it->start];
        } else {
            throw std::out_of_range("Index out of range");
        }
    }
    
    size_t size() const {
        return totalSize.load();
    }
};
