//Unbounded Lock Free Deque
//Class name ULFD so when you initialize a ULFD it will be a struct that will create a lookup vector of vector that reserves 50 space, as well as initial size if 
that was passed in as a paremeter otherwise store 2 as default, so the lookup vector of vector in a default scenario will go sizes of 2 2 4 8 etc where the next 
size is the sum of all its previous this will help us not have to copy previous elements when resizing the dequeu and the look up vector can help us just define 
more space and when needed to access specific index we can do very simple binary search in the lookup vectors 50 reserved space like lets say if we are looking 
for index 11 we are trying to find the largest 2 to power less than and smallest two to power greater than and then we know which vector it corresponds to so we 
can then just go there and acess the element smartly there 

#include <iostream>
#include <vector>
#include <atomic>
#include <memory>
#include <algorithm>
#include <stdexcept>

template<typename T>
class ULFD {
private:
    std::vector<std::vector<T>> lookupVector; 
    std::atomic<size_t> frontIdx;             
    std::atomic<size_t> backIdx;             
    size_t initialSize;                      


    void grow() {
        size_t newSize = totalCapacity + (totalCapacity ? totalCapacity : initialSize);
        lookupVector.emplace_back(std::vector<T>(newSize));
        totalCapacity += newSize;
    }

    size_t findBlock(size_t idx) const {
        size_t sum = 0;
        for (size_t i = 0; i < lookupVector.size(); ++i) {
            sum += lookupVector[i].size();
            if (idx < sum) {
                return i;
            }
        }
        throw std::out_of_range("Index out of range");
    }

public:
    ULFD(size_t initialSize = 2) : frontIdx(0), backIdx(0), initialSize(initialSize), totalCapacity(0) {
        lookupVector.reserve(50);  // Reserve space for 50 blocks
        grow();  // Initialize with the first block
    }

    void push_back(const T& value) {
        size_t idx = backIdx.fetch_add(1);
        if (idx >= totalCapacity) {
            grow();
        }
        size_t blockIdx = findBlock(idx);
        lookupVector[blockIdx][idx - (totalCapacity - lookupVector[blockIdx].size())] = value;
    }

    void push_front(const T& value) {
        size_t idx = frontIdx.fetch_sub(1) - 1;
        if (idx >= totalCapacity) {
            grow();
        }
        size_t blockIdx = findBlock(idx);
        lookupVector[blockIdx][idx - (totalCapacity - lookupVector[blockIdx].size())] = value;
    }

    T& operator[](size_t idx) {
        size_t blockIdx = findBlock(idx);
        return lookupVector[blockIdx][idx - (totalCapacity - lookupVector[blockIdx].size())];
    }

    void pop_back() {
        if (backIdx > frontIdx) {
            --backIdx;
        } else {
            throw std::out_of_range("Deque is empty");
        }
    }

    void pop_front() {
        if (frontIdx < backIdx) {
            ++frontIdx;
        } else {
            throw std::out_of_range("Deque is empty");
        }
    }

    size_t size() const {
        return backIdx - frontIdx;
    }

    bool empty() const {
        return size() == 0;
    }
};
