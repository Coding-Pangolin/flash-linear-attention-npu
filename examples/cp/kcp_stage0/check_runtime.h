#pragma once

#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Check(aclError error, const char* what)
{
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + std::to_string(error));
    }
}

struct Device {
    aclrtStream stream = nullptr;
    Device()
    {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
    ~Device()
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }
};

template <class T>
class Buffer {
public:
    explicit Buffer(size_t count) : bytes_(count * sizeof(T))
    {
        Check(aclrtMalloc(&ptr_, bytes_, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { aclrtFree(ptr_); }
    uint8_t* Ptr() const { return static_cast<uint8_t*>(ptr_); }
    void Write(const std::vector<T>& data)
    {
        Check(aclrtMemcpy(ptr_, bytes_, data.data(), bytes_, ACL_MEMCPY_HOST_TO_DEVICE), "copy to NPU");
    }
    std::vector<T> Read() const
    {
        std::vector<T> data(bytes_ / sizeof(T));
        Check(aclrtMemcpy(data.data(), bytes_, ptr_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST), "copy to host");
        return data;
    }
private:
    size_t bytes_;
    void* ptr_ = nullptr;
};

uint16_t ToBf16(float x)
{
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

float FromBf16(uint16_t x)
{
    uint32_t bits = static_cast<uint32_t>(x) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float Verify(float actual, double expected, double absTolerance, double relTolerance, const char* name)
{
    const double error = std::abs(actual - expected);
    if (!std::isfinite(error) || error > absTolerance + relTolerance * std::abs(expected)) {
        throw std::runtime_error(std::string(name) + " mismatch: actual " + std::to_string(actual)
                                 + ", expected " + std::to_string(expected));
    }
    return static_cast<float>(error);
}
} // namespace
