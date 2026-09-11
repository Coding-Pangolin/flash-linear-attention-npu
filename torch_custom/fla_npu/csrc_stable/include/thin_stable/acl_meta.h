// Shared descriptor/metadata layer for the Stable-ABI thin launchers.
//
// Everything here talks to CANN through the dlopen'd acl* symbols and to torch
// exclusively through the aoti_torch_* C shims.  No ATen/c10 headers, no
// pybind11, so a launcher built on top of it stays valid across torch versions
// (see docs/architecture/stable-abi-migration.md).
#pragma once

#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>

#include "thin_launcher/runtime.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fla_npu_thin {
namespace stable {

typedef struct aclTensor aclTensor;
typedef struct aclIntArray aclIntArray;
typedef struct aclOpExecutor aclOpExecutor;

constexpr int32_t kAclFormatNd = 2;

// torch ScalarType -> ACL data type (mirrors csrc_thin/src/tensor_desc.cpp).
inline int32_t acl_dtype(int32_t scalar_type) {
  switch (scalar_type) {
    case 6:
      return 0;  // kFloat  -> ACL_FLOAT
    case 5:
      return 1;  // kHalf   -> ACL_FLOAT16
    case 1:
      return 2;  // kChar   -> ACL_INT8
    case 3:
      return 3;  // kInt    -> ACL_INT32
    case 0:
      return 4;  // kByte   -> ACL_UINT8
    case 2:
      return 6;  // kShort  -> ACL_INT16
    case 4:
      return 9;  // kLong   -> ACL_INT64
    case 7:
      return 11;  // kDouble -> ACL_DOUBLE
    case 11:
      return 12;  // kBool   -> ACL_BOOL
    case 15:
      return 27;  // kBFloat16 -> ACL_BF16
    default:
      throw std::runtime_error(
          "fla_npu_thin(stable): unsupported tensor dtype id " +
          std::to_string(scalar_type));
  }
}

inline int64_t element_size(int32_t scalar_type) {
  switch (scalar_type) {
    case 0:
    case 1:
    case 11:
      return 1;
    case 2:
    case 5:
    case 15:
      return 2;
    case 3:
    case 6:
      return 4;
    case 4:
    case 7:
      return 8;
    default:
      throw std::runtime_error(
          "fla_npu_thin(stable): unsupported dtype id " +
          std::to_string(scalar_type));
  }
}

using AclCreateTensorFn = aclTensor* (*)(const int64_t*, uint64_t, int32_t,
                                         const int64_t*, int64_t, int32_t,
                                         const int64_t*, uint64_t, void*);
using AclDestroyTensorFn = int (*)(aclTensor*);
using LaunchFn = int (*)(void*, uint64_t, aclOpExecutor*, void*);
using AclCreateIntArrayFn = aclIntArray* (*)(const int64_t*, uint64_t);
using AclDestroyIntArrayFn = int (*)(aclIntArray*);

struct TensorMeta {
  bool defined = false;
  void* data = nullptr;
  int64_t ndim = 0;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t storage_offset = 0;
  int64_t storage_numel = 0;
  int32_t scalar_type = 0;
  int32_t device_type = 0;
  int32_t device_index = 0;
  bool contiguous = false;
};

// Fills `meta` from a raw handle.  Deliberately does not construct a
// torch::stable::Tensor: that constructor *steals* ownership, so a temporary
// would release the dispatcher's own tensor and the process crashes later.
// (That was the root cause of the first two handle-unboxing attempts.)
inline void fill_meta(AtenTensorHandle handle, TensorMeta* meta) {
  // A missing optional arrives as a null handle; short-circuiting keeps the
  // needed aoti_torch_* set inside what older libtorch builds export.
  if (handle == nullptr) {
    return;
  }
  meta->defined = true;
  void* tensor_data = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle, &tensor_data));
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(handle, &meta->ndim));
  int64_t* sizes = nullptr;
  int64_t* strides = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle, &sizes));
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle, &strides));
  meta->sizes.assign(sizes, sizes + meta->ndim);
  meta->strides.assign(strides, strides + meta->ndim);
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_storage_offset(handle, &meta->storage_offset));
  // Storage extent in elements, exactly like the ctypes path computes it from
  // `untyped_storage().nbytes() // element_size` (aoti_torch_get_storage_numel
  // is the *view* numel and is wrong for paged/offset states).
  int64_t storage_bytes = 0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(handle, &storage_bytes));
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(handle, &meta->scalar_type));
  const int64_t item_size = element_size(meta->scalar_type);
  meta->storage_numel = storage_bytes / item_size;
  // aclCreateTensor wants the *storage* base address and takes storage_offset
  // separately; the shim's data_ptr already includes the offset.
  meta->data = static_cast<uint8_t*>(tensor_data) -
               meta->storage_offset * item_size;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_type(handle, &meta->device_type));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(handle, &meta->device_index));
  // Computed locally rather than via aoti_torch_is_contiguous (2.9-only).
  int64_t expected = 1;
  meta->contiguous = true;
  for (int64_t dim = meta->ndim - 1; dim >= 0; --dim) {
    if (meta->sizes[dim] != 1 && meta->strides[dim] != expected) {
      meta->contiguous = false;
      break;
    }
    expected *= meta->sizes[dim];
  }
}

inline TensorMeta meta_of_handle(AtenTensorHandle handle) {
  TensorMeta meta;
  fill_meta(handle, &meta);
  return meta;
}

inline TensorMeta meta_of(const torch::stable::Tensor& tensor) {
  return meta_of_handle(tensor.get());
}

inline int64_t size_of(const TensorMeta& meta, int64_t dim) {
  if (dim < 0 || dim >= meta.ndim) {
    throw std::runtime_error(
        "fla_npu_thin(stable): size_of dim out of range");
  }
  return meta.sizes[static_cast<size_t>(dim)];
}

inline TensorMeta meta_optional_handle(std::optional<AtenTensorHandle> handle) {
  TensorMeta meta;
  if (handle.has_value()) {
    fill_meta(*handle, &meta);
  }
  return meta;
}

// RAII wrapper around aclCreateTensor / aclDestroyTensor.  Same semantics as the
// ctypes and pybind paths: contiguous tensors describe storage with the logical
// shape, non-contiguous ones fall back to a flat storage extent.
class AclTensorView {
 public:
  explicit AclTensorView(const TensorMeta& meta) {
    if (!meta.defined) {
      return;
    }
    const std::vector<int64_t> storage_dims =
        meta.contiguous ? meta.sizes
                        : std::vector<int64_t>{meta.storage_numel};
    auto create = reinterpret_cast<AclCreateTensorFn>(
        Runtime::instance().symbol("aclCreateTensor"));
    ptr_ = create(meta.sizes.data(), static_cast<uint64_t>(meta.ndim),
                  acl_dtype(meta.scalar_type), meta.strides.data(),
                  meta.storage_offset, kAclFormatNd, storage_dims.data(),
                  static_cast<uint64_t>(storage_dims.size()), meta.data);
    if (ptr_ == nullptr) {
      throw std::runtime_error(
          "fla_npu_thin(stable): aclCreateTensor returned nullptr");
    }
  }

  ~AclTensorView() {
    if (ptr_ != nullptr) {
      auto destroy = reinterpret_cast<AclDestroyTensorFn>(
          Runtime::instance().symbol("aclDestroyTensor"));
      destroy(ptr_);
    }
  }

  AclTensorView(const AclTensorView&) = delete;
  AclTensorView& operator=(const AclTensorView&) = delete;

  aclTensor* get() const { return ptr_; }

 private:
  aclTensor* ptr_ = nullptr;
};

// Allocate a contiguous tensor of `meta`'s shape/dtype/device through the C
// shim.  torch::stable::empty_like would be an aten::empty_like dispatcher round
// trip (and needs a Tensor, reviving the stealing-temporary trap).
inline torch::stable::Tensor allocate_like(const TensorMeta& meta) {
  std::vector<int64_t> strides(meta.sizes.size(), 1);
  for (size_t dim = meta.sizes.size(); dim-- > 1;) {
    strides[dim - 1] = strides[dim] * meta.sizes[dim];
  }
  AtenTensorHandle handle = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      static_cast<int64_t>(meta.sizes.size()), meta.sizes.data(),
      strides.data(), meta.scalar_type, meta.device_type, meta.device_index,
      &handle));
  return torch::stable::Tensor(handle);  // steals the new reference
}

// Allocate a contiguous tensor with explicit sizes/dtype on `device_source`'s
// device (used by the generated adapters, whose output shapes come from the
// spec's structured `output` description).
inline torch::stable::Tensor allocate_sizes(
    const std::vector<int64_t>& sizes, int32_t dtype,
    const TensorMeta& device_source) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (size_t dim = sizes.size(); dim-- > 1;) {
    strides[dim - 1] = strides[dim] * sizes[dim];
  }
  AtenTensorHandle handle = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      static_cast<int64_t>(sizes.size()), sizes.data(), strides.data(), dtype,
      device_source.device_type, device_source.device_index, &handle));
  return torch::stable::Tensor(handle);
}

// Allocate a 1-D byte buffer (workspace) on `meta`'s device.
inline torch::stable::Tensor allocate_bytes(int64_t bytes,
                                            const TensorMeta& meta) {
  const int64_t sizes[1] = {bytes};
  const int64_t strides[1] = {1};
  constexpr int32_t kTorchByte = 0;
  AtenTensorHandle handle = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      1, sizes, strides, kTorchByte, meta.device_type, meta.device_index,
      &handle));
  return torch::stable::Tensor(handle);
}

// ---------------------------------------------------------------------------
// int[] arguments.
//
// torch 2.9's stable value conversions have no list support at all (no
// aoti_torch_*list* shim, no ToImpl<std::vector<T>>), so an `int[]` schema
// argument cannot be read on this side of the ABI.  The launchers therefore
// take such arrays as a *host int64 tensor* instead (the Python wrapper builds
// one) and the values are copied out here.  That keeps every op expressible
// without a list-capable shim.
// ---------------------------------------------------------------------------
inline std::vector<int64_t> host_int_values(AtenTensorHandle handle) {
  std::vector<int64_t> values;
  if (handle == nullptr) {
    return values;
  }
  TensorMeta meta;
  fill_meta(handle, &meta);
  if (!meta.defined) {
    return values;
  }
  const int64_t count = meta.storage_numel > 0 ? meta.storage_numel : 0;
  if (meta.scalar_type == 4) {  // kLong
    const auto* data = static_cast<const int64_t*>(meta.data);
    values.assign(data, data + count);
  } else if (meta.scalar_type == 3) {  // kInt
    const auto* data = static_cast<const int32_t*>(meta.data);
    values.reserve(static_cast<size_t>(count));
    for (int64_t index = 0; index < count; ++index) {
      values.push_back(static_cast<int64_t>(data[index]));
    }
  } else {
    throw std::runtime_error(
        "fla_npu_thin(stable): int[] argument must be an int32/int64 tensor");
  }
  return values;
}

// RAII aclIntArray built from host values (a null/empty vector yields nullptr,
// matching the ctypes path where an absent option is passed as nullptr).
class AclIntArrayView {
 public:
  explicit AclIntArrayView(const std::vector<int64_t>& values)
      : owned_(values) {
    if (owned_.empty()) {
      return;
    }
    auto create = reinterpret_cast<AclCreateIntArrayFn>(
        Runtime::instance().symbol("aclCreateIntArray"));
    ptr_ = create(owned_.data(), static_cast<uint64_t>(owned_.size()));
    if (ptr_ == nullptr) {
      throw std::runtime_error(
          "fla_npu_thin(stable): aclCreateIntArray returned nullptr");
    }
  }

  ~AclIntArrayView() {
    if (ptr_ != nullptr) {
      auto destroy = reinterpret_cast<AclDestroyIntArrayFn>(
          Runtime::instance().symbol("aclDestroyIntArray"));
      destroy(ptr_);
    }
  }

  AclIntArrayView(const AclIntArrayView&) = delete;
  AclIntArrayView& operator=(const AclIntArrayView&) = delete;

  aclIntArray* get() const { return ptr_; }

 private:
  std::vector<int64_t> owned_;
  aclIntArray* ptr_ = nullptr;
};

}  // namespace stable
}  // namespace fla_npu_thin
