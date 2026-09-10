// Stable-ABI thin launcher spike: npu_recurrent_gated_delta_rule.
//
// This translation unit deliberately includes ONLY torch/csrc/stable/* (plus the
// torch-free dlopen helper shared with csrc_thin).  Nothing from ATen/c10 or
// pybind11 may appear here: the compile-once/run-on-many guarantee comes from
// touching nothing but the aoti_torch_* C shims.
#include <torch/csrc/stable/library.h>
#ifndef FLA_STABLE_NO_DEBUG_PROBE
#include <torch/csrc/stable/accelerator.h>
#endif
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>

#include "thin_launcher/runtime.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using torch::stable::Tensor;

// ---------------------------------------------------------------------------
// CANN / aclnn ABI (resolved at run time through the shared Runtime helper).
// ---------------------------------------------------------------------------
typedef struct aclTensor aclTensor;
typedef struct aclIntArray aclIntArray;
typedef struct aclOpExecutor aclOpExecutor;

constexpr int32_t kAclFloat = 0;
constexpr int32_t kAclFloat16 = 1;
constexpr int32_t kAclInt8 = 2;
constexpr int32_t kAclInt32 = 3;
constexpr int32_t kAclUint8 = 4;
constexpr int32_t kAclInt16 = 6;
constexpr int32_t kAclInt64 = 9;
constexpr int32_t kAclDouble = 11;
constexpr int32_t kAclBool = 12;
constexpr int32_t kAclBf16 = 27;
constexpr int32_t kAclFormatNd = 2;

// torch ScalarType -> ACL data type (mirrors csrc_thin/src/tensor_desc.cpp).
int32_t acl_dtype(int32_t scalar_type) {
  switch (scalar_type) {
    case 6:  // kFloat
      return kAclFloat;
    case 5:  // kHalf
      return kAclFloat16;
    case 1:  // kChar
      return kAclInt8;
    case 3:  // kInt
      return kAclInt32;
    case 0:  // kByte
      return kAclUint8;
    case 2:  // kShort
      return kAclInt16;
    case 4:  // kLong
      return kAclInt64;
    case 7:  // kDouble
      return kAclDouble;
    case 11:  // kBool
      return kAclBool;
    case 15:  // kBFloat16
      return kAclBf16;
    default:
      throw std::runtime_error(
          "fla_npu_thin(stable): unsupported tensor dtype id " +
          std::to_string(scalar_type));
  }
}

// Element size in bytes for the storage-extent computation below.
int64_t element_size(int32_t scalar_type) {
  switch (scalar_type) {
    case 0:   // kByte
    case 1:   // kChar
    case 11:  // kBool
      return 1;
    case 2:  // kShort
    case 5:  // kHalf
    case 15:  // kBFloat16
      return 2;
    case 3:   // kInt
    case 6:   // kFloat
      return 4;
    case 4:   // kLong
    case 7:   // kDouble
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

// Metadata pulled out of a stable tensor handle via the C shims.
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

TensorMeta meta_of(const torch::stable::Tensor& tensor) {
  TensorMeta meta;
  // A missing optional arrives as a null handle.  Short-circuiting here keeps
  // this launcher inside the aoti_torch_* set that older libtorch builds export
  // (aoti_torch_is_defined / aoti_torch_is_contiguous are 2.9-only).
  const AtenTensorHandle handle = tensor.get();
  if (handle == nullptr) {
    return meta;
  }
  meta.defined = true;
  void* tensor_data = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle, &tensor_data));
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(handle, &meta.ndim));
  int64_t* sizes = nullptr;
  int64_t* strides = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle, &sizes));
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle, &strides));
  meta.sizes.assign(sizes, sizes + meta.ndim);
  meta.strides.assign(strides, strides + meta.ndim);
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_storage_offset(handle, &meta.storage_offset));
  // Storage extent in elements, exactly like the ctypes path computes it from
  // `untyped_storage().nbytes() // element_size`.  (aoti_torch_get_storage_numel
  // is the *view* numel, which is wrong for paged/offset states.)
  int64_t storage_bytes = 0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(handle, &storage_bytes));
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(handle, &meta.scalar_type));
  const int64_t item_size = element_size(meta.scalar_type);
  meta.storage_numel = storage_bytes / item_size;
  // aclCreateTensor wants the *storage* base address and takes storage_offset
  // separately; the shim's data_ptr already includes the offset, so subtract it
  // back out or the offset would be applied twice (see the same note in
  // csrc_thin/src/tensor_desc.cpp).
  meta.data = static_cast<uint8_t*>(tensor_data) -
              meta.storage_offset * item_size;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_type(handle, &meta.device_type));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(handle, &meta.device_index));
  // Computed locally rather than via aoti_torch_is_contiguous.
  int64_t expected = 1;
  meta.contiguous = true;
  for (int64_t dim = meta.ndim - 1; dim >= 0; --dim) {
    if (meta.sizes[dim] != 1 && meta.strides[dim] != expected) {
      meta.contiguous = false;
      break;
    }
    expected *= meta.sizes[dim];
  }
  return meta;
}

// Default-constructed torch::stable::Tensor holds an *uninitialised* handle, so
// optionals must be tested with has_value() instead of value_or(Tensor()).
TensorMeta meta_optional(const std::optional<torch::stable::Tensor>& tensor) {
  if (!tensor.has_value()) {
    return TensorMeta{};
  }
  return meta_of(*tensor);
}

// RAII wrapper around aclCreateTensor / aclDestroyTensor.  Mirrors the ctypes
// and pybind paths: contiguous tensors describe storage with the logical shape,
// non-contiguous ones fall back to a flat storage extent.
class AclTensorView {
 public:
  explicit AclTensorView(const TensorMeta& meta) {
    if (!meta.defined) {
      return;
    }
    storage_dims_ = meta.contiguous ? meta.sizes
                                    : std::vector<int64_t>{meta.storage_numel};
    auto create = reinterpret_cast<AclCreateTensorFn>(
        fla_npu_thin::Runtime::instance().symbol("aclCreateTensor"));
    ptr_ = create(meta.sizes.data(), static_cast<uint64_t>(meta.ndim),
                  acl_dtype(meta.scalar_type), meta.strides.data(),
                  meta.storage_offset, kAclFormatNd, storage_dims_.data(),
                  static_cast<uint64_t>(storage_dims_.size()), meta.data);
    if (ptr_ == nullptr) {
      throw std::runtime_error(
          "fla_npu_thin(stable): aclCreateTensor returned nullptr");
    }
  }

  ~AclTensorView() {
    if (ptr_ != nullptr) {
      auto destroy = reinterpret_cast<AclDestroyTensorFn>(
          fla_npu_thin::Runtime::instance().symbol("aclDestroyTensor"));
      destroy(ptr_);
    }
  }

  AclTensorView(const AclTensorView&) = delete;
  AclTensorView& operator=(const AclTensorView&) = delete;

  aclTensor* get() const { return ptr_; }

 private:
  aclTensor* ptr_ = nullptr;
  std::vector<int64_t> storage_dims_;
};

using GetWorkspaceFn = int (*)(const aclTensor*, const aclTensor*,
                               const aclTensor*, const aclTensor*, aclTensor*,
                               const aclTensor*, const aclTensor*,
                               const aclTensor*, const aclTensor*,
                               const aclTensor*, float, aclTensor*, uint64_t*,
                               aclOpExecutor**);

// ---------------------------------------------------------------------------
// Op implementation.
// ---------------------------------------------------------------------------
Tensor run_recurrent_gated_delta_rule(const Tensor& query, const Tensor& key,
                                      const Tensor& value, const Tensor& state,
                                      const Tensor& beta,
                                      const Tensor& actual_seq_lengths,
                                      const Tensor& ssm_state_indices,
                                      const std::optional<Tensor>&
                                          num_accepted_tokens,
                                      const std::optional<Tensor>& g,
                                      const std::optional<Tensor>& gk,
                                      double scale, int64_t stream) {
  auto& rt = fla_npu_thin::Runtime::instance();
  auto get_ws = reinterpret_cast<GetWorkspaceFn>(
      rt.symbol("aclnnRecurrentGatedDeltaRuleGetWorkspaceSize"));
  auto launch =
      reinterpret_cast<LaunchFn>(rt.symbol("aclnnRecurrentGatedDeltaRule"));

  // Output allocation: same shape/dtype/device as `value` (identical to the
  // ctypes/pybind paths, which allocate `_shape(value)` with value's options).
  Tensor out = torch::stable::empty_like(value);
  const TensorMeta out_meta = meta_of(out);

  AclTensorView v_query(meta_of(query));
  AclTensorView v_key(meta_of(key));
  AclTensorView v_value(meta_of(value));
  AclTensorView v_beta(meta_of(beta));
  AclTensorView v_state(meta_of(state));
  AclTensorView v_seq(meta_of(actual_seq_lengths));
  AclTensorView v_idx(meta_of(ssm_state_indices));
  AclTensorView v_g(meta_optional(g));
  AclTensorView v_gk(meta_optional(gk));
  AclTensorView v_accepted(meta_optional(num_accepted_tokens));
  AclTensorView v_out(meta_of(out));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  const int get_ret = get_ws(
      v_query.get(), v_key.get(), v_value.get(), v_beta.get(), v_state.get(),
      v_seq.get(), v_idx.get(), v_g.get(), v_gk.get(), v_accepted.get(),
      static_cast<float>(scale), v_out.get(), &workspace_size, &executor);
  if (get_ret != 0) {
    throw std::runtime_error(
        "fla_npu_thin(stable): aclnnRecurrentGatedDeltaRuleGetWorkspaceSize "
        "failed: " +
        std::to_string(get_ret));
  }

  torch::stable::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    // Device scratch, allocated through the stable shim exactly like the
    // ctypes/pybind paths allocate it with torch.empty(..., uint8, device).
    int64_t ws_sizes[1] = {static_cast<int64_t>(workspace_size)};
    int64_t ws_strides[1] = {1};
    constexpr int32_t kTorchByte = 0;
    AtenTensorHandle ws_handle = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
        1, ws_sizes, ws_strides, kTorchByte, out_meta.device_type,
        out_meta.device_index, &ws_handle));
    workspace = torch::stable::Tensor(ws_handle);  // steals ownership
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_data_ptr(workspace.get(), &workspace_ptr));
  }
  const int launch_ret = launch(workspace_ptr, workspace_size, executor,
                                reinterpret_cast<void*>(stream));
  if (launch_ret != 0) {
    throw std::runtime_error(
        "fla_npu_thin(stable): aclnnRecurrentGatedDeltaRule failed: " +
        std::to_string(launch_ret));
  }
  return out;
}

// Boxed entry point: unbox StableIValues, run, pack the single output.
void boxed_recurrent_gated_delta_rule(StableIValue* stack,
                                      uint64_t num_inputs,
                                      uint64_t num_outputs) {
  (void)num_inputs;
  (void)num_outputs;
  const Tensor query = to<Tensor>(stack[0]);
  const Tensor key = to<Tensor>(stack[1]);
  const Tensor value = to<Tensor>(stack[2]);
  const Tensor state = to<Tensor>(stack[3]);
  const Tensor beta = to<Tensor>(stack[4]);
  const Tensor actual_seq_lengths = to<Tensor>(stack[5]);
  const Tensor ssm_state_indices = to<Tensor>(stack[6]);
  const auto num_accepted_tokens = to<std::optional<Tensor>>(stack[7]);
  const auto g = to<std::optional<Tensor>>(stack[8]);
  const auto gk = to<std::optional<Tensor>>(stack[9]);
  const double scale = to<double>(stack[10]);
  const int64_t stream = to<int64_t>(stack[11]);
  Tensor out = run_recurrent_gated_delta_rule(
      query, key, value, state, beta, actual_seq_lengths, ssm_state_indices,
      num_accepted_tokens, g, gk, scale, stream);
  stack[0] = from(out);
}

// Reports the current stream for `device_index` in two forms:
//   [0] aoti_torch_stream_id(aoti_torch_get_current_stream(...))
//   [1] stable::accelerator::getCurrentStream(...).id()
// Python compares these against torch_npu's raw accessor; both returned 0 on
// torch_npu 2.9.0.post2, i.e. the stable stream API does not map to the NPU
// stream yet.
#ifndef FLA_STABLE_NO_DEBUG_PROBE
void boxed_stream_probe(StableIValue* stack, uint64_t num_inputs,
                        uint64_t num_outputs) {
  (void)num_inputs;
  (void)num_outputs;
  const int64_t device_index = to<int64_t>(stack[0]);
  int64_t shim_id = -1;
  StreamHandle handle = nullptr;
  if (aoti_torch_get_current_stream(static_cast<int32_t>(device_index),
                                    &handle) == 0 &&
      handle != nullptr) {
    if (aoti_torch_stream_id(handle, &shim_id) != 0) {
      shim_id = -2;
    }
    aoti_torch_delete_stream(handle);
  }
  int64_t stream_id = 0;
  try {
    auto stream = torch::stable::accelerator::getCurrentStream(
        static_cast<int32_t>(device_index));
    stream_id = static_cast<int64_t>(stream.id());
  } catch (...) {
    stream_id = -1;
  }
  stack[0] = from(shim_id);
  stack[1] = from(stream_id);
}
#endif  // FLA_STABLE_NO_DEBUG_PROBE

}  // namespace

STABLE_TORCH_LIBRARY(fla_npu_thin, m) {
  m.def(
      "npu_recurrent_gated_delta_rule(Tensor query, Tensor key, Tensor value, "
      "Tensor(a!) state, Tensor beta, Tensor actual_seq_lengths, "
      "Tensor ssm_state_indices, Tensor? num_accepted_tokens, Tensor? g, "
      "Tensor? gk, float scale, int stream) -> Tensor");
  // Debug helper for the stream question: report the current stream two ways so
  // Python can compare them against torch_npu's raw accessor.
#ifndef FLA_STABLE_NO_DEBUG_PROBE
  m.def("_stream_probe(int device_index) -> (int, int)");
#endif
}

STABLE_TORCH_LIBRARY_IMPL(fla_npu_thin, CompositeExplicitAutograd, m) {
  m.impl("npu_recurrent_gated_delta_rule",
         &boxed_recurrent_gated_delta_rule);
#ifndef FLA_STABLE_NO_DEBUG_PROBE
  m.impl("_stream_probe", &boxed_stream_probe);
#endif
}
