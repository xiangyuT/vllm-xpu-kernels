# Custom kernel additions on the decode development branch

`_C.rms_norm_gated_decode(out, input, gate, weight, epsilon)` writes gated
RMSNorm into `out`. Inputs are contiguous XPU FP16 tensors: input/gate/output
have shape `[rows,128]`, weight is `[128]`. The operation computes RMSNorm
before SiLU gating, with FP32 intermediate arithmetic and one final FP16 cast.
Inputs are unchanged; callers supply a separate output buffer. This entry
uses the existing `_C` SYCL build and adds no dependency.

The initial vLLM dispatch is limited to the validated GDN decode shape
`[24,128]` in inference/eager execution; other configurations retain fallback.
