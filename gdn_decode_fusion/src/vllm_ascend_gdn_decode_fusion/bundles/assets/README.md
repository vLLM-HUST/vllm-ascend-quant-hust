# Bundled assets

* `patches/gdn-decode-fusion.patch` - complete diff against the vllm-ascend
  tree for the GDN decode-path fusion (three files).
* `validation/check_bit_exact.py` - standalone NPU check that compares the
  fused split+norm kernel with `split` + `torch.cat` + `l2norm_fwd`, and the
  stride-aware gating kernel with a PyTorch reference.
* `method.md` - how the fusion works, the fallbacks and the safety argument.

Apply the patch with `gdn-decode-fusion install`; it uses `git apply` (or
`patch`) against the checkout, keeps a backup and aborts without changes when
the context does not match.
