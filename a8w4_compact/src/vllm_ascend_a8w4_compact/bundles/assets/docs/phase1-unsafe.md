# Phase-1 post bulk copy is TP2-unsafe (kept only as history)

`patches/history/0001-a8w4-post-bulk-grouplist.patch` replaced the post's
`groupListGM.GetValue` walks with one `DataCopyPad` into a new 2 KB UB buffer.

On TP1 it was numerically correct but performance-neutral. On TP2 the extra
buffer overran the post's UB budget (`mmOutQueue` + `quantOutQueue` +
`quantScaleOutQueue` + `weightAuxiliaryMatrixInQueue` + `reduceWorkspace` are
sized to ~191 KB by tiling):

* stock and phase-1-only packages both boot on TP1;
* on TP2, both the phase-1-only and the phase-1+batch1 packages faulted in
  `GroupedMatmulSwigluQuantV2_fa3d..._2`, program id 115, with
  `fftsplus aivector error` and a non-zero `vec error info`, followed by an
  HCCL watchdog;
* the same full vendor package with the stock kernel booted normally.

The final patch does not add a UB buffer in the post. The post's O(E) walk is
replaced by the descriptor binary search and the O(1) next-active advance,
which use only scalar reads from the GM descriptor and add no UB pressure.

Do not enable phase-1 together with this patch.
