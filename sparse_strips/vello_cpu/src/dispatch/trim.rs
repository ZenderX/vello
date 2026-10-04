// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shrinking the dispatchers' reusable buffers back to recent use between frames.

use crate::coarse::CommandBucketer;
use crate::record::RecordedFill;
use alloc::vec::Vec;
use vello_common::record::CommandRecorder;
use vello_common::strip_generator::StripStorage;

/// Fractional bits of [`UsageMark`]'s fixed-point mark, so that small marks decay smoothly too.
const MARK_FRACTION_BITS: u32 = 8;

/// Each frame a [`UsageMark`] loses 1/2^`MARK_DECAY_SHIFT` of itself, which halves it in
/// about 2839 frames (47 s at 60 Hz).
///
/// A faster decay shrinks and regrows buffers so often under unpaced rendering that the
/// reallocations fragment the process heap.
const MARK_DECAY_SHIFT: u32 = 12;

/// Capacity, in bytes, that [`trim_to_mark`] always leaves a buffer, so small buffers are not
/// churned.
const TRIM_FLOOR_BYTES: usize = 4096;

/// A decaying high-water mark of how much of a reusable buffer recent frames used.
///
/// The dispatchers' buffers live across frames and only ever grow. Without a trim their
/// capacity ratchets up to the heaviest frame ever rendered.
#[derive(Debug, Default)]
pub(super) struct UsageMark {
    frame_peak: usize,
    scaled_mark: u64,
}

impl UsageMark {
    pub(super) fn record(&mut self, used: usize) {
        self.frame_peak = self.frame_peak.max(used);
    }

    /// Fold the frame's peak into the mark, decay it, and return it.
    pub(super) fn end_frame(&mut self) -> usize {
        let peak = u64::try_from(self.frame_peak)
            .unwrap_or(u64::MAX)
            .saturating_mul(1 << MARK_FRACTION_BITS);
        let decayed = self.scaled_mark - (self.scaled_mark >> MARK_DECAY_SHIFT);
        self.scaled_mark = decayed.max(peak);
        self.frame_peak = 0;
        usize::try_from(self.scaled_mark.div_ceil(1 << MARK_FRACTION_BITS)).unwrap_or(usize::MAX)
    }
}

/// Release `buf`'s capacity once it exceeds twice `mark`, down to the power of two at or
/// above the mark.
///
/// Power-of-two targets match the capacities `Vec` growth produces, so the blocks a trim
/// frees and a later regrowth allocates stay in a few size classes the heap can reuse.
/// Only call this between frames, on the thread that owns `buf`.
pub(super) fn trim_to_mark<T>(buf: &mut Vec<T>, mark: usize) {
    let mark = mark.max(TRIM_FLOOR_BYTES / size_of::<T>().max(1));
    if buf.capacity() > mark.saturating_mul(2) {
        buf.shrink_to(mark.checked_next_power_of_two().unwrap_or(mark));
    }
}

/// Fold `buf`'s current length into `mark`, then trim `buf` to the decayed mark.
fn trim_to_use<T>(mark: &mut UsageMark, buf: &mut Vec<T>) {
    mark.record(buf.len());
    trim_to_mark(buf, mark.end_frame());
}

/// Recent use of the buffers that hold one frame's recorded scene.
///
/// These follow the scene, so they too would otherwise keep the heaviest frame's capacity
/// for good.
#[derive(Debug, Default)]
pub(super) struct SceneMarks {
    nodes: UsageMark,
    draws: UsageMark,
    layers: UsageMark,
    strips: UsageMark,
    alphas: UsageMark,
    fill_attrs: UsageMark,
    /// The most render commands any one strip row held.
    row_cmds: UsageMark,
    /// The most depth commands any one strip row held.
    row_depth_cmds: UsageMark,
}

impl SceneMarks {
    /// Trim the scene buffers, while they still hold the frame just rendered, to recent use.
    pub(super) fn end_frame(
        &mut self,
        recorder: &mut CommandRecorder<RecordedFill>,
        strip_storage: &mut StripStorage,
        bucketer: &mut CommandBucketer,
    ) {
        trim_to_use(&mut self.nodes, &mut recorder.nodes);
        trim_to_use(&mut self.draws, &mut recorder.draws);
        trim_to_use(&mut self.layers, &mut recorder.layers);
        trim_to_use(&mut self.strips, &mut strip_storage.strips);
        trim_to_use(&mut self.alphas, &mut strip_storage.alphas);
        trim_to_use(&mut self.fill_attrs, &mut bucketer.paint_fill_attrs);
        let rows = bucketer.rows_mut();
        for row in rows.iter() {
            self.row_cmds.record(row.render_cmds.len());
            self.row_depth_cmds.record(row.depth_cmds.len());
        }
        let (cmds_mark, depth_mark) = (self.row_cmds.end_frame(), self.row_depth_cmds.end_frame());
        for row in rows {
            trim_to_mark(&mut row.render_cmds, cmds_mark);
            trim_to_mark(&mut row.depth_cmds, depth_mark);
        }
    }
}
