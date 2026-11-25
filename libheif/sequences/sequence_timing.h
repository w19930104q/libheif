/*
 * HEIF codec.
 * Copyright (c) 2026 libheif contributors
 *
 * This file is part of libheif.
 *
 * libheif is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * libheif is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libheif.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LIBHEIF_SEQUENCE_TIMING_H
#define LIBHEIF_SEQUENCE_TIMING_H

#include "error.h"

#include <cstdint>
#include <vector>

// The single rounding rule for all sequence-timing conversions. The sequence
// timebase, the per-frame durations and the total duration are kept in one
// consistent time-unit table by always rounding to nearest (ties away from
// zero) and by deriving per-frame durations from the cumulative timeline
// instead of rounding each frame independently. This makes a timescale
// conversion reversible: converting forth and back returns the original
// durations whenever the rounded frame start times are distinct.

// Rescale a single duration from old_base to new_base, rounding to nearest.
// Fails on a zero timescale or when the result does not fit into uint64_t.
Result<uint64_t> rescale_duration_to_timescale(uint64_t duration, uint32_t old_base, uint32_t new_base);

// Rescale per-sample durations to a new timescale on the cumulative timeline
// ("rounding is handled by the total timebase"): each sample start time is
// rounded once from the cumulative media time and the returned durations are
// the differences of consecutive rounded start times. Hence
//
//   sum(result) == rescale_duration_to_timescale(sum(durations))
//
// and no per-frame rounding error accumulates (a frame never absorbs the
// rounding error of its predecessor, which is what made the second frame of a
// short sequence linger after a playback-rate switch).
//
// Fails - modifying nothing - when a sample would be assigned a zero duration
// or a duration above UINT32_MAX, or when the cumulative time overflows.
Result<std::vector<uint32_t>> rescale_sample_durations_to_timescale(const std::vector<uint32_t>& durations,
                                                                    uint32_t old_base, uint32_t new_base);

#endif //LIBHEIF_SEQUENCE_TIMING_H
