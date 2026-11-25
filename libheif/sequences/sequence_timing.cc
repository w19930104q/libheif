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

#include "sequence_timing.h"

#include <limits>


Result<uint64_t> rescale_duration_to_timescale(uint64_t duration, uint32_t old_base, uint32_t new_base)
{
  if (old_base == 0 || new_base == 0) {
    return Error{
      heif_error_Invalid_input,
      heif_suberror_Unspecified,
      "Cannot rescale a duration with a zero timescale."
    };
  }

  // Split duration = q * old_base + r so that no intermediate exceeds uint64_t:
  // r < old_base <= 2^32-1 and new_base <= 2^32-1, hence
  // r * new_base + old_base / 2 < 2^64 always holds.
  uint64_t q = duration / old_base;
  uint64_t r = duration % old_base;

  if (q > std::numeric_limits<uint64_t>::max() / new_base) {
    return Error{
      heif_error_Invalid_input,
      heif_suberror_Unspecified,
      "Duration overflow while rescaling to the new timescale."
    };
  }

  uint64_t scaled = q * new_base;
  uint64_t remainder = (r * new_base + old_base / 2) / old_base;

  if (scaled > std::numeric_limits<uint64_t>::max() - remainder) {
    return Error{
      heif_error_Invalid_input,
      heif_suberror_Unspecified,
      "Duration overflow while rescaling to the new timescale."
    };
  }

  return scaled + remainder;
}


Result<std::vector<uint32_t>> rescale_sample_durations_to_timescale(const std::vector<uint32_t>& durations,
                                                                    uint32_t old_base, uint32_t new_base)
{
  if (old_base == 0 || new_base == 0) {
    return Error{
      heif_error_Invalid_input,
      heif_suberror_Unspecified,
      "Cannot rescale sample durations with a zero timescale."
    };
  }

  std::vector<uint32_t> rescaled;
  rescaled.reserve(durations.size());

  uint64_t cumulative_media_time = 0;  // in old_base units
  uint64_t previous_start_time = 0;    // rounded, in new_base units

  for (uint32_t duration : durations) {
    cumulative_media_time += duration;

    auto endResult = rescale_duration_to_timescale(cumulative_media_time, old_base, new_base);
    if (!endResult) {
      return endResult.error();
    }

    uint64_t end_time = *endResult;

    // rescale_duration_to_timescale() is non-decreasing in its input, so the
    // rounded end times are non-decreasing as well.
    uint64_t sample_duration = end_time - previous_start_time;

    if (sample_duration == 0) {
      return Error{
        heif_error_Invalid_input,
        heif_suberror_Unspecified,
        "Rescaling to the new timescale would collapse a sample to zero duration."
      };
    }

    if (sample_duration > std::numeric_limits<uint32_t>::max()) {
      return Error{
        heif_error_Invalid_input,
        heif_suberror_Unspecified,
        "Rescaling to the new timescale would overflow a sample duration."
      };
    }

    rescaled.push_back(static_cast<uint32_t>(sample_duration));
    previous_start_time = end_time;
  }

  return rescaled;
}
