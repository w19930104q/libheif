/*
  libheif regression test for sequence timescale rewriting.

  MIT License

  Copyright (c) 2026 libheif contributors

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

// A camera burst mixes frame intervals of 33.333 ms and 50 ms in one sequence.
// When such a sequence is rewritten to a different timescale (e.g. on a
// playback-rate switch), rounding each frame duration independently lets the
// rounding error accumulate unevenly across frames - the second frame of a
// short sequence lingered 1/30 s longer than the first. The rewrite must
// instead round on the cumulative timeline ("the total timebase"), so that
// the sequence timebase, the per-frame durations and the total duration stay
// in one reversible time-unit table.
//
// Field sample HF4: three frames with 29.97 fps / 50 ms / 29.97 fps intervals
// (durations 1001, 1500, 1001 at a timescale of 30000). Written with the
// original timebase and rewritten with the doubled timebase, both writings
// must give the per-frame start times 0, 2002, 5002 in units of 1/60000 s and
// the same cumulative duration (7004 / 60000 s).

#include "catch_amalgamated.hpp"
#include "libheif/heif.h"
#include "libheif/heif_sequences.h"
#include "sequence_file_builder.h"

#include <cstdint>
#include <vector>

namespace {

heif_error mem_writer(heif_context*, const void* data, size_t size, void* userdata)
{
  auto* out = static_cast<std::vector<uint8_t>*>(userdata);
  const auto* p = static_cast<const uint8_t*>(data);
  out->insert(out->end(), p, p + size);
  return heif_error{heif_error_Ok, heif_suberror_Unspecified, nullptr};
}

// Build the HF4 sequence as a file: three samples with the durations
// {1001, 1500, 1001} (29.97 fps, 50 ms, 29.97 fps) at a timescale of 30000.
std::vector<uint8_t> build_hf4_file()
{
  seqfile::SequenceFileParams p;
  p.timescale = 30000;
  p.mvhd_duration = 3502;
  p.mdhd_duration = 3502;
  p.with_editlist = false;
  p.stts = {{1, 1001}, {1, 1500}, {1, 1001}};
  p.num_samples = 3;
  return seqfile::build_sequence_file(p);
}

heif_context* read_context(const std::vector<uint8_t>& file)
{
  heif_context* ctx = heif_context_alloc();
  REQUIRE(ctx != nullptr);
  heif_error err = heif_context_read_from_memory(ctx, file.data(), file.size(), nullptr);
  INFO("read_from_memory: " << (err.message ? err.message : ""));
  REQUIRE(err.code == heif_error_Ok);
  REQUIRE(heif_context_has_sequence(ctx) == 1);
  return ctx;
}

// Read the durations of all remaining raw samples of the track.
std::vector<uint32_t> read_sample_durations(heif_track* track)
{
  std::vector<uint32_t> durations;
  for (;;) {
    heif_raw_sequence_sample* sample = nullptr;
    heif_error err = heif_track_get_next_raw_sequence_sample(track, &sample);
    if (err.code == heif_error_End_of_sequence) {
      break;
    }
    INFO("get_next_raw_sequence_sample: " << (err.message ? err.message : ""));
    REQUIRE(err.code == heif_error_Ok);
    REQUIRE(sample != nullptr);
    durations.push_back(heif_raw_sequence_sample_get_duration(sample));
    heif_raw_sequence_sample_release(sample);
  }
  return durations;
}

// The per-frame start times, normalized to units of 1/60000 s.
std::vector<uint64_t> frame_starts_in_60000(const std::vector<uint32_t>& durations, uint32_t timescale)
{
  std::vector<uint64_t> starts;
  uint64_t t = 0;
  for (uint32_t d : durations) {
    starts.push_back(t * 60000 / timescale);
    t += d;
  }
  return starts;
}

} // namespace


TEST_CASE("HF4: rewriting a sequence to the doubled timescale keeps frame starts and total duration")
{
  std::vector<uint8_t> file = build_hf4_file();

  // Original timebase: durations 1001, 1500, 1001 -> starts 0, 2002, 5002 in 1/60000 s.
  std::vector<uint32_t> expected_original = {1001, 1500, 1001};
  {
    heif_context* ctx = read_context(file);
    heif_track* track = heif_context_get_track(ctx, 0);
    REQUIRE(track != nullptr);
    REQUIRE(heif_track_get_timescale(track) == 30000);
    CHECK(read_sample_durations(track) == expected_original);
    heif_track_release(track);
    heif_context_free(ctx);
  }

  // Rewrite with the doubled timebase. (Fresh context: the raw-sample
  // iteration state is per track and cannot be rewound.)
  heif_context* ctx = read_context(file);
  heif_track* track = heif_context_get_track(ctx, 0);
  REQUIRE(track != nullptr);

  heif_error err = heif_context_rewrite_sequence_timescale(ctx, 60000);
  INFO("rewrite_sequence_timescale: " << (err.message ? err.message : ""));
  REQUIRE(err.code == heif_error_Ok);

  CHECK(heif_context_get_sequence_timescale(ctx) == 60000);
  CHECK(heif_track_get_timescale(track) == 60000);

  std::vector<uint32_t> expected_doubled = {2002, 3000, 2002};
  CHECK(read_sample_durations(track) == expected_doubled);

  // The anchor of the field result: both writings give the per-frame start
  // times 0, 2002, 5002 in units of 1/60000 s and the same cumulative duration.
  std::vector<uint64_t> expected_starts = {0, 2002, 5002};
  CHECK(frame_starts_in_60000(expected_original, 30000) == expected_starts);
  CHECK(frame_starts_in_60000(expected_doubled, 60000) == expected_starts);

  // Total duration: 3502 at 30000 and 7004 at 60000 are the same time span.
  CHECK(heif_context_get_sequence_duration(ctx) == 7004);

  heif_track_release(track);
  heif_context_free(ctx);
}


TEST_CASE("sequence timescale rewrite is reversible")
{
  std::vector<uint8_t> file = build_hf4_file();

  heif_context* ctx = read_context(file);
  heif_track* track = heif_context_get_track(ctx, 0);
  REQUIRE(track != nullptr);

  REQUIRE(heif_track_rewrite_timescale(track, 60000).code == heif_error_Ok);
  REQUIRE(heif_track_rewrite_timescale(track, 30000).code == heif_error_Ok);

  CHECK(heif_track_get_timescale(track) == 30000);
  std::vector<uint32_t> expected = {1001, 1500, 1001};
  CHECK(read_sample_durations(track) == expected);

  heif_track_release(track);
  heif_context_free(ctx);
}


TEST_CASE("frame duration rounding is anchored at the total timebase")
{
  // Three frames of 1 tick at timescale 3, rewritten to timescale 10.
  // Rounding each duration independently would give {3, 3, 3} (total 9), which
  // no longer matches the rescaled total duration of 10. Rounding on the
  // cumulative timeline gives {3, 4, 3}: the sum equals the rescaled total.
  seqfile::SequenceFileParams p;
  p.timescale = 3;
  p.mvhd_duration = 3;
  p.mdhd_duration = 3;
  p.with_editlist = false;
  p.stts = {{1, 1}, {1, 1}, {1, 1}};
  p.num_samples = 3;
  std::vector<uint8_t> file = seqfile::build_sequence_file(p);

  heif_context* ctx = read_context(file);
  heif_track* track = heif_context_get_track(ctx, 0);
  REQUIRE(track != nullptr);

  REQUIRE(heif_track_rewrite_timescale(track, 10).code == heif_error_Ok);

  std::vector<uint32_t> expected = {3, 4, 3};
  CHECK(read_sample_durations(track) == expected);

  // The rescaled total duration equals the sum of the rescaled frame durations.
  CHECK(heif_context_get_sequence_duration(ctx) == 3); // mvhd untouched by the track-level rewrite
  REQUIRE(heif_context_rewrite_sequence_timescale(ctx, 10).code == heif_error_Ok);
  CHECK(heif_context_get_sequence_duration(ctx) == 10);

  heif_track_release(track);
  heif_context_free(ctx);
}


TEST_CASE("failed sequence timescale rewrite keeps the old sequence and frame order")
{
  seqfile::SequenceFileParams p;
  p.timescale = 1000;
  p.mvhd_duration = 750;
  p.mdhd_duration = 750;
  p.with_editlist = false;
  p.stts = {{1, 100}, {1, 250}, {1, 400}};
  p.num_samples = 3;
  std::vector<uint8_t> file = seqfile::build_sequence_file(p);

  heif_context* ctx = read_context(file);
  heif_track* track = heif_context_get_track(ctx, 0);
  REQUIRE(track != nullptr);

  // Timescale 2 collapses the first frame (100/1000 s) to zero duration.
  heif_error err = heif_context_rewrite_sequence_timescale(ctx, 2);
  CHECK(err.code != heif_error_Ok);

  // The old sequence timing and the frame order are preserved.
  CHECK(heif_context_get_sequence_timescale(ctx) == 1000);
  CHECK(heif_context_get_sequence_duration(ctx) == 750);
  CHECK(heif_track_get_timescale(track) == 1000);
  std::vector<uint32_t> expected = {100, 250, 400};
  CHECK(read_sample_durations(track) == expected);

  // A zero timescale is rejected as well.
  err = heif_track_rewrite_timescale(track, 0);
  CHECK(err.code != heif_error_Ok);
  CHECK(heif_track_get_timescale(track) == 1000);

  heif_track_release(track);
  heif_context_free(ctx);
}


TEST_CASE("HF4: encoding with the original and the rewritten timebase gives the same timeline")
{
  if (!heif_have_encoder_for_format(heif_compression_uncompressed) ||
      !heif_have_decoder_for_format(heif_compression_uncompressed)) {
    SKIP("No uncompressed encoder/decoder available");
  }

  const uint32_t WIDTH = 64;
  const uint32_t HEIGHT = 64;
  const std::vector<uint32_t> durations = {1001, 1500, 1001}; // at timescale 30000

  auto encode_sequence = [&](uint32_t rewrite_timescale) {
    heif_context* ctx = heif_context_alloc();
    REQUIRE(ctx != nullptr);
    heif_context_set_sequence_timescale(ctx, 30000);

    heif_encoder* encoder = nullptr;
    REQUIRE(heif_context_get_encoder_for_format(ctx, heif_compression_uncompressed, &encoder).code == heif_error_Ok);
    REQUIRE(encoder != nullptr);

    heif_track_options* track_options = heif_track_options_alloc();
    REQUIRE(track_options != nullptr);
    heif_track_options_set_timescale(track_options, 30000);

    heif_track* track = nullptr;
    heif_error err = heif_context_add_visual_sequence_track(ctx, WIDTH, HEIGHT,
                                                            heif_track_type_image_sequence,
                                                            track_options, nullptr, &track);
    INFO("add_visual_sequence_track: " << (err.message ? err.message : ""));
    REQUIRE(err.code == heif_error_Ok);
    REQUIRE(track != nullptr);
    heif_track_options_release(track_options);

    for (uint32_t duration : durations) {
      heif_image* img = nullptr;
      REQUIRE(heif_image_create(WIDTH, HEIGHT, heif_colorspace_RGB, heif_chroma_interleaved_RGB, &img).code == heif_error_Ok);
      REQUIRE(img != nullptr);
      REQUIRE(heif_image_add_plane(img, heif_channel_interleaved, WIDTH, HEIGHT, 8).code == heif_error_Ok);
      heif_image_set_duration(img, duration);

      err = heif_track_encode_sequence_image(track, img, encoder, nullptr);
      INFO("encode_sequence_image: " << (err.message ? err.message : ""));
      REQUIRE(err.code == heif_error_Ok);
      heif_image_release(img);
    }

    REQUIRE(heif_track_encode_end_of_sequence(track, encoder).code == heif_error_Ok);

    if (rewrite_timescale != 0) {
      err = heif_context_rewrite_sequence_timescale(ctx, rewrite_timescale);
      INFO("rewrite_sequence_timescale: " << (err.message ? err.message : ""));
      REQUIRE(err.code == heif_error_Ok);
    }

    std::vector<uint8_t> file;
    heif_writer writer{};
    writer.writer_api_version = 1;
    writer.write = mem_writer;
    REQUIRE(heif_context_write(ctx, &writer, &file).code == heif_error_Ok);

    heif_track_release(track);
    heif_encoder_release(encoder);
    heif_context_free(ctx);
    return file;
  };

  // Write with the original timebase, and rewrite with the doubled timebase.
  std::vector<uint8_t> file_original = encode_sequence(0);
  std::vector<uint8_t> file_rewritten = encode_sequence(60000);
  REQUIRE(!file_original.empty());
  REQUIRE(!file_rewritten.empty());

  auto read_timing = [](const std::vector<uint8_t>& file, uint32_t expected_timescale,
                        const std::vector<uint32_t>& expected_durations,
                        uint64_t expected_sequence_duration) {
    heif_context* ctx = read_context(file);
    CHECK(heif_context_get_sequence_timescale(ctx) == expected_timescale);
    CHECK(heif_context_get_sequence_duration(ctx) == expected_sequence_duration);

    heif_track* track = heif_context_get_track(ctx, 0);
    REQUIRE(track != nullptr);
    CHECK(heif_track_get_timescale(track) == expected_timescale);
    CHECK(read_sample_durations(track) == expected_durations);

    heif_track_release(track);
    heif_context_free(ctx);
  };

  read_timing(file_original, 30000, {1001, 1500, 1001}, 3502);
  read_timing(file_rewritten, 60000, {2002, 3000, 2002}, 7004);

  // Both writings describe the same timeline: frame starts 0, 2002, 5002 and
  // total 7004 in units of 1/60000 s.
  std::vector<uint64_t> expected_starts = {0, 2002, 5002};
  CHECK(frame_starts_in_60000({1001, 1500, 1001}, 30000) == expected_starts);
  CHECK(frame_starts_in_60000({2002, 3000, 2002}, 60000) == expected_starts);
}
