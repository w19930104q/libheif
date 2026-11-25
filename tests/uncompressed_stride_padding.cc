/*
  libheif unit tests

  MIT License

  Copyright (c) 2026 Dirk Farin <dirk.farin@gmail.com>

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

// Regression tests for the handling of row-end padding in uncompressed (unci)
// high-bit-depth images.
//
// A microscopy platform stores planar images with a row stride that is wider
// than the valid samples: a 3x2, 10-bit grayscale image occupies 6 valid bytes
// per row, but the platform lays each row out on, for example, 10 bytes and
// fills the tail padding with a marker value. Such a stride changes only the
// storage layout, never the 2x3 valid image. Both sides of the codec therefore
// have to process exactly the valid samples:
//
//   * the encoder reads one row at a time through the plane's stride and must
//     never copy a padding byte into the coded tile (it would re-enter the
//     valid samples after a round trip and pollute the third column);
//   * the decoder writes exactly the samples declared by 'uncC' and must never
//     touch the padding region of the destination plane;
//   * a tile whose bytes run out before the declared samples have been read is
//     an error, not a successful decode of a half-filled, zero-padded image.
//
// The first test builds HF1 through the public API (the library allocator
// chooses its own wide stride), marks every padding byte, and verifies the
// coded tile and the round-tripped image. The second test hand-builds unci
// files so that the decoder-side row-alignment and short-data handling can be
// exercised directly.

#include "catch_amalgamated.hpp"
#include "libheif/heif.h"
#include "api_structs.h"
#include "image/pixelimage.h"
#include "codecs/uncompressed/unc_encoder.h"
#include "codecs/uncompressed/unc_boxes.h"
#include "codecs/uncompressed/unc_decoder.h"
#include "codecs/uncompressed/unc_types.h"
#include "test_utils.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int W = 3;
constexpr int H = 2;
constexpr int BPP = 10;

const uint16_t y_values[H][W] = {
    {100, 200, 300},
    {400, 500, 600},
};

const uint16_t a_values[H][W] = {
    {900, 800, 700},
    {650, 550, 450},
};

constexpr uint8_t PAD = 0x7E;

// Dense MSB-first packing of 'width' samples of 'bpp' bits, one row per call.
void append_packed_row(std::vector<uint8_t>& out, const uint16_t samples[],
                       uint32_t width, uint16_t bpp)
{
  uint64_t accumulator = 0;
  int accumulated_bits = 0;

  for (uint32_t x = 0; x < width; x++) {
    accumulator = (accumulator << bpp) | samples[x];
    accumulated_bits += bpp;
    while (accumulated_bits >= 8) {
      accumulated_bits -= 8;
      out.push_back(static_cast<uint8_t>(accumulator >> accumulated_bits));
      accumulator &= (uint64_t{1} << accumulated_bits) - 1;
    }
  }

  // The final partial byte is padded with zero bits on the LSB side.
  if (accumulated_bits > 0) {
    out.push_back(static_cast<uint8_t>(accumulator << (8 - accumulated_bits)));
  }
}

// Hand-built two-component (monochrome + alpha), component-interleave unci
// file. Both components are 10 bit, so the tile data is densely bit-packed.
// 'row_align_size' is written into uncC and 'item_data' is stored verbatim,
// which lets tests feed malformed or padded data.
std::vector<uint8_t> build_mono_alpha_unci(uint32_t width, uint32_t height,
                                           uint32_t row_align_size,
                                           const std::vector<uint8_t>& item_data)
{
  std::vector<uint8_t> ftyp_payload;
  append_fourcc(ftyp_payload, "mif1");
  put_u32_be(ftyp_payload, 0);
  append_fourcc(ftyp_payload, "mif1");
  append_fourcc(ftyp_payload, "heic");
  std::vector<uint8_t> ftyp = make_box("ftyp", ftyp_payload);

  std::vector<uint8_t> hdlr_payload;
  put_u32_be(hdlr_payload, 0);
  append_fourcc(hdlr_payload, "pict");
  put_u32_be(hdlr_payload, 0);
  put_u32_be(hdlr_payload, 0);
  put_u32_be(hdlr_payload, 0);
  hdlr_payload.push_back(0);
  std::vector<uint8_t> hdlr = make_box("hdlr", hdlr_payload, true);

  std::vector<uint8_t> pitm_payload;
  put_u16_be(pitm_payload, 1);
  std::vector<uint8_t> pitm = make_box("pitm", pitm_payload, true);

  std::vector<uint8_t> infe_payload;
  put_u16_be(infe_payload, 1);
  put_u16_be(infe_payload, 0);
  append_fourcc(infe_payload, "unci");
  append_cstr(infe_payload, "");
  std::vector<uint8_t> infe = make_box("infe", infe_payload, true, 2);

  std::vector<uint8_t> iinf_payload;
  put_u16_be(iinf_payload, 1);
  append(iinf_payload, infe);
  std::vector<uint8_t> iinf = make_box("iinf", iinf_payload, true);

  std::vector<uint8_t> ispe_payload;
  put_u32_be(ispe_payload, width);
  put_u32_be(ispe_payload, height);
  std::vector<uint8_t> ispe = make_box("ispe", ispe_payload, true);

  std::vector<uint8_t> cmpd_payload;
  put_u32_be(cmpd_payload, 2);
  put_u16_be(cmpd_payload, static_cast<uint16_t>(heif_cmpd_component_type_monochrome));
  put_u16_be(cmpd_payload, static_cast<uint16_t>(heif_cmpd_component_type_alpha));
  std::vector<uint8_t> cmpd = make_box("cmpd", cmpd_payload);

  std::vector<uint8_t> uncC_payload;
  put_u32_be(uncC_payload, 0); // profile
  put_u32_be(uncC_payload, 2); // number of components
  for (uint16_t c = 0; c < 2; c++) {
    put_u16_be(uncC_payload, c);  // component_index
    uncC_payload.push_back(BPP - 1); // bit depth minus one -> 10 bit
    uncC_payload.push_back(0);    // unsigned integer
    uncC_payload.push_back(0);    // component_align_size (dense bit packing)
  }
  uncC_payload.push_back(0);                 // sampling: no subsampling
  uncC_payload.push_back(0);                 // interleave: component
  uncC_payload.push_back(0);                 // block size
  uncC_payload.push_back(0);                 // flags
  put_u32_be(uncC_payload, 0);               // pixel size
  put_u32_be(uncC_payload, row_align_size);
  put_u32_be(uncC_payload, 0);               // tile align size
  put_u32_be(uncC_payload, 0);               // tile columns minus one
  put_u32_be(uncC_payload, 0);               // tile rows minus one
  std::vector<uint8_t> uncC = make_box("uncC", uncC_payload, true);

  std::vector<uint8_t> ipco_payload;
  append(ipco_payload, ispe);
  append(ipco_payload, cmpd);
  append(ipco_payload, uncC);
  std::vector<uint8_t> ipco = make_box("ipco", ipco_payload);

  std::vector<uint8_t> ipma_payload;
  put_u32_be(ipma_payload, 1);
  put_u16_be(ipma_payload, 1);
  ipma_payload.push_back(3);
  ipma_payload.push_back(0x80 | 1);
  ipma_payload.push_back(0x80 | 2);
  ipma_payload.push_back(0x80 | 3);
  std::vector<uint8_t> ipma = make_box("ipma", ipma_payload, true);

  std::vector<uint8_t> iprp_payload;
  append(iprp_payload, ipco);
  append(iprp_payload, ipma);
  std::vector<uint8_t> iprp = make_box("iprp", iprp_payload);

  std::vector<uint8_t> idat = make_box("idat", item_data);

  std::vector<uint8_t> iloc_payload;
  put_u16_be(iloc_payload, (4 << 12) | (4 << 8)); // offset/length sizes of 4
  put_u16_be(iloc_payload, 1);                     // item count
  put_u16_be(iloc_payload, 1);                     // item ID
  put_u16_be(iloc_payload, 0x0001);                // construction method: idat
  put_u16_be(iloc_payload, 0);                     // data reference index
  put_u16_be(iloc_payload, 1);                     // extent count
  put_u32_be(iloc_payload, 0);
  put_u32_be(iloc_payload, static_cast<uint32_t>(item_data.size()));
  std::vector<uint8_t> iloc = make_box("iloc", iloc_payload, true, 1);

  // idat has to be the last box of meta.
  std::vector<uint8_t> meta_payload;
  append(meta_payload, hdlr);
  append(meta_payload, pitm);
  append(meta_payload, iinf);
  append(meta_payload, iprp);
  append(meta_payload, iloc);
  append(meta_payload, idat);
  std::vector<uint8_t> meta = make_box("meta", meta_payload, true);

  std::vector<uint8_t> file;
  append(file, ftyp);
  append(file, meta);
  return file;
}

heif_error decode_primary(const std::vector<uint8_t>& file, heif_image** out_image,
                          heif_context** out_ctx, heif_image_handle** out_handle)
{
  heif_context* ctx = heif_context_alloc();
  REQUIRE(ctx != nullptr);

  heif_error err = heif_context_read_from_memory_without_copy(ctx, file.data(), file.size(), nullptr);
  if (err.code != heif_error_Ok) {
    heif_context_free(ctx);
    return err;
  }

  heif_image_handle* handle = nullptr;
  err = heif_context_get_primary_image_handle(ctx, &handle);
  if (err.code != heif_error_Ok) {
    heif_context_free(ctx);
    return err;
  }

  heif_image* image = nullptr;
  err = heif_decode_image(handle, &image, heif_colorspace_undefined,
                          heif_chroma_undefined, nullptr);
  if (err.code != heif_error_Ok) {
    heif_image_handle_release(handle);
    heif_context_free(ctx);
    return err;
  }

  *out_image = image;
  *out_handle = handle;
  *out_ctx = ctx;
  return err;
}

} // namespace

// HF1: encode a 3x2, 10-bit monochrome + alpha image whose planes carry marked
// row-end padding. The coded tile must contain only the dense-packed valid
// samples, and the full encode/decode round trip must restore every value
// while leaving the plane's padding region under the decoder allocator's
// control (never written as sample content).
TEST_CASE("HF1 narrow-stride padding never enters the uncompressed tile")
{
  heif_image* image = nullptr;
  heif_error err = heif_image_create(W, H, heif_colorspace_monochrome,
                                     heif_chroma_monochrome, &image);
  REQUIRE(err.code == heif_error_Ok);
  REQUIRE(heif_image_add_plane(image, heif_channel_Y, W, H, BPP).code == heif_error_Ok);
  REQUIRE(heif_image_add_plane(image, heif_channel_Alpha, W, H, BPP).code == heif_error_Ok);

  int y_stride = 0;
  int a_stride = 0;
  uint8_t* yp = heif_image_get_plane(image, heif_channel_Y, &y_stride);
  uint8_t* ap = heif_image_get_plane(image, heif_channel_Alpha, &a_stride);
  REQUIRE(yp != nullptr);
  REQUIRE(ap != nullptr);
  // The library allocator pads the row; the valid region is only W*2 bytes.
  REQUIRE(y_stride >= W * 2);
  REQUIRE(a_stride >= W * 2);

  for (int y = 0; y < H; y++) {
    reinterpret_cast<uint16_t*>(yp + y * y_stride)[0] = y_values[y][0];
    reinterpret_cast<uint16_t*>(yp + y * y_stride)[1] = y_values[y][1];
    reinterpret_cast<uint16_t*>(yp + y * y_stride)[2] = y_values[y][2];
    reinterpret_cast<uint16_t*>(ap + y * a_stride)[0] = a_values[y][0];
    reinterpret_cast<uint16_t*>(ap + y * a_stride)[1] = a_values[y][1];
    reinterpret_cast<uint16_t*>(ap + y * a_stride)[2] = a_values[y][2];

    std::memset(yp + y * y_stride + W * 2, PAD, y_stride - W * 2);
    std::memset(ap + y * a_stride + W * 2, PAD, a_stride - W * 2);
  }

  // --- inspect the coded tile through the internal encoder factory
  auto pixel_image = reinterpret_cast<heif_image*>(image)->image;

  heif_encoding_options* options = heif_encoding_options_alloc();
  auto enc_result = unc_encoder_factory::get_unc_encoder(pixel_image, *options);
  REQUIRE(enc_result.error().error_code == heif_error_Ok);

  auto tile_result = (*enc_result)->encode_tile(pixel_image);
  REQUIRE(tile_result.error().error_code == heif_error_Ok);
  const std::vector<uint8_t>& tile = *tile_result;

  // Dense packing: 3 samples of 10 bits = 30 bits = 4 bytes per row,
  // 2 rows per component, 2 components -> 16 bytes, no padding.
  constexpr size_t kBytesPerRow = 4;
  REQUIRE(tile.size() == static_cast<size_t>(kBytesPerRow * H * 2));
  for (uint8_t byte : tile) {
    CHECK(byte != PAD);
  }

  // The tile must be exactly the independently computed dense bitstream.
  std::vector<uint8_t> expected;
  for (int y = 0; y < H; y++) {
    append_packed_row(expected, y_values[y], W, BPP);
  }
  for (int y = 0; y < H; y++) {
    append_packed_row(expected, a_values[y], W, BPP);
  }
  REQUIRE(tile == expected);

  // --- full public API round trip
  heif_context* ctx = heif_context_alloc();
  heif_encoder* enc = nullptr;
  REQUIRE(heif_context_get_encoder_for_format(ctx, heif_compression_uncompressed, &enc).code == heif_error_Ok);
  options->macOS_compatibility_workaround_no_nclx_profile = true;

  heif_image_handle* handle = nullptr;
  REQUIRE(heif_context_encode_image(ctx, image, enc, options, &handle).code == heif_error_Ok);

  std::string out_path = get_tests_output_file_path("hf1_narrow_stride.heif");
  REQUIRE(heif_context_write_to_file(ctx, out_path.c_str()).code == heif_error_Ok);

  heif_image_handle_release(handle);
  heif_encoder_release(enc);
  heif_encoding_options_free(options);
  heif_context_free(ctx);
  heif_image_release(image);

  // --- decode and compare every valid sample
  heif_context* dctx = heif_context_alloc();
  REQUIRE(heif_context_read_from_file(dctx, out_path.c_str(), nullptr).code == heif_error_Ok);
  heif_image_handle* dh = get_primary_image_handle(dctx);

  heif_image* decoded = nullptr;
  REQUIRE(heif_decode_image(dh, &decoded, heif_colorspace_undefined,
                            heif_chroma_undefined, nullptr).code == heif_error_Ok);

  int ds = 0;
  const auto* dy = reinterpret_cast<const uint16_t*>(
      heif_image_get_plane_readonly(decoded, heif_channel_Y, &ds));
  int das = 0;
  const auto* da = reinterpret_cast<const uint16_t*>(
      heif_image_get_plane_readonly(decoded, heif_channel_Alpha, &das));
  REQUIRE(dy != nullptr);
  REQUIRE(da != nullptr);
  // Alpha stays an independently readable plane.
  CHECK(heif_image_has_channel(decoded, heif_channel_Alpha));

  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      INFO("Y at " << y << "," << x);
      CHECK(dy[y * (ds / 2) + x] == y_values[y][x]);
      INFO("Alpha at " << y << "," << x);
      CHECK(da[y * (das / 2) + x] == a_values[y][x]);
    }
  }

  heif_image_release(decoded);
  heif_image_handle_release(dh);
  heif_context_free(dctx);
}

// Row alignment padding declared through uncC is storage layout, not content.
// Build a file whose rows carry row_align padding between them and verify that
// only the three valid samples of each row are decoded.
TEST_CASE("unci row alignment padding is not decoded as samples")
{
  constexpr uint32_t kRowAlign = 8;
  // Each densely packed row is 4 bytes, padded to 8; two components, two rows.
  std::vector<uint8_t> item_data;
  for (int y = 0; y < H; y++) {
    append_packed_row(item_data, y_values[y], W, BPP);
    item_data.resize(item_data.size() + kRowAlign - 4, PAD);
  }
  for (int y = 0; y < H; y++) {
    append_packed_row(item_data, a_values[y], W, BPP);
    item_data.resize(item_data.size() + kRowAlign - 4, PAD);
  }
  REQUIRE(item_data.size() == kRowAlign * H * 2);

  std::vector<uint8_t> file = build_mono_alpha_unci(W, H, kRowAlign, item_data);

  heif_image* decoded = nullptr;
  heif_context* ctx = nullptr;
  heif_image_handle* handle = nullptr;
  heif_error err = decode_primary(file, &decoded, &ctx, &handle);
  INFO("decode error: " << err.message);
  REQUIRE(err.code == heif_error_Ok);

  int ds = 0;
  const auto* dy = reinterpret_cast<const uint16_t*>(
      heif_image_get_plane_readonly(decoded, heif_channel_Y, &ds));
  int das = 0;
  const auto* da = reinterpret_cast<const uint16_t*>(
      heif_image_get_plane_readonly(decoded, heif_channel_Alpha, &das));
  REQUIRE(dy != nullptr);
  REQUIRE(da != nullptr);

  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      INFO("Y at " << y << "," << x);
      CHECK(dy[y * (ds / 2) + x] == y_values[y][x]);
      INFO("Alpha at " << y << "," << x);
      CHECK(da[y * (das / 2) + x] == a_values[y][x]);
    }
  }

  heif_image_release(decoded);
  heif_image_handle_release(handle);
  heif_context_free(ctx);
}

// A tile that ends before all declared samples have been read must be rejected
// by the tile decoder itself. The 'iloc' extent length usually catches a short
// item first, but the decoder must not depend on that: BitReader returns zero
// bits past the end of its buffer, so without its own check the missing
// samples would be stored as zeros and the decode would still report success,
// leaving a half-filled target image. This drives decode_tile() directly with
// an undersized buffer so the decoder's own guard is exercised.
TEST_CASE("unc decoder rejects a tile buffer shorter than the uncC layout")
{
  auto cmpd = std::make_shared<Box_cmpd>();
  cmpd->set_components({static_cast<uint16_t>(heif_cmpd_component_type_monochrome),
                        static_cast<uint16_t>(heif_cmpd_component_type_alpha)});

  auto uncC = std::make_shared<Box_uncC>();
  uncC->set_profile(0);
  uncC->set_sampling_type(sampling_mode_no_subsampling);
  uncC->set_interleave_type(interleave_mode_component);
  uncC->set_block_size(0);
  uncC->set_components_little_endian(false);
  uncC->add_component({0, BPP, component_format_unsigned, 0});
  uncC->add_component({1, BPP, component_format_unsigned, 0});

  auto img = std::make_shared<HeifPixelImage>();
  img->create(W, H, heif_colorspace_undefined, heif_chroma_undefined);
  auto id0_result = img->add_component(W, H, heif_cmpd_component_type_monochrome,
                                       heif_component_datatype_unsigned_integer, BPP, nullptr);
  auto id1_result = img->add_component(W, H, heif_cmpd_component_type_alpha,
                                       heif_component_datatype_unsigned_integer, BPP, nullptr);
  REQUIRE(!id0_result.is_error());
  REQUIRE(!id1_result.is_error());
  const uint32_t id0 = *id0_result;
  const uint32_t id1 = *id1_result;

  auto decoderResult = unc_decoder_factory::get_unc_decoder(W, H, cmpd, uncC, {id0, id1});
  REQUIRE(!decoderResult.is_error());
  auto& decoder = *decoderResult;

  decoder->ensure_channel_list(img);

  std::vector<uint8_t> full_tile;
  for (int y = 0; y < H; y++) {
    append_packed_row(full_tile, y_values[y], W, BPP);
  }
  for (int y = 0; y < H; y++) {
    append_packed_row(full_tile, a_values[y], W, BPP);
  }
  REQUIRE(full_tile.size() == 16);

  // The correctly sized tile decodes without error.
  std::shared_ptr<HeifPixelImage> target = img;
  Error err = decoder->decode_tile(full_tile, target, 0, 0);
  REQUIRE(!err);

  // A buffer missing part of the second component must be rejected rather than
  // decoded with zero-filled alpha samples.
  std::vector<uint8_t> short_tile(full_tile.begin(), full_tile.end() - 3);
  err = decoder->decode_tile(short_tile, target, 0, 0);
  CHECK(err.error_code == heif_error_Invalid_input);
  CHECK(err.sub_error_code == heif_suberror_End_of_data);
}

