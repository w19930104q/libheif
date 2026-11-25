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
  FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// End-to-end preservation of the overlay layer order, offsets and alpha across
// the write and the read sides of an 'iovl' item.
//
// The synthetic files use uncompressed ('unci') input images, so the test needs
// no codec plugin. Each input carries its alpha as a planar component of the
// same item (a four-component RGBA 'unci'), which the decoder hands to the
// overlay machinery as an RGB image with an Alpha plane.
//
// Two files are built from the same three layers: one with the original
// declaration order and one with two layer declarations swapped. The decoded
// composition is compared with a reference source-over compositor. The
// business rules checked here are:
//
//   * the layer array order is the composition order - nothing re-sorts by
//     image size or alpha,
//   * swapping two declarations swaps the visible color of pixels where the
//     two layers overlap, while each layer's own effective pixels (its color
//     and alpha where it alone covers the canvas) stay unchanged,
//   * a layer's transparent samples never write over pixels they do not cover.

#include "catch_amalgamated.hpp"
#include "libheif/heif.h"
#include "test_utils.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

const uint32_t CANVAS_W = 4;
const uint32_t CANVAS_H = 3;

struct Rgba {
  uint8_t r, g, b, a;
};

struct InputLayer
{
  heif_item_id item_id;
  uint32_t w, h;
  int32_t dx, dy;
  Rgba color;  // uniform color and alpha over the whole input image
};


// Pixel-interleaved, 8-bit per component RGBA item data.
std::vector<uint8_t> rgba_item_data(uint32_t w, uint32_t h, Rgba c)
{
  std::vector<uint8_t> data;
  data.reserve(static_cast<size_t>(w) * h * 4);
  for (uint32_t i = 0; i < w * h; i++) {
    data.push_back(c.r);
    data.push_back(c.g);
    data.push_back(c.b);
    data.push_back(c.a);
  }
  return data;
}


std::vector<uint8_t> make_ispe(uint32_t w, uint32_t h)
{
  std::vector<uint8_t> payload;
  put_u32_be(payload, w);
  put_u32_be(payload, h);
  return make_box("ispe", payload, /*full=*/true);
}

std::vector<uint8_t> make_cmpd_rgba()
{
  std::vector<uint8_t> payload;
  put_u32_be(payload, 4);
  put_u16_be(payload, heif_cmpd_component_type_red);
  put_u16_be(payload, heif_cmpd_component_type_green);
  put_u16_be(payload, heif_cmpd_component_type_blue);
  put_u16_be(payload, heif_cmpd_component_type_alpha);
  return make_box("cmpd", payload);
}

// uncC (v0): four 8-bit unsigned components, pixel interleave, one tile.
std::vector<uint8_t> make_uncC_rgba()
{
  std::vector<uint8_t> payload;
  put_u32_be(payload, 0);  // profile
  put_u32_be(payload, 4);  // component_count
  for (uint16_t c = 0; c < 4; c++) {
    put_u16_be(payload, c);  // component_index
    payload.push_back(7);    // bit_depth_minus_one -> 8 bit
    payload.push_back(0);    // unsigned
    payload.push_back(0);    // align_size
  }
  payload.push_back(0);  // sampling_type
  payload.push_back(1);  // interleave_type: pixel
  payload.push_back(0);  // block_size
  payload.push_back(0);  // flags
  put_u32_be(payload, 0);  // pixel_size
  put_u32_be(payload, 0);  // row_align_size
  put_u32_be(payload, 0);  // tile_align_size
  put_u32_be(payload, 0);  // tile_columns_minus_one
  put_u32_be(payload, 0);  // tile_rows_minus_one
  return make_box("uncC", payload, /*full=*/true);
}

std::vector<uint8_t> make_infe(heif_item_id id, const char* type, bool hidden)
{
  std::vector<uint8_t> payload;
  put_u16_be(payload, static_cast<uint16_t>(id));
  put_u16_be(payload, 0);
  append_fourcc(payload, type);
  append_cstr(payload, "");
  return make_box("infe", payload, /*full=*/true, /*version=*/2, /*flags=*/hidden ? 1u : 0u);
}

// ImageOverlay payload (ISO/IEC 23008-12 6.6.2.2), 16-bit fields: version,
// flags, background RGBA, canvas size, one signed (x,y) offset per layer.
std::vector<uint8_t> make_overlay_spec(const std::vector<const InputLayer*>& order)
{
  std::vector<uint8_t> s;
  s.push_back(0);  // version
  s.push_back(0);  // flags: 16-bit fields
  // transparent black background: its alpha must survive onto the canvas
  for (int i = 0; i < 4; i++) {
    put_u16_be(s, 0);
  }
  put_u16_be(s, static_cast<uint16_t>(CANVAS_W));
  put_u16_be(s, static_cast<uint16_t>(CANVAS_H));
  for (const InputLayer* l : order) {
    put_u16_be(s, static_cast<uint16_t>(static_cast<int16_t>(l->dx)));
    put_u16_be(s, static_cast<uint16_t>(static_cast<int16_t>(l->dy)));
  }
  return s;
}

// Build a file with one 'iovl' primary item referencing the input layers in the
// given declaration order.
std::vector<uint8_t> build_file(const std::vector<InputLayer>& layers,
                                const std::vector<size_t>& order,
                                heif_item_id iovl_id)
{
  std::vector<uint8_t> ftyp_payload;
  append_fourcc(ftyp_payload, "mif1");
  put_u32_be(ftyp_payload, 0);
  append_fourcc(ftyp_payload, "mif1");
  append_fourcc(ftyp_payload, "heic");
  auto ftyp = make_box("ftyp", ftyp_payload);

  std::vector<uint8_t> hdlr_payload;
  put_u32_be(hdlr_payload, 0);
  append_fourcc(hdlr_payload, "pict");
  for (int i = 0; i < 3; i++) {
    put_u32_be(hdlr_payload, 0);
  }
  hdlr_payload.push_back(0);
  auto hdlr = make_box("hdlr", hdlr_payload, /*full=*/true);

  std::vector<uint8_t> pitm_payload;
  put_u16_be(pitm_payload, static_cast<uint16_t>(iovl_id));
  auto pitm = make_box("pitm", pitm_payload, /*full=*/true);

  // iinf: the input layers, then the overlay item.
  std::vector<uint8_t> iinf_payload;
  put_u16_be(iinf_payload, static_cast<uint16_t>(layers.size() + 1));
  for (const InputLayer& l : layers) {
    append(iinf_payload, make_infe(l.item_id, "unci", /*hidden=*/true));
  }
  append(iinf_payload, make_infe(iovl_id, "iovl", /*hidden=*/false));
  auto iinf = make_box("iinf", iinf_payload, /*full=*/true);

  // ipco: each input gets ispe + cmpd + uncC; the overlay gets an ispe.
  std::vector<uint8_t> ipco_payload;
  for (const InputLayer& l : layers) {
    append(ipco_payload, make_ispe(l.w, l.h));
    append(ipco_payload, make_cmpd_rgba());
    append(ipco_payload, make_uncC_rgba());
  }
  const uint16_t overlay_ispe_index = static_cast<uint16_t>(1 + layers.size() * 3);
  append(ipco_payload, make_ispe(CANVAS_W, CANVAS_H));
  auto ipco = make_box("ipco", ipco_payload);

  std::vector<uint8_t> ipma_payload;
  put_u32_be(ipma_payload, static_cast<uint32_t>(layers.size() + 1));  // entry_count
  for (size_t i = 0; i < layers.size(); i++) {
    const uint16_t base = static_cast<uint16_t>(1 + i * 3);
    put_u16_be(ipma_payload, static_cast<uint16_t>(layers[i].item_id));
    ipma_payload.push_back(3);
    ipma_payload.push_back(static_cast<uint8_t>(0x80 | base));
    ipma_payload.push_back(static_cast<uint8_t>(0x80 | (base + 1)));
    ipma_payload.push_back(static_cast<uint8_t>(0x80 | (base + 2)));
  }
  put_u16_be(ipma_payload, static_cast<uint16_t>(iovl_id));
  ipma_payload.push_back(1);
  ipma_payload.push_back(static_cast<uint8_t>(0x80 | overlay_ispe_index));
  auto ipma = make_box("ipma", ipma_payload, /*full=*/true);

  std::vector<uint8_t> iprp_payload;
  append(iprp_payload, ipco);
  append(iprp_payload, ipma);
  auto iprp = make_box("iprp", iprp_payload);

  // idat: layer pixel data followed by the overlay spec.
  std::vector<uint8_t> idat_payload;
  struct Extent {
    heif_item_id id;
    uint32_t off, len;
  };
  std::vector<Extent> extents;
  for (const InputLayer& l : layers) {
    std::vector<uint8_t> data = rgba_item_data(l.w, l.h, l.color);
    extents.push_back({l.item_id, static_cast<uint32_t>(idat_payload.size()),
                       static_cast<uint32_t>(data.size())});
    append(idat_payload, data);
  }
  std::vector<const InputLayer*> ordered;
  for (size_t i : order) {
    ordered.push_back(&layers[i]);
  }
  std::vector<uint8_t> spec = make_overlay_spec(ordered);
  extents.push_back({iovl_id, static_cast<uint32_t>(idat_payload.size()),
                     static_cast<uint32_t>(spec.size())});
  append(idat_payload, spec);
  auto idat = make_box("idat", idat_payload);

  std::vector<uint8_t> iloc_payload;
  iloc_payload.push_back((4 << 4) | 4);  // offset_size=4, length_size=4
  iloc_payload.push_back(0);             // base_offset_size=0, index_size=0
  put_u16_be(iloc_payload, static_cast<uint16_t>(extents.size()));
  for (const Extent& e : extents) {
    put_u16_be(iloc_payload, static_cast<uint16_t>(e.id));
    put_u16_be(iloc_payload, 0x0001);  // construction_method=1 (idat)
    put_u16_be(iloc_payload, 0);       // data_reference_index
    put_u16_be(iloc_payload, 1);       // extent_count
    put_u32_be(iloc_payload, e.off);
    put_u32_be(iloc_payload, e.len);
  }
  auto iloc = make_box("iloc", iloc_payload, /*full=*/true, /*version=*/1);

  // iref: one dimg entry, the to-item order is the layer composition order.
  std::vector<uint8_t> dimg_payload;
  put_u16_be(dimg_payload, static_cast<uint16_t>(iovl_id));
  put_u16_be(dimg_payload, static_cast<uint16_t>(order.size()));
  for (size_t i : order) {
    put_u16_be(dimg_payload, static_cast<uint16_t>(layers[i].item_id));
  }
  auto iref = make_box("iref", make_box("dimg", dimg_payload), /*full=*/true);

  std::vector<uint8_t> meta_payload;
  append(meta_payload, hdlr);
  append(meta_payload, pitm);
  append(meta_payload, iinf);
  append(meta_payload, iprp);
  append(meta_payload, iloc);
  append(meta_payload, iref);
  append(meta_payload, idat);
  auto meta = make_box("meta", meta_payload, /*full=*/true);

  std::vector<uint8_t> file;
  append(file, ftyp);
  append(file, meta);
  return file;
}


// Decode the primary image to planar 8-bit RGBA.
struct Decoded
{
  uint32_t w, h;
  std::vector<uint8_t> r, g, b, a;
};

Decoded decode_primary_rgba(heif_context* ctx)
{
  heif_image_handle* handle = nullptr;
  REQUIRE(heif_context_get_primary_image_handle(ctx, &handle).code == heif_error_Ok);

  heif_image* img = nullptr;
  heif_error err = heif_decode_image(handle, &img, heif_colorspace_RGB, heif_chroma_444, nullptr);
  INFO("decode error: " << err.message);
  REQUIRE(err.code == heif_error_Ok);

  Decoded out;
  out.w = static_cast<uint32_t>(heif_image_get_width(img, heif_channel_R));
  out.h = static_cast<uint32_t>(heif_image_get_height(img, heif_channel_R));
  REQUIRE(heif_image_has_channel(img, heif_channel_Alpha));

  const heif_channel channels[4] = {heif_channel_R, heif_channel_G, heif_channel_B, heif_channel_Alpha};
  std::vector<uint8_t>* planes[4] = {&out.r, &out.g, &out.b, &out.a};
  for (int c = 0; c < 4; c++) {
    int stride = 0;
    const uint8_t* p = heif_image_get_plane_readonly(img, channels[c], &stride);
    REQUIRE(p != nullptr);
    planes[c]->resize(static_cast<size_t>(out.w) * out.h);
    for (uint32_t y = 0; y < out.h; y++) {
      memcpy(&(*planes[c])[y * out.w], p + y * stride, out.w);
    }
  }

  heif_image_release(img);
  heif_image_handle_release(handle);
  return out;
}


// Independent straight-alpha source-over reference, starting on a transparent
// black canvas.
struct Reference
{
  uint32_t w, h;
  std::vector<uint32_t> r, g, b, a;

  Reference(uint32_t w_, uint32_t h_)
      : w(w_), h(h_), r(w_ * h_, 0), g(w_ * h_, 0), b(w_ * h_, 0), a(w_ * h_, 0) {}

  void paint(const InputLayer& l)
  {
    for (uint32_t y = 0; y < l.h; y++) {
      for (uint32_t x = 0; x < l.w; x++) {
        const int64_t cx = static_cast<int64_t>(l.dx) + x;
        const int64_t cy = static_cast<int64_t>(l.dy) + y;
        if (cx < 0 || cy < 0 || cx >= static_cast<int64_t>(w) || cy >= static_cast<int64_t>(h)) {
          continue;
        }
        const size_t idx = static_cast<size_t>(cy) * w + static_cast<size_t>(cx);
        const uint32_t As = l.color.a;
        const uint32_t Ab = a[idx];
        const uint32_t rest = 255 - As;
        const uint32_t Aout = As + Ab * rest / 255;
        a[idx] = Aout;
        if (Aout == 0) {
          r[idx] = g[idx] = b[idx] = 0;
        }
        else {
          const uint32_t cw = Ab * rest / 255;
          r[idx] = (l.color.r * As + r[idx] * cw) / Aout;
          g[idx] = (l.color.g * As + g[idx] * cw) / Aout;
          b[idx] = (l.color.b * As + b[idx] * cw) / Aout;
        }
      }
    }
  }
};

void require_matches(const Decoded& d, const Reference& ref)
{
  REQUIRE(d.w == ref.w);
  REQUIRE(d.h == ref.h);
  for (size_t i = 0; i < ref.r.size(); i++) {
    const uint32_t x = static_cast<uint32_t>(i % ref.w);
    const uint32_t y = static_cast<uint32_t>(i / ref.w);
    INFO("pixel (" << x << "," << y << ")");
    REQUIRE(d.r[i] == ref.r[i]);
    REQUIRE(d.g[i] == ref.g[i]);
    REQUIRE(d.b[i] == ref.b[i]);
    REQUIRE(d.a[i] == ref.a[i]);
  }
}

} // namespace


// HF3 sample: a 4x3 opaque base, a semi-transparent badge at (1,0) and an
// opaque corner mark at (2,1). The badge and corner do not overlap. Swapping
// their declaration order must not change any decoded pixel, and every layer's
// own effective pixels must survive the write/read round trip.
TEST_CASE("iovl layer order: disjoint HF3 layers and alpha survive write and read")
{
  const heif_item_id base_id = 1, badge_id = 2, corner_id = 3, iovl_id = 4;

  std::vector<InputLayer> layers = {
      {base_id, CANVAS_W, CANVAS_H, 0, 0, {10, 20, 30, 255}},
      {badge_id, 1, 1, 1, 0, {200, 100, 50, 128}},
      {corner_id, 1, 1, 2, 1, {0, 0, 200, 255}},
  };

  for (const std::vector<size_t>& order : {std::vector<size_t>{0, 1, 2},
                                           std::vector<size_t>{0, 2, 1}}) {
    auto bytes = build_file(layers, order, iovl_id);

    heif_context* ctx = heif_context_alloc();
    REQUIRE(ctx != nullptr);
    heif_error err = heif_context_read_from_memory_without_copy(ctx, bytes.data(), bytes.size(), nullptr);
    INFO("read error: " << err.message);
    REQUIRE(err.code == heif_error_Ok);

    Decoded decoded = decode_primary_rgba(ctx);

    Reference ref(CANVAS_W, CANVAS_H);
    for (size_t i : order) {
      ref.paint(layers[i]);
    }
    require_matches(decoded, ref);

    // Explicit HF3 expectations: the badge pixel blends onto the opaque base and
    // is itself opaque; the corner replaces the base at its single pixel.
    const size_t badge_idx = 0 * CANVAS_W + 1;
    CHECK(decoded.a[badge_idx] == 255);
    CHECK(decoded.r[badge_idx] == static_cast<uint8_t>((200u * 128u + 10u * 127u) / 255u));

    const size_t corner_idx = 1 * CANVAS_W + 2;
    CHECK(decoded.r[corner_idx] == 0);
    CHECK(decoded.g[corner_idx] == 0);
    CHECK(decoded.b[corner_idx] == 200);
    CHECK(decoded.a[corner_idx] == 255);

    heif_context_free(ctx);
  }
}


// Two semi-transparent layers overlap at one pixel. Swapping the declaration
// order swaps the overlap color, leaves pixels covered by only one layer
// untouched, and leaves the background transparent where no layer covers it.
TEST_CASE("iovl layer order: swapping declarations swaps only the overlap color")
{
  const heif_item_id red_id = 1, green_id = 2, iovl_id = 3;

  // 2x1 red badge at (0,0): exclusive pixel (0,0), shared pixel (1,0).
  std::vector<InputLayer> layers = {
      {red_id, 2, 1, 0, 0, {200, 0, 0, 128}},
      {green_id, 1, 1, 1, 0, {0, 200, 0, 128}},
  };

  auto bytes_abc = build_file(layers, {0, 1}, iovl_id);
  auto bytes_cba = build_file(layers, {1, 0}, iovl_id);

  heif_context* ctx1 = heif_context_alloc();
  heif_context* ctx2 = heif_context_alloc();
  REQUIRE(heif_context_read_from_memory_without_copy(ctx1, bytes_abc.data(), bytes_abc.size(), nullptr).code
          == heif_error_Ok);
  REQUIRE(heif_context_read_from_memory_without_copy(ctx2, bytes_cba.data(), bytes_cba.size(), nullptr).code
          == heif_error_Ok);

  Decoded red_first = decode_primary_rgba(ctx1);
  Decoded green_first = decode_primary_rgba(ctx2);

  Reference ref1(CANVAS_W, CANVAS_H);
  ref1.paint(layers[0]);
  ref1.paint(layers[1]);
  require_matches(red_first, ref1);

  Reference ref2(CANVAS_W, CANVAS_H);
  ref2.paint(layers[1]);
  ref2.paint(layers[0]);
  require_matches(green_first, ref2);

  // Exclusive red pixel: identical in both orders and half transparent.
  CHECK(red_first.r[0] == green_first.r[0]);
  CHECK(red_first.a[0] == green_first.a[0]);
  CHECK(red_first.a[0] == 128);

  // Overlap pixel: different colors, but the accumulated alpha is the same.
  const size_t overlap = 1;
  CHECK(red_first.r[overlap] != green_first.r[overlap]);
  CHECK(red_first.g[overlap] != green_first.g[overlap]);
  CHECK(red_first.a[overlap] == green_first.a[overlap]);

  // Uncovered background stays fully transparent.
  for (size_t i = 2; i < CANVAS_W * CANVAS_H; i++) {
    CHECK(red_first.a[i] == 0);
    CHECK(green_first.a[i] == 0);
  }

  heif_context_free(ctx1);
  heif_context_free(ctx2);
}


// The overlay written through the public heif_context_add_overlay_image() API
// keeps the same order, offsets and alpha when the file is read back.
TEST_CASE("iovl layer order: API-written overlay round trip preserves order and alpha")
{
  const heif_item_id base_id = 1, badge_id = 2, corner_id = 3;
  (void) base_id;
  (void) badge_id;
  (void) corner_id;

  std::vector<InputLayer> layers = {
      {1, CANVAS_W, CANVAS_H, 0, 0, {10, 20, 30, 255}},
      {2, 1, 1, 1, 0, {200, 100, 50, 128}},
      {3, 1, 1, 2, 1, {0, 0, 200, 255}},
  };

  // First hand-build a file, read it, then re-emit the same overlay through the
  // writing API and decode the result again. The writing API needs an encoder
  // for the input images; the uncompressed encoder is built into this
  // configuration, so no external plugin is required.
  auto bytes = build_file(layers, {0, 1, 2}, 4);

  heif_context* read_ctx = heif_context_alloc();
  REQUIRE(heif_context_read_from_memory_without_copy(read_ctx, bytes.data(), bytes.size(), nullptr).code
          == heif_error_Ok);

  heif_context* write_ctx = heif_context_alloc();
  heif_item_id written_ids[3];

  for (int i = 0; i < 3; i++) {
    heif_image_handle* in_handle = nullptr;
    REQUIRE(heif_context_get_image_handle(read_ctx, layers[i].item_id, &in_handle).code == heif_error_Ok);

    heif_image* in_img = nullptr;
    REQUIRE(heif_decode_image(in_handle, &in_img, heif_colorspace_RGB, heif_chroma_444, nullptr).code
            == heif_error_Ok);

    heif_encoder* encoder = nullptr;
    heif_error enc_err = heif_context_get_encoder_for_format(write_ctx, heif_compression_uncompressed, &encoder);
    INFO("encoder error: " << enc_err.message);
    REQUIRE(enc_err.code == heif_error_Ok);

    heif_image_handle* out_handle = nullptr;
    REQUIRE(heif_context_encode_image(write_ctx, in_img, encoder, nullptr, &out_handle).code
            == heif_error_Ok);
    written_ids[i] = heif_image_handle_get_item_id(out_handle);

    heif_image_handle_release(out_handle);
    heif_encoder_release(encoder);
    heif_image_release(in_img);
    heif_image_handle_release(in_handle);
  }

  int32_t offsets[6] = {0, 0, 1, 0, 2, 1};
  uint16_t background[4] = {0, 0, 0, 0};  // transparent black
  heif_image_handle* iovl_handle = nullptr;
  REQUIRE(heif_context_add_overlay_image(write_ctx, CANVAS_W, CANVAS_H, 3, written_ids,
                                         offsets, background, &iovl_handle)
                  .code == heif_error_Ok);
  REQUIRE(heif_context_set_primary_image(write_ctx, iovl_handle).code == heif_error_Ok);

  std::string path = get_tests_output_file_path("overlay_layer_order_roundtrip.heif");
  REQUIRE(heif_context_write_to_file(write_ctx, path.c_str()).code == heif_error_Ok);

  heif_image_handle_release(iovl_handle);
  heif_context_free(write_ctx);
  heif_context_free(read_ctx);

  heif_context* reread = heif_context_alloc();
  REQUIRE(heif_context_read_from_file(reread, path.c_str(), nullptr).code == heif_error_Ok);

  Decoded decoded = decode_primary_rgba(reread);
  Reference ref(CANVAS_W, CANVAS_H);
  for (const InputLayer& l : layers) {
    ref.paint(l);
  }
  require_matches(decoded, ref);

  heif_context_free(reread);
}
