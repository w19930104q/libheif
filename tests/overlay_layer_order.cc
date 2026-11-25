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

// Layer order, offset and alpha preservation for HeifPixelImage::overlay()
// when the canvas itself carries an alpha plane, as the 'iovl' decoder builds
// it.
//
// The earlier implementation only blended the colors and discarded the
// background alpha: a semi-transparent lower layer was composited as if the
// canvas were fully opaque, so it "covered" whatever was drawn above it in the
// overlapping area, and the transparent parts of a layer could not leave the
// pixels beneath them untouched. The canvas now keeps an alpha plane and every
// layer is composited with the straight-alpha source-over rule:
//
//   Aout = As + Ab * (1 - As)
//   Cout = (Cs * As + Cb * Ab * (1 - As)) / Aout
//
// Every test builds the same small layers, composites them onto an alpha
// canvas in two declaration orders, and compares the result with an
// independent reference implementation that applies the same formula to plain
// vectors.

#include "image/pixelimage.h"
#include "catch_amalgamated.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace {

const uint32_t CANVAS_W = 4;
const uint32_t CANVAS_H = 3;

struct Rgba {
  uint8_t r, g, b, a;
};

struct LayerSpec {
  uint32_t w, h;
  int32_t dx, dy;
  bool has_alpha;
  std::vector<Rgba> px;  // size w*h; 'a' is ignored when has_alpha == false (opaque)
};


// A canvas initialized to a uniform RGBA fill.
std::shared_ptr<HeifPixelImage> make_canvas(Rgba fill)
{
  auto img = std::make_shared<HeifPixelImage>();
  img->create(CANVAS_W, CANVAS_H, heif_colorspace_RGB, heif_chroma_444);

  for (heif_channel ch : {heif_channel_R, heif_channel_G, heif_channel_B, heif_channel_Alpha}) {
    REQUIRE(img->add_channel(ch, CANVAS_W, CANVAS_H, 8, heif_get_global_security_limits()).error_code == heif_error_Ok);
  }

  size_t stride = 0;
  uint8_t* p_r = img->get_channel_memory(heif_channel_R, &stride);
  uint8_t* p_g = img->get_channel_memory(heif_channel_G, &stride);
  uint8_t* p_b = img->get_channel_memory(heif_channel_B, &stride);
  uint8_t* p_a = img->get_channel_memory(heif_channel_Alpha, &stride);

  for (uint32_t y = 0; y < CANVAS_H; y++) {
    memset(p_r + y * stride, fill.r, CANVAS_W);
    memset(p_g + y * stride, fill.g, CANVAS_W);
    memset(p_b + y * stride, fill.b, CANVAS_W);
    memset(p_a + y * stride, fill.a, CANVAS_W);
  }

  return img;
}


std::shared_ptr<HeifPixelImage> make_layer(const LayerSpec& spec)
{
  auto img = std::make_shared<HeifPixelImage>();
  img->create(spec.w, spec.h, heif_colorspace_RGB, heif_chroma_444);

  for (heif_channel ch : {heif_channel_R, heif_channel_G, heif_channel_B}) {
    REQUIRE(img->add_channel(ch, spec.w, spec.h, 8, heif_get_global_security_limits()).error_code == heif_error_Ok);
  }
  if (spec.has_alpha) {
    REQUIRE(img->add_channel(heif_channel_Alpha, spec.w, spec.h, 8,
                             heif_get_global_security_limits()).error_code == heif_error_Ok);
  }

  size_t stride = 0;
  uint8_t* planes[4];
  planes[0] = img->get_channel_memory(heif_channel_R, &stride);
  planes[1] = img->get_channel_memory(heif_channel_G, &stride);
  planes[2] = img->get_channel_memory(heif_channel_B, &stride);
  planes[3] = spec.has_alpha ? img->get_channel_memory(heif_channel_Alpha, &stride) : nullptr;

  for (uint32_t y = 0; y < spec.h; y++) {
    for (uint32_t x = 0; x < spec.w; x++) {
      const Rgba& v = spec.px[y * spec.w + x];
      planes[0][y * stride + x] = v.r;
      planes[1][y * stride + x] = v.g;
      planes[2][y * stride + x] = v.b;
      if (planes[3]) {
        planes[3][y * stride + x] = v.a;
      }
    }
  }

  return img;
}


uint8_t read_plane(const std::shared_ptr<HeifPixelImage>& img, heif_channel ch, uint32_t x, uint32_t y)
{
  size_t stride = 0;
  const uint8_t* p = img->get_channel_memory(ch, &stride);
  return p[y * stride + x];
}

Rgba read_rgba(const std::shared_ptr<HeifPixelImage>& img, uint32_t x, uint32_t y)
{
  return {read_plane(img, heif_channel_R, x, y),
          read_plane(img, heif_channel_G, x, y),
          read_plane(img, heif_channel_B, x, y),
          read_plane(img, heif_channel_Alpha, x, y)};
}


// Independent reference composition on plain vectors, one R,G,B,A vector each.
struct ReferenceCanvas
{
  std::vector<uint8_t> r, g, b, a;

  explicit ReferenceCanvas(Rgba fill)
      : r(CANVAS_W * CANVAS_H, fill.r), g(CANVAS_W * CANVAS_H, fill.g),
        b(CANVAS_W * CANVAS_H, fill.b), a(CANVAS_W * CANVAS_H, fill.a) {}

  void overlay(const LayerSpec& spec)
  {
    for (uint32_t y = 0; y < spec.h; y++) {
      for (uint32_t x = 0; x < spec.w; x++) {
        const int64_t cx = static_cast<int64_t>(spec.dx) + x;
        const int64_t cy = static_cast<int64_t>(spec.dy) + y;
        if (cx < 0 || cy < 0 || cx >= static_cast<int64_t>(CANVAS_W) || cy >= static_cast<int64_t>(CANVAS_H)) {
          continue;
        }

        const Rgba& in = spec.px[y * spec.w + x];
        const uint32_t As = spec.has_alpha ? in.a : 255;
        const size_t idx = static_cast<size_t>(cy) * CANVAS_W + static_cast<size_t>(cx);
        const uint32_t Ab = a[idx];
        const uint32_t rest = 255 - As;
        const uint32_t Aout = As + Ab * rest / 255;

        a[idx] = static_cast<uint8_t>(Aout);
        if (Aout == 0) {
          r[idx] = g[idx] = b[idx] = 0;
        }
        else {
          const uint32_t canvas_weight = Ab * rest / 255;
          r[idx] = static_cast<uint8_t>((in.r * As + r[idx] * canvas_weight) / Aout);
          g[idx] = static_cast<uint8_t>((in.g * As + g[idx] * canvas_weight) / Aout);
          b[idx] = static_cast<uint8_t>((in.b * As + b[idx] * canvas_weight) / Aout);
        }
      }
    }
  }

  Rgba at(uint32_t x, uint32_t y) const
  {
    const size_t idx = y * CANVAS_W + x;
    return {r[idx], g[idx], b[idx], a[idx]};
  }
};


void require_matches_reference(const std::shared_ptr<HeifPixelImage>& canvas,
                               const ReferenceCanvas& reference)
{
  for (uint32_t y = 0; y < CANVAS_H; y++) {
    for (uint32_t x = 0; x < CANVAS_W; x++) {
      const Rgba actual = read_rgba(canvas, x, y);
      const Rgba expected = reference.at(x, y);
      INFO("pixel (" << x << "," << y << ") actual=(" << actual.r << "," << actual.g << ","
                     << actual.b << "," << static_cast<int>(actual.a) << ") expected=("
                     << expected.r << "," << expected.g << "," << expected.b << ","
                     << static_cast<int>(expected.a) << ")");
      REQUIRE(actual.r == expected.r);
      REQUIRE(actual.g == expected.g);
      REQUIRE(actual.b == expected.b);
      REQUIRE(actual.a == expected.a);
    }
  }
}


// Composite the layers in the given order both through HeifPixelImage::overlay()
// and through the reference, and require identical results.
std::shared_ptr<HeifPixelImage>
composite_and_check(Rgba background, const std::vector<LayerSpec>& layers, const std::vector<size_t>& order)
{
  auto canvas = make_canvas(background);
  ReferenceCanvas reference(background);

  for (size_t idx : order) {
    auto layer = make_layer(layers[idx]);
    REQUIRE(canvas->overlay(layer, layers[idx].dx, layers[idx].dy).error_code == heif_error_Ok);
    reference.overlay(layers[idx]);
  }

  require_matches_reference(canvas, reference);
  return canvas;
}

} // namespace


// HF3 sample: a 4x3 opaque base covering the canvas, a semi-transparent 1x1
// badge at (1,0) and an opaque 1x1 corner mark at (2,1). The badge and corner
// do not overlap, so swapping their declaration order must leave every pixel
// unchanged; each layer's own effective pixels, including its resulting alpha,
// have to survive the round trip.
TEST_CASE("overlay on an alpha canvas: HF3 layers keep order, offsets and alpha")
{
  const Rgba transparent{0, 0, 0, 0};

  LayerSpec base;
  base.w = CANVAS_W;
  base.h = CANVAS_H;
  base.dx = 0;
  base.dy = 0;
  base.has_alpha = false;
  base.px.resize(CANVAS_W * CANVAS_H, Rgba{10, 20, 30, 255});

  LayerSpec badge;
  badge.w = 1;
  badge.h = 1;
  badge.dx = 1;
  badge.dy = 0;
  badge.has_alpha = true;
  badge.px = {Rgba{200, 100, 50, 128}};

  LayerSpec corner;
  corner.w = 1;
  corner.h = 1;
  corner.dx = 2;
  corner.dy = 1;
  corner.has_alpha = true;
  corner.px = {Rgba{0, 0, 200, 255}};

  const std::vector<LayerSpec> layers = {base, badge, corner};

  auto order_abc = composite_and_check(transparent, layers, {0, 1, 2});
  auto order_acb = composite_and_check(transparent, layers, {0, 2, 1});

  // The two orders produce identical pixels because no layer overlaps another.
  for (uint32_t y = 0; y < CANVAS_H; y++) {
    for (uint32_t x = 0; x < CANVAS_W; x++) {
      REQUIRE(read_rgba(order_abc, x, y).r == read_rgba(order_acb, x, y).r);
      REQUIRE(read_rgba(order_abc, x, y).g == read_rgba(order_acb, x, y).g);
      REQUIRE(read_rgba(order_abc, x, y).b == read_rgba(order_acb, x, y).b);
      REQUIRE(read_rgba(order_abc, x, y).a == read_rgba(order_acb, x, y).a);
    }
  }

  // Background alpha: the base covers the whole canvas, so every pixel is at
  // least as opaque as the base.
  for (uint32_t y = 0; y < CANVAS_H; y++) {
    for (uint32_t x = 0; x < CANVAS_W; x++) {
      REQUIRE(read_rgba(order_abc, x, y).a == 255);
    }
  }

  // The badge keeps its own alpha and color on top of the opaque base.
  const Rgba badge_pixel = read_rgba(order_abc, 1, 0);
  CHECK(badge_pixel.a == 255);  // opaque base underneath -> result is opaque
  CHECK(badge_pixel.r == static_cast<uint8_t>((200u * 128u + 10u * 127u) / 255u));

  // The opaque corner replaces the base color at exactly its one pixel.
  const Rgba corner_pixel = read_rgba(order_abc, 2, 1);
  CHECK(corner_pixel.r == 0);
  CHECK(corner_pixel.g == 0);
  CHECK(corner_pixel.b == 200);
  CHECK(corner_pixel.a == 255);
}


// When two semi-transparent layers overlap, the declaration order decides the
// visible color at the overlap: swapping the declarations swaps the overlap
// color, while pixels covered by only one of the layers stay unchanged.
TEST_CASE("overlay on an alpha canvas: swapping declarations swaps the overlap color only")
{
  const Rgba transparent{0, 0, 0, 0};

  // 2x1 badge: exclusive pixel at (0,0), overlap pixel at (1,0)
  LayerSpec lower;
  lower.w = 2;
  lower.h = 1;
  lower.dx = 0;
  lower.dy = 0;
  lower.has_alpha = true;
  lower.px = {Rgba{200, 0, 0, 128}, Rgba{200, 0, 0, 128}};

  // 1x1 corner placed exactly on the right pixel of the badge.
  LayerSpec upper;
  upper.w = 1;
  upper.h = 1;
  upper.dx = 1;
  upper.dy = 0;
  upper.has_alpha = true;
  upper.px = {Rgba{0, 200, 0, 128}};

  const std::vector<LayerSpec> layers = {lower, upper};

  auto lower_first = composite_and_check(transparent, layers, {0, 1});
  auto upper_first = composite_and_check(transparent, layers, {1, 0});

  // The exclusive badge pixel is independent of the declaration order.
  const Rgba exclusive_a = read_rgba(lower_first, 0, 0);
  const Rgba exclusive_b = read_rgba(upper_first, 0, 0);
  CHECK(exclusive_a.r == exclusive_b.r);
  CHECK(exclusive_a.g == exclusive_b.g);
  CHECK(exclusive_a.b == exclusive_b.b);
  CHECK(exclusive_a.a == exclusive_b.a);
  CHECK(exclusive_a.a == 128);  // nothing underneath on the transparent canvas

  // Every other canvas pixel is untouched background.
  for (uint32_t y = 0; y < CANVAS_H; y++) {
    for (uint32_t x = 0; x < CANVAS_W; x++) {
      if ((x == 0 || x == 1) && y == 0) {
        continue;
      }
      CHECK(read_rgba(lower_first, x, y).a == 0);
      CHECK(read_rgba(upper_first, x, y).a == 0);
    }
  }

  // The overlap pixel changes with the declaration order.
  const Rgba overlap_a = read_rgba(lower_first, 1, 0);
  const Rgba overlap_b = read_rgba(upper_first, 1, 0);
  CHECK(overlap_a.r != overlap_b.r);
  CHECK(overlap_a.g != overlap_b.g);
  // The accumulated alpha is order-independent for source-over.
  CHECK(overlap_a.a == overlap_b.a);
}


// A fully transparent sample of an upper layer must not write over the canvas
// pixel beneath it, while an opaque sample right next to it replaces that
// neighbor. This is the "uncovered pixels are not written early" guarantee.
TEST_CASE("overlay on an alpha canvas: transparent samples leave the canvas underneath untouched")
{
  const Rgba opaque_red{200, 0, 0, 255};
  auto canvas = make_canvas(opaque_red);
  ReferenceCanvas reference(opaque_red);

  LayerSpec layer;
  layer.w = 2;
  layer.h = 1;
  layer.dx = 0;
  layer.dy = 0;
  layer.has_alpha = true;
  layer.px = {Rgba{0, 200, 0, 0},   // fully transparent
              Rgba{0, 200, 0, 255}}; // fully opaque

  auto img = make_layer(layer);
  REQUIRE(canvas->overlay(img, layer.dx, layer.dy).error_code == heif_error_Ok);
  reference.overlay(layer);
  require_matches_reference(canvas, reference);

  const Rgba untouched = read_rgba(canvas, 0, 0);
  CHECK(untouched.r == 200);
  CHECK(untouched.g == 0);
  CHECK(untouched.a == 255);

  const Rgba replaced = read_rgba(canvas, 1, 0);
  CHECK(replaced.r == 0);
  CHECK(replaced.g == 200);
  CHECK(replaced.a == 255);
}


// An overlay placed completely outside the canvas modifies neither the colors
// nor the alpha plane.
TEST_CASE("overlay on an alpha canvas: an outside layer changes nothing, alpha included")
{
  const Rgba fill{40, 50, 60, 128};
  auto canvas = make_canvas(fill);

  LayerSpec layer;
  layer.w = 1;
  layer.h = 1;
  layer.dx = static_cast<int32_t>(CANVAS_W);
  layer.dy = 0;
  layer.has_alpha = true;
  layer.px = {Rgba{255, 255, 255, 255}};

  auto img = make_layer(layer);
  REQUIRE(canvas->overlay(img, layer.dx, layer.dy).error_code == heif_error_Ok);

  for (uint32_t y = 0; y < CANVAS_H; y++) {
    for (uint32_t x = 0; x < CANVAS_W; x++) {
      const Rgba p = read_rgba(canvas, x, y);
      CHECK(p.r == fill.r);
      CHECK(p.g == fill.g);
      CHECK(p.b == fill.b);
      CHECK(p.a == fill.a);
    }
  }
}


// The same HF3 structure with 16-bit samples, exercising the wide-sample path
// of the source-over composition.
TEST_CASE("overlay on an alpha canvas: 16-bit HF3 layers match the reference composition")
{
  const int bits = 16;
  const uint16_t amax = 0xFFFF;

  auto canvas16 = std::make_shared<HeifPixelImage>();
  canvas16->create(CANVAS_W, CANVAS_H, heif_colorspace_RGB, heif_chroma_444);
  for (heif_channel ch : {heif_channel_R, heif_channel_G, heif_channel_B, heif_channel_Alpha}) {
    REQUIRE(canvas16->add_channel(ch, CANVAS_W, CANVAS_H, bits,
                                  heif_get_global_security_limits()).error_code == heif_error_Ok);
  }
  // transparent black background
  canvas16->fill_channel(heif_channel_R, 0);
  canvas16->fill_channel(heif_channel_G, 0);
  canvas16->fill_channel(heif_channel_B, 0);
  canvas16->fill_channel(heif_channel_Alpha, 0);

  auto make_16 = [&](uint32_t w, uint32_t h, uint16_t r, uint16_t g, uint16_t b,
                     bool with_alpha, uint16_t alpha) {
    auto img = std::make_shared<HeifPixelImage>();
    img->create(w, h, heif_colorspace_RGB, heif_chroma_444);
    for (heif_channel ch : {heif_channel_R, heif_channel_G, heif_channel_B}) {
      REQUIRE(img->add_channel(ch, w, h, bits, heif_get_global_security_limits()).error_code == heif_error_Ok);
    }
    if (with_alpha) {
      REQUIRE(img->add_channel(heif_channel_Alpha, w, h, bits,
                               heif_get_global_security_limits()).error_code == heif_error_Ok);
    }
    img->fill_channel(heif_channel_R, r);
    img->fill_channel(heif_channel_G, g);
    img->fill_channel(heif_channel_B, b);
    if (with_alpha) {
      img->fill_channel(heif_channel_Alpha, alpha);
    }
    return img;
  };

  // opaque base over the whole canvas
  auto base = make_16(CANVAS_W, CANVAS_H, 1000, 2000, 3000, false, amax);
  REQUIRE(canvas16->overlay(base, 0, 0).error_code == heif_error_Ok);

  // semi-transparent badge at (1,0)
  const uint16_t badge_alpha = amax / 2;
  auto badge = make_16(1, 1, 40000, 20000, 10000, true, badge_alpha);
  REQUIRE(canvas16->overlay(badge, 1, 0).error_code == heif_error_Ok);

  // Reference for the badge pixel.
  const uint32_t As = badge_alpha;
  const uint32_t Ab = amax;  // base made the pixel opaque
  const uint32_t rest = amax - As;
  const uint32_t Aout = As + Ab * rest / amax;
  const uint32_t weight = Ab * rest / amax;
  const uint16_t expected_r = static_cast<uint16_t>((40000u * As + 1000u * weight) / Aout);
  const uint16_t expected_a = static_cast<uint16_t>(Aout);

  size_t stride = 0;
  const uint16_t* pr = reinterpret_cast<const uint16_t*>(canvas16->get_channel_memory(heif_channel_R, &stride));
  const uint16_t* pa = reinterpret_cast<const uint16_t*>(canvas16->get_channel_memory(heif_channel_Alpha, &stride));
  CHECK(pr[1] == expected_r);
  CHECK(pa[1] == expected_a);

  // A base-only pixel keeps the base values and full opacity.
  CHECK(pr[0] == 1000);
  CHECK(pa[0] == amax);
}
