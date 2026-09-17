// t4_prodaddr_check.cpp — 生产寻址臂的**编译期一致性门**（零板端动作）
//   同时 include 本臂的参数单一来源 + 生产布局头 t4_mmq_canon.h，逐 (row,u) 比对 canonical 落位表，
//   并核对记录字节数/板尺寸/chunk 尺寸。任何一侧改动 ⇒ 这里编译失败 ⇒ 不许上板。
#include <cstdio>
#include "t4_prodaddr_layout.h"
#include "t4_mmq_canon.h"

constexpr bool layout_identical() {
  for (int row = 0; row < t4prod::ROWS; ++row)
    for (int u = 0; u < 8; ++u)
      if (t4prod::b_smem_off(row, u) != t4mmq::b_smem_off(row, u)) return false;
  return true;
}
static_assert(layout_identical(), "b_smem_off 表与生产 t4_mmq_canon.h 不一致 ⇒ 停手");
static_assert(t4prod::REC == t4mmq::B_REC, "记录字节数与生产不一致");
static_assert(t4prod::B_SUB == t4mmq::B_SUB, "canonical 板尺寸与生产不一致");
static_assert(t4prod::NSUB == t4mmq::NSUB, "每 chunk 的板数与生产不一致");
static_assert(t4prod::CPR * 256 == 5120, "CPR 必须对应 K=5120 主形状");
static_assert(t4prod::ROW_STRIDE == t4prod::CPR * t4mmq::B_REC * 4, "行步长必须是 CPR×4×36（无补零）");

int main() {
  std::printf("OK LAYOUT_IDENTICAL CPR=%d ROW_STRIDE=%d TILE_BYTES=%d REC=%d\n",
              t4prod::CPR, t4prod::ROW_STRIDE, t4prod::TILE_BYTES, t4prod::REC);
  return 0;
}
