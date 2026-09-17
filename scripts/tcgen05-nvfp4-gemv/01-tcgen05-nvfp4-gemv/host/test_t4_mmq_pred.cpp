// test_t4_mmq_pred.cpp — `t4_mmq_pred.h` 的真值表单测（宿主；无 CUDA、无 ggml、不占板）。
//
// 口径：正例 = **已实测验证过的真实形状**（results/20260916-108/109/112）；负控 = 任何一条不满足就必须拒绝。
// 生产侧耦合只在 t4_mmq.cu 的 static_assert（常量与 ggml/CUDA 真值绑定），本测只验证判定逻辑本身。
#include "t4_mmq_pred.h"

#include <cstdio>

static int g_total = 0, g_fail = 0;

static void ck(const char * name, const t4_mmq_pred_in & s, const bool want) {
    const bool got = t4_mmq_pred_ok(s);
    const bool ok  = (got == want);
    ++g_total;
    if (!ok) { ++g_fail; }
    std::printf("  %-52s 期望=%-5s 实得=%-5s %s\n", name, want ? "true" : "false",
                got ? "true" : "false", ok ? "OK" : "**FAIL**");
}

// 真实形状基线：gate/up（K=5120, N=14336, M=MREAL=13, 行步长 80 块 = 2880 B）
static t4_mmq_pred_in base() {
    t4_mmq_pred_in s = {};
    s.cc = 1010; s.arch = 1010; s.type_x = 40;
    s.ids_branch = false;
    s.ncols_x = 5120;
    s.nrows_x = 14336;
    s.ncols_dst = 13;
    s.stride_row_x = 80;          // 块数：2880 B / 36 B
    s.nchannels_y = 1;
    s.nsamples_y = 1;
    s.nchannels_x = 1;
    s.nsamples_x = 1;
    s.type_y = 0;                 // F32 激活
    s.act32_ok = true;
    s.ptr4_ok = true;
    return s;
}

int main() {
    std::printf("== 正例（必须接受）==\n");
    ck("gate/up 真实形状 M=13（stride 80 块 = 2880 B）", base(), true);
    { auto s = base(); s.ncols_dst = 128; ck("M=128 全宽（原型验证过）", s, true); }
    { auto s = base(); s.ncols_dst = 1;   ck("M=1 单 token（补零等价）", s, true); }
    { auto s = base(); s.ncols_x = 17408; s.nrows_x = 5120; s.stride_row_x = 272;
      ck("ffn_down 真实形状 K=17408 / N=5120", s, true); }
    { auto s = base(); s.ncols_x = 256; s.nrows_x = 32; s.stride_row_x = 4;
      ck("K=256 下界 / N=32 下界 / stride=4 块", s, true); }
    { auto s = base(); s.stride_row_x = 81; ck("行步长补到 81 块（对齐无约束）", s, true); }

    std::printf("== 负控（必须拒绝）==\n");
    { auto s = base(); s.cc = 1200;       ck("cc=1200（Blackwell sm_120，非 Thor）", s, false); }
    { auto s = base(); s.arch = 1200;     ck("arch=1200（编译期未含 sm_101a）", s, false); }
    { auto s = base(); s.type_x = 39;     ck("type=MXFP4", s, false); }
    { auto s = base(); s.type_x = 2;      ck("type=Q4_0", s, false); }
    { auto s = base(); s.type_y = 1;      ck("激活 type=F16（会被当 F32 读）", s, false); }
    { auto s = base(); s.nchannels_x = 2; ck("权重 ne02=2（批权重）", s, false); }
    { auto s = base(); s.ids_branch = true; ck("ids / expert_bounds 分支", s, false); }
    { auto s = base(); s.nchannels_y = 2; ck("ne12=2（通道批）", s, false); }
    { auto s = base(); s.nsamples_y = 2;  ck("ne13=2（多样本）", s, false); }
    { auto s = base(); s.ncols_x = 4992; s.stride_row_x = 78;
      ck("K=4992（非 KCHUNK 整数倍）", s, false); }
    { auto s = base(); s.nrows_x = 14335; ck("N=14335（非 n-tile 整数倍）", s, false); }
    { auto s = base(); s.ncols_dst = 0;   ck("M=0", s, false); }
    { auto s = base(); s.ncols_dst = 129; ck("M=129（超 TMEM lane 数）", s, false); }
    { auto s = base(); s.stride_row_x = 79; ck("stride=79 块（装不下 K/64=80 条记录）", s, false); }
    { auto s = base(); s.act32_ok = false; ck("激活行非 32 B 对齐（A 类事故防线）", s, false); }
    { auto s = base(); s.ptr4_ok = false; ck("基址非 4 B 对齐", s, false); }
    { auto s = base(); s.ncols_x = 0;     ck("K=0", s, false); }
    { auto s = base(); s.nrows_x = 0;     ck("N=0", s, false); }

    std::printf("\n== 合计 %d 例，失败 %d ==\n", g_total, g_fail);
    std::printf("%s\n", g_fail == 0 ? "PRED_TRUTH_TABLE=PASS" : "PRED_TRUTH_TABLE=FAIL");
    return g_fail == 0 ? 0 : 1;
}
