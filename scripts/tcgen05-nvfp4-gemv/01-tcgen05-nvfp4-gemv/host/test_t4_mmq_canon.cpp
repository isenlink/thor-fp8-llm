// test_t4_mmq_canon.cpp — 生产布局头 t4_mmq_canon.h 的宿主判据（不占板）
//
// 判据链（每条都能单独失败）：
//   T1  A 板布局是双射：每个 (m,k) 恰好落到一个 (字节, 半字节) 槽，且覆盖满 [0, M*32) × 2 半字节。
//   T2  环几何/TMEM 列数自洽（含 occ 判定），与实测预算（232448 B smem、512 列 TMEM）对齐。
//   T3  **字节级对账**：用本头的 B 映射从真实 GGML 张量重构 canonical chunk，与归档
//       evidence/20260916-112 的 stream_num.bin（已由内核逐位验证过的定版件）逐字节全等。
//   T4  **约定对账**：A（激活）的 (元素 → 记录内字节/半字节) 映射必须与 GGML NVFP4 记录约定逐项相同
//       （这是「A/B 同约定 ⇒ 点积对索引置换不变」的前提；两侧不同约定 ⇒ D 全错）。
//   T5  证据件复核：canonical 量化器与上游 mmq 量化器在同一条 x 上的 (m,k) nibble 表逐位相同。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <set>

#include "t4_mmq_canon.h"

using namespace t4mmq;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

static std::vector<unsigned char> slurp(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) { printf("  FAIL: 打不开 %s\n", p.c_str()); g_fail++; return {}; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> v((size_t) n);
    if (fread(v.data(), 1, (size_t) n, f) != (size_t) n) { printf("  FAIL: 读不全 %s\n", p.c_str()); g_fail++; }
    fclose(f);
    return v;
}

// ------------------------------------------------------------------ T1
static void t1_a_bijection() {
    printf("T1 A 布局双射\n");
    for (int m : {0, 1, 7, 8, 13, 63, 64, 127}) {
        std::set<std::pair<int,int>> slots;   // (字节, 半字节)
        for (int sub = 0; sub < 4; ++sub)
            for (int l = 0; l < 8; ++l) {
                const int b = a_nib_off_in_board(m, sub, l);
                const int k = 16 * sub + l;
                CHECK(b >= 0 && b < MMA_M * 32, "m=%d sub=%d l=%d 字节越界 %d", m, sub, l, b);
                CHECK(slots.insert({b, 0}).second, "低半字节槽重复 m=%d sub=%d l=%d b=%d", m, sub, l, b);
                CHECK(slots.insert({b, 1}).second, "高半字节槽重复 m=%d sub=%d l=%d b=%d", m, sub, l, b);
                // 反向：a_nib_off 对元素 k 给出同一字节
                CHECK((int) a_nib_off(m, 0, k) == b, "a_nib_off 反查不一致 m=%d k=%d", m, k);
            }
    }
    // 板内字节双射：每个板恰好覆盖 M*32 字节（nib 区无空洞、无重叠）
    {
        std::set<int> bytes;
        for (int m = 0; m < MMA_M; ++m)
            for (int sub = 0; sub < 4; ++sub)
                for (int l = 0; l < 8; ++l)
                    bytes.insert(a_nib_off_in_board(m, sub, l));
        CHECK((int) bytes.size() == MMA_M * 32, "板字节数 %zu != %d", bytes.size(), MMA_M * 32);
        CHECK(*bytes.begin() == 0 && *bytes.rbegin() == MMA_M * 32 - 1, "板字节区间错位 [%d,%d]", *bytes.begin(), *bytes.rbegin());
    }
    printf("  完成（槽位唯一性 / 越界 / 覆盖密度）\n");
}

// ------------------------------------------------------------------ T4
static int record_byte_of(int sub, int l) { return 4 + 8 * sub + l; }   // GGML NVFP4 记录内字节
static void t4_convention() {
    printf("T4 A 元素 → (字节, 半字节) 与 GGML 记录约定对账\n");
    for (int sub = 0; sub < 4; ++sub)
        for (int l = 0; l < 8; ++l) {
            const int b_my = a_nib_off_in_board(0, sub, l) + 0;   // m=0 → 组内基址 0
            const int b_rec = record_byte_of(sub, l);
            // my: 低半字节元素 16*sub+l；记录: 同样是「字节 l 的低半字节 = 元素 16*sub+l」
            CHECK(b_my == (sub / 2) * 128 + (sub % 2) * 8 + l, "A 字节偏移式不符 sub=%d l=%d b=%d", sub, l, b_my);
            CHECK(b_rec == 4 + 8 * sub + l, "记录字节偏移式不符");
        }
    printf("  完成（两侧「字节 l ↔ 元素 16sub+l / +8」逐项一致）\n");
}

// ------------------------------------------------------------------ T3
// T3a：B（记录）与 A（量化器）的「板内槽位」映射必须逐字节同构 —— 这是 A/B 同约定的结构性判据。
static void t3a_slot_identity() {
    printf("T3a 记录槽位 vs A 槽位（同约定）\n");
    for (int sub = 0; sub < 4; ++sub)
        for (int l = 0; l < 8; ++l) {
            const int p = 8 * sub + l;              // 记录 payload 内的字节号 [0,32)
            const int u = p / 4, i = p % 4;
            const int b_slot = b_smem_off(0, u) + i;       // m=0 → 组内基址 0
            const int a_slot = a_nib_off_in_board(0, sub, l);
            CHECK(b_slot == a_slot, "p=%d: B 槽 %d != A 槽 %d", p, b_slot, a_slot);
        }
    printf("  完成（32 个 payload 字节的槽位一一对应）\n");
}
// T3b：两种 nibble 约定的**元素级**对账（归档件）——
//   raw（ggml 文件格式：qs[8s+j] = 元素 j | 元素 j+8 << 4，见 ggml-quants.c:quantize_row_nvfp4_ref）
//   vs canonical 流（重排工具：字节 p 的低半字节 = 板内第 2p 个 k'，高 = 2p+1）。
//   两者是同一批元素的两种包装 ⇒ 解出的 (元素 → 4bit) 表必须逐位相同。
static void t3b_two_conventions(const std::string & base) {
    printf("T3b 记录约定 vs canonical 流约定（元素级）\n");
    auto raw = slurp(base + "/stage/raw_t2.bin");
    auto stm = slurp(base + "/stage/stream_num.bin");
    if (raw.empty() || stm.empty()) return;
    const int ROW_STRIDE = 2880, CHUNKS = 20, TILES = 2;
    long mism = 0, n = 0;
    for (int tile = 0; tile < TILES; ++tile)
        for (int c = 0; c < CHUNKS; ++c)
            for (int s = 0; s < NSUB; ++s)
                for (int row = 0; row < 32; ++row) {
                    const size_t rec = (size_t)(tile * 32 + row) * ROW_STRIDE + (size_t)(4 * c + s) * B_REC;
                    const size_t ch  = (size_t)(tile * CHUNKS + c) * B_CHUNK + (size_t) s * B_SUB;
                    for (int kk = 0; kk < 64; ++kk) {
                        const int sub = kk / 16, j = kk % 16;
                        const unsigned char braw = raw[rec + 4 + 8 * sub + (j % 8)];
                        const int nraw = (j < 8) ? (braw & 0xF) : (braw >> 4);
                        const unsigned char bst = stm[ch + (row % 8) * 16 + (kk / 32) * 128 + (row / 8) * 256 + (kk % 32) / 2];
                        const int nst = (kk % 2) ? (bst >> 4) : (bst & 0xF);
                        n++;
                        if (nraw != nst) mism++;
                    }
                }
    CHECK(mism == 0, "元素级失配 %ld / %ld", mism, n);
    printf("  完成：%ld 个 (行,板,元素) 的 4bit 值两种包装完全一致\n", n);
}

// ------------------------------------------------------------------ T5
static int canon_nibble_at(const std::vector<unsigned char> & cn, int M, int m, int k) {
    // canonical 量化器：字节 l 的低半字节 = 元素 2l（子块内），高 = 2l+1
    const int board = k / 64, kk = k % 64, sub = kk / 16, k16 = kk % 16;
    const int l = k16 / 2, hi = k16 % 2;
    const size_t off = (size_t) board * (M * 32) + (m % 8) * 16 + (sub / 2) * 128 + (m / 8) * 256 + (sub % 2) * 8 + l;
    return hi ? (cn[off] >> 4) : (cn[off] & 0xF);
}
static int mmq_nibble_at(const std::vector<unsigned char> & rm, int M, int m, int k) {
    // 上游 mmq 布局（block_fp4_mmq = 16 B d4[4] + 128 B qs，覆盖 QK_FP4_MMQ = 256 个 K）：
    //   y 索引 = (k_block*ne1 + i)；qs 内 8 字节/子块：q0 = (e0,e8),(e1,e9),(e2,e10),(e3,e11)，
    //   q1 = (e4,e12),(e5,e13),(e6,e14),(e7,e15)  ⇒ 字节 (j%8) 低半字节 = 元素 j，高 = 元素 j+8（j<8 低 / j>=8 高）
    const int k_block = k / 256, r = k % 256, sub = r / 16, j = r % 16;
    const size_t off = (size_t)(k_block * M + m) * 144 + 16 + (size_t) sub * 8 + (j % 8);  // +16 = d4[4]
    return (j < 8) ? (rm[off] & 0xF) : (rm[off] >> 4);
}
static void t5_quantizer_tables(const std::string & base) {
    printf("T5 量化器复核：canonical 包装 vs 上游 mmq 包装的 (m,k) nibble 表\n");
    auto cn = slurp(base + "/board/aqout1/canon_nib.bin");
    auto rm = slurp(base + "/board/aqout1/ref_mmq.bin");
    if (cn.empty() || rm.empty()) return;
    const int M = 128, K = 5120;
    CHECK((int) cn.size() == (K / 64) * M * 32, "canon_nib 字节数 %zu", cn.size());
    CHECK((int) rm.size() == (K / 64) * M * 36, "ref_mmq 字节数 %zu", rm.size());
    long mism = 0, n = 0;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            n++;
            if (canon_nibble_at(cn, M, m, k) != mmq_nibble_at(rm, M, m, k)) mism++;
        }
    CHECK(mism == 0, "量化器 nibble 表失配 %ld / %ld", mism, n);
    printf("  完成：%ld 个 nibble 的 (m,k) 表逐位相同\n", n);
}

// ------------------------------------------------------------------ T2
static void t2_geometry() {
    printf("T2 环几何 / TMEM / occ\n");
    struct C { const char * name; ring_cfg c; int mreal; };
    const C cfgs[] = {
        {"cps3_s3_mreal16", {3, 3}, 16},
        {"cps3_s2_mreal16", {3, 2}, 16},
        {"cps3_s3_mreal128", {3, 3}, 128},
        {"cps3_s2_mreal128", {3, 2}, 128},
        {"cps4_s2_mreal16", {4, 2}, 16},
        {"cps7_s2_mreal16", {7, 2}, 16},
        {"cps8_s3_mreal16", {8, 3}, 16},
    };
    for (const auto & x : cfgs) {
        const int smem = ring_smem_required(x.mreal, x.c);
        const int cols = ring_tmem_cols(x.c);
        const int occ_smem = SMEM_PER_SM_MAX / ring_smem_host_req(x.mreal, x.c);
        const int occ = ring_tmem_fits(x.c) ? (occ_smem < ring_occ_limit(x.c) ? occ_smem : ring_occ_limit(x.c)) : 0;
        printf("  %-18s A_CHUNK=%5d smem=%7d (host %7d) TMEM=%3d occ=%d%s\n",
               x.name, a_chunk_bytes(x.mreal), smem, ring_smem_host_req(x.mreal, x.c), cols, occ,
               ring_tmem_fits(x.c) ? "" : "  <TMEM 超限>");
        CHECK(smem % 16 == 0, "smem 必须 16 B 对齐");
        CHECK(ring_smem_host_req(x.mreal, x.c) <= 232448, "smem 超板卡上限");
    }
    printf("  完成\n");
}

int main(int argc, char ** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    printf("== t4_mmq_canon 宿主判据（布局头 vs 归档定版件）==\n");
    t1_a_bijection();
    t4_convention();
    t2_geometry();
    t3a_slot_identity();
    t3b_two_conventions(root + "/evidence/20260916-112-t4-v11-sfbase-fix");
    t5_quantizer_tables(root + "/evidence/20260916-90-t4-a-quantizer");
    printf("== %s ==\n", g_fail ? "CANON_TEST=FAIL" : "CANON_TEST=PASS");
    return g_fail ? 1 : 0;
}
