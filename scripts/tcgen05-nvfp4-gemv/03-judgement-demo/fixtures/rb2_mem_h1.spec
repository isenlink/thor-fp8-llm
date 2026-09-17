# R-B2 H-mem 臂 · **成立**判据（与 h0 必须恰好一个 PASS）：生产寻址就是那 1.66×
# H-mem 成立 = 原型（4 B/次、同请求数、同 smem、同握手）只要换成生产寻址就掉到腿的量级
#   成立 ⇒ 257 GB/s 这个目标在真实布局下**不可达** ⇒ R-B 线前提不成立、停线（修法只有"载入时一次性 repack"）
RB1_MAIN_GBPS	MAX	170
RB1_MAIN_DMISMATCH	MAX	0
RB1_MAIN_MBAR_OK	MIN	1
RB1_MAIN_PROD_FAIL	MAX	0
RB1_MAIN_SMEM	EQ	114752
