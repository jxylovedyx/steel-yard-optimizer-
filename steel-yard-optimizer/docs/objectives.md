# objectives
双目标：
1) 切割作业总完工时间最小（makespan）
2) 在库钢板后续翻板/倒垛次数最少（relocations）

落地策略建议：
- 词典序：先最小 makespan，再在近优解中最小 relocations
- 或加权：score = w1*makespan + w2*relocations（配置化）
