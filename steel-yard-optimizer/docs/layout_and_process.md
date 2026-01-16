# layout_and_process

## 布局对象
- yard: 库区/垛位
- buffers: 缓存位/辊道/预处理工位
- cutting_lines: 多条切割线

## 数据流程
raw -> interim -> instances -> plan/metrics -> results/<run_id>/

## 运行流程（离线/滚动）
1) 导入 raw
2) 清洗映射到 interim
3) 生成 instance（标准输入）
4) 求解生成 plan + metrics
5) 导出下发（CSV/JSON），并保留 config_snapshot 与日志
