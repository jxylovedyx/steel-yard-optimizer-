# data_dictionary
以 schemas/*.schema.json 为准，这里补充字段解释与缺失值规则。

## plate
- plate_id: 唯一标识
- spec: thickness/width/length/weight
- location: area/stack_id/level
- blocked_by: 阻塞它的上层板（可选；也可由层序推导）
- eligible_lines: 可供哪些切割线（可选）
- status: in_yard/reserved/to_cut/cut

## demand
- cutting_line_id
- items: plate_id 列表
- priority / due_time（可选）

## equipment
- cranes: max_load_ton, pick_seconds, place_seconds
- travel_model: seconds_per_meter（或更复杂模型）
