# 6-Node DAG Experiment - 修复总结

## 修复日期
2026-01-27

## 修复内容

### 1. dag_dataset.py

#### Bug修复
- **第120行**: 修复x5定义错误
  - 错误: `x5 = (x4 + x5) % self.vocab_size//2` (x5引用了自己)
  - 正确: `x5 = (x3 + x4) % 16` (汇聚x3和x4)

#### 测试修复
- **第173行**: 修复测试断言
  - 错误: `assert x5 == (x4 + x5) % self.vocab_size//2`
  - 正确: `assert x5 == (x3 + x4) % 16`

- **第187行**: 修复打印输出
  - 错误: `x4={x4:2d}, x5={x5:2d} → x5={x5:2d} ((x4+x5)%self.vocab_size//2)`
  - 正确: `x3={x3:2d}, x4={x4:2d} → x5={x5:2d} ((x3+x4)%16)`

- **第195-208行**: 修复信息内容分析示例
  - 更新为基于x0 (root)的完整信息传播示例

- **第268, 276行**: 修复硬编码的模数
  - 小vocab: `x3 == (x1 + x2) % (dataset_small.vocab_size // 4)`
  - 大vocab: `x3 == (x1 + x2) % (dataset_large.vocab_size // 4)`

### 2. train_dag.py

#### 参数更新
- **第52行**: 更新默认参数
  - 错误: `def compute_dag_selection_metrics(actions_list, seq_length=4)`
  - 正确: `def compute_dag_selection_metrics(actions_list, seq_length=6)`

#### 指标计算扩展
- **第98-111行**: 添加第4、5步和最后一步的指标跟踪
  - 新增 `fourth_step/p_x*` 指标
  - 新增 `fifth_step/p_x*` 指标
  - 将 `last_step` 从t=3改为t=5
  - 更新拓扑正确性检查: `(first_actions == 0) & (last_actions == 5)`

#### 注释更新
- **第155, 217行**: 更新序列长度注释从T=4到T=6
- **第268行**: 更新action注释包含x4和x5

#### 输出更新
- **第481-486行**: 更新main函数的打印输出
  - 描述完整的6节点DAG结构
- **第685-711行**: 更新最终结果输出
  - 改为检查`last_step/p_x5`而非`last_step/p_x3`
  - 简化成功标准检查

## 6节点DAG结构

```
       x0 (Root, depth 0)
      /    \
     /      \
   x1        x2     (Branches, depth 1)
     \      /
      \    /
       x3 (depth 2)
        |
       x4 (depth 2)
        |
       x5 (Sink, depth 3)
```

### 数学定义
- x0: 随机采样 ∈ [0, vocab_size)
- x1 = x0 // 2
- x2 = (x0 + 1) // 2
- x3 = (x1 + x2) % (vocab_size // 4)
- x4 = (x3 + 1) // 2
- x5 = (x3 + x4) % 16

### 预期Agent行为
1. **第1步**: 选择x0 (root) - 概率应 >90%
2. **第2-3步**: 选择x1和x2 (任意顺序)
3. **第4步**: 选择x3 (汇聚点)
4. **第5步**: 选择x4
5. **第6步**: 选择x5 (sink) - 概率应 >90%

## 验证

所有测试通过:
```bash
python dag_exp/dag_dataset.py
# ✅ All tests passed!
```

## 下一步

现在可以开始训练:
```bash
/home/admin/anaconda3/envs/order_lando/bin/python dag_exp/train_dag.py
```
