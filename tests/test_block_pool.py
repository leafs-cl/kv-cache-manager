"""KV Cache 块池测试：allocate / append / free 的语义与不变量。

验收标准
1. 空闲块数 + 已用块数 == num_blocks，任何时刻成立
2. 一个块要么空闲、要么被某个序列占用，不能被两个序列同时占用
3. 返回 False 的调用不允许改变任何状态

"""
import random

import pytest

from kv_block_allocator.kv_cache_block_pool import KVCacheBlockPool


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def assert_invariants(pool: KVCacheBlockPool) -> None:
    """任何操作之后都必须成立的不变量。"""
    all_blocks = [b for blocks in pool.seq_block_table.values() for b in blocks]
    assert len(all_blocks) == len(set(all_blocks)), "同一个块被两个序列同时占用"
    assert pool.avail_block_num() + len(all_blocks) == pool.num_blocks
    assert len(pool.block_alloc_table) == len(all_blocks)
    assert len(pool.block_alloc_table) == len(set(pool.block_alloc_table)), "block_alloc_table 有重复项"
    for seq_id, tokens in pool.seq_token_table.items():
        need = ceil_div(tokens, pool.block_size)
        have = len(pool.seq_block_table[seq_id])
        assert have == need, f"seq {seq_id}: {tokens} tokens 需要 {need} 块，实际 {have} 块"


class TestAllocate:
    def test_takes_min_blocks_with_ceil(self):
        """10 tokens / block_size=4 -> 3 块"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 10) is True
        assert pool.block_table(1) == [0, 1, 2]
        assert pool.avail_block_num() == 5
        assert_invariants(pool)

    def test_exact_multiple(self):
        """12 tokens 恰好 3 块，不多给。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 12) is True
        assert pool.block_table(1) == [0, 1, 2]
        assert_invariants(pool)

    def test_no_block_sharing_between_seqs(self):
        """两个序列的块表互不相交。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 10) is True
        assert pool.allocate(2, 6) is True
        assert pool.block_table(1) == [0, 1, 2]
        assert pool.block_table(2) == [3, 4]
        assert not set(pool.block_table(1)) & set(pool.block_table(2))
        assert_invariants(pool)

    def test_insufficient_is_atomic(self):
        """块不够时返回 False，且不改任何状态。"""
        pool = KVCacheBlockPool(num_blocks=4, block_size=4)
        assert pool.allocate(1, 8) is True
        snapshot = (pool.avail_block_num(), list(pool.block_alloc_table), dict(pool.seq_block_table))
        assert pool.allocate(2, 100) is False
        assert (pool.avail_block_num(), list(pool.block_alloc_table), dict(pool.seq_block_table)) == snapshot
        assert 2 not in pool.seq_block_table

    def test_duplicate_allocate(self):
        """同一序列重复分配返回 False。"""
        pool = KVCacheBlockPool(num_blocks=4, block_size=4)
        assert pool.allocate(1, 4) is True
        assert pool.allocate(1, 4) is False
        assert_invariants(pool)

    def test_full_capacity(self):
        """恰好装满全部块；再分配返回 False。"""
        pool = KVCacheBlockPool(num_blocks=4, block_size=4)
        assert pool.allocate(1, 16) is True
        assert pool.block_table(1) == [0, 1, 2, 3]
        assert pool.avail_block_num() == 0
        assert pool.allocate(2, 4) is False
        assert_invariants(pool)


class TestAppend:
    def test_fits_in_last_block(self):
        """2 tokens 只占 1 块且块内还有 2 个 slot，追加 2 个 token 不需要新块。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 2) is True
        assert pool.append(1, 2) is True
        assert pool.block_table(1) == [0]
        assert pool.seq_token_table[1] == 4
        assert pool.avail_block_num() == 7
        assert_invariants(pool)

    def test_crosses_into_new_block(self):
        """2 tokens 追加 5 -> 7 tokens 需要 2 块，新增 1 块。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 2) is True
        assert pool.append(1, 5) is True
        assert pool.block_table(1) == [0, 1]
        assert pool.seq_token_table[1] == 7
        assert_invariants(pool)

    def test_multiple_new_blocks(self):
        """2 tokens 追加 10 -> 12 tokens 需要 3 块，新增 2 块。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 2) is True
        assert pool.append(1, 10) is True
        assert pool.block_table(1) == [0, 1, 2]
        assert pool.seq_token_table[1] == 12
        assert_invariants(pool)

    def test_exactly_at_block_boundary(self):
        """4 tokens 恰好占满 1 块，再追加 1 个 token 必须新开 1 块。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 4) is True
        assert pool.block_table(1) == [0]
        assert pool.append(1, 1) is True
        assert pool.block_table(1) == [0, 1], "跨块边界时必须新分配块"
        assert pool.seq_token_table[1] == 5
        assert pool.avail_block_num() == 6
        assert_invariants(pool)

    def test_append_zero_tokens_rejected(self):
        """append 0 个 token 同样属于非法输入：抛 ValueError，不占新块也不改 token 计数。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 8) is True   # 占 [0, 1]
        assert pool.allocate(2, 4) is True   # 占 [2]
        assert pool.free(1) is True          # 0, 1 变回空闲
        snapshot = (pool.avail_block_num(), pool.block_table(2), pool.seq_token_table[2])
        with pytest.raises(ValueError):
            pool.append(2, 0)
        assert (pool.avail_block_num(), pool.block_table(2), pool.seq_token_table[2]) == snapshot
        assert_invariants(pool)

    def test_unknown_seq(self):
        """对不存在的序列 append 返回 False。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.append(99, 4) is False

    def test_insufficient_is_atomic(self):
        """块不够时返回 False，且不改任何状态。"""
        pool = KVCacheBlockPool(num_blocks=4, block_size=4)
        assert pool.allocate(1, 8) is True
        snapshot = (pool.avail_block_num(), list(pool.block_alloc_table), pool.block_table(1), pool.seq_token_table[1])
        assert pool.append(1, 100) is False
        assert (pool.avail_block_num(), list(pool.block_alloc_table), pool.block_table(1), pool.seq_token_table[1]) == snapshot

    def test_token_by_token_append(self):
        """逐 token 追加 20 次：每 4 个 token 长一块，全程不变量成立。"""
        pool = KVCacheBlockPool(num_blocks=32, block_size=4)
        assert pool.allocate(1, 1) is True
        for i in range(2, 21):
            assert pool.append(1, 1) is True, f"追加第 {i} 个 token 失败"
            assert len(pool.block_table(1)) == ceil_div(i, 4), f"{i} tokens 时块数应为 {ceil_div(i, 4)}"
            assert_invariants(pool)


class TestFree:
    def test_free_returns_all_blocks(self):
        """释放后所有块归还，三张表清空。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 10) is True
        assert pool.free(1) is True
        assert pool.avail_block_num() == 8
        assert pool.block_alloc_table == []
        assert not pool.seq_block_table and not pool.seq_token_table
        assert_invariants(pool)

    def test_freed_blocks_are_reusable(self):
        """释放后的块能被再次分配，且仍按编号最小优先。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 10) is True
        assert pool.free(1) is True
        assert pool.allocate(2, 4) is True
        assert pool.block_table(2) == [0]
        assert_invariants(pool)

    def test_free_twice(self):
        """重复释放返回 False。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 4) is True
        assert pool.free(1) is True
        assert pool.free(1) is False
        assert_invariants(pool)

    def test_free_unknown(self):
        """释放不存在的序列返回 False。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.free(99) is False


class TestInvalidInput:
    def test_allocate_zero_tokens_rejected(self):
        """0 token 属于非法输入：抛 ValueError，且不注册任何状态（vLLM 的 allocate_slots 同样拒绝）。"""
        pool = KVCacheBlockPool(num_blocks=4, block_size=4)
        with pytest.raises(ValueError):
            pool.allocate(1, 0)
        assert 1 not in pool.seq_token_table
        assert 1 not in pool.seq_block_table
        assert pool.avail_block_num() == 4
        assert pool.block_alloc_table == []

    def test_allocate_negative_tokens_rejected(self):
        """负数 token 属于非法输入：抛 ValueError，不能写出负 token 计数、更不能吞掉一个块。"""
        pool = KVCacheBlockPool(num_blocks=4, block_size=4)
        with pytest.raises(ValueError):
            pool.allocate(1, -4)
        assert 1 not in pool.seq_token_table
        assert pool.avail_block_num() == 4
        assert pool.block_alloc_table == []

    def test_append_negative_tokens_rejected(self):
        """append 负数同样抛 ValueError，且不改状态。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 4) is True
        snapshot = (pool.avail_block_num(), pool.block_table(1), pool.seq_token_table[1])
        with pytest.raises(ValueError):
            pool.append(1, -4)
        assert (pool.avail_block_num(), pool.block_table(1), pool.seq_token_table[1]) == snapshot

    def test_block_table_unknown_seq_raises(self):
        """未知序列查询块表抛 KeyError（当前约定，与 allocate/free 返回 False 的风格不同）。"""
        pool = KVCacheBlockPool(num_blocks=4, block_size=4)
        with pytest.raises(KeyError):
            pool.block_table(99)


class TestAppendBoundaries:
    def test_append_exact_block_from_boundary(self):
        """恰好占满 1 块后追加 4 个 token（整块）→ 新增 1 块。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 4) is True
        assert pool.append(1, 4) is True
        assert pool.block_table(1) == [0, 1]
        assert pool.seq_token_table[1] == 8
        assert_invariants(pool)

    def test_append_leftover_equals_free_slots(self):
        """2/4 占块，追加 6（1 整块 + 2 剩余）恰好用满最后 2 个 slot → 只新增 1 块。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 2) is True
        assert pool.append(1, 6) is True
        assert pool.block_table(1) == [0, 1]
        assert pool.seq_token_table[1] == 8
        assert_invariants(pool)

    def test_append_uses_last_available_block(self):
        """只剩 1 块时，正好需要 1 块的 append 必须成功并用光空闲块。"""
        pool = KVCacheBlockPool(num_blocks=2, block_size=4)
        assert pool.allocate(1, 4) is True
        assert pool.avail_block_num() == 1
        assert pool.append(1, 1) is True
        assert pool.avail_block_num() == 0
        assert pool.block_table(1) == [0, 1]
        assert_invariants(pool)

    def test_block_size_one(self):
        """block_size=1：每个 token 独占一块，追加即长块。"""
        pool = KVCacheBlockPool(num_blocks=4, block_size=1)
        assert pool.allocate(1, 1) is True
        assert pool.block_table(1) == [0]
        assert pool.append(1, 2) is True
        assert pool.block_table(1) == [0, 1, 2]
        assert pool.seq_token_table[1] == 3
        assert_invariants(pool)

    def test_large_block_size_partial_fill(self):
        """block_size=8：1 token 占 1 块；追加 7 个仍不新开块，再追加 1 个才新开。"""
        pool = KVCacheBlockPool(num_blocks=4, block_size=8)
        assert pool.allocate(1, 1) is True
        assert pool.append(1, 7) is True
        assert pool.block_table(1) == [0]
        assert pool.seq_token_table[1] == 8
        assert pool.append(1, 1) is True
        assert pool.block_table(1) == [0, 1]
        assert_invariants(pool)


class TestPoolEdge:
    def test_zero_block_pool(self):
        """num_blocks=0：任何正 token 的分配都必须失败。"""
        pool = KVCacheBlockPool(num_blocks=0, block_size=4)
        assert pool.allocate(1, 1) is False
        assert pool.avail_block_num() == 0
        assert pool.block_alloc_table == []

    def test_exhaust_then_free_then_allocate(self):
        """池子耗尽 → 分配失败 → 释放后同一块可被重新分配。"""
        pool = KVCacheBlockPool(num_blocks=2, block_size=4)
        assert pool.allocate(1, 4) is True
        assert pool.allocate(2, 4) is True
        assert pool.avail_block_num() == 0
        assert pool.allocate(3, 4) is False
        assert pool.free(1) is True
        assert pool.allocate(3, 4) is True
        assert pool.block_table(3) == [0]
        assert_invariants(pool)

    def test_reallocate_same_seq_after_free(self):
        """释放后同一 seq_id 可以重新分配。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 8) is True
        assert pool.free(1) is True
        assert pool.allocate(1, 4) is True
        assert pool.block_table(1) == [0]
        assert pool.seq_token_table[1] == 4
        assert_invariants(pool)

    def test_append_to_freed_seq(self):
        """已释放的序列再 append 必须返回 False。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 4) is True
        assert pool.free(1) is True
        assert pool.append(1, 4) is False
        assert_invariants(pool)

    def test_block_table_returns_copy(self):
        """block_table 返回副本，外部修改不影响内部块表。"""
        pool = KVCacheBlockPool(num_blocks=8, block_size=4)
        assert pool.allocate(1, 8) is True
        table = pool.block_table(1)
        table.append(99)
        assert pool.block_table(1) == [0, 1]
        assert_invariants(pool)


class TestRandomStress:
    def test_random_ops_keep_invariants(self):
        """随机 allocate / append / free 1000 轮，每一步都校验不变量与返回值语义。"""
        random.seed(20261022)
        bs = 4
        pool = KVCacheBlockPool(num_blocks=16, block_size=bs)
        live: dict[int, int] = {}
        for step in range(1000):
            r = random.random()
            if r < 0.35 and live:
                seq_id = random.choice(list(live))
                assert pool.free(seq_id) is True
                del live[seq_id]
            elif r < 0.6 and live:
                seq_id = random.choice(list(live))
                n = random.randint(1, 10)
                avail_before = pool.avail_block_num()
                need = ceil_div(live[seq_id] + n, bs) - ceil_div(live[seq_id], bs)
                assert pool.append(seq_id, n) == (need <= avail_before), f"step {step}: append 返回值不符合预期"
                if need <= avail_before:
                    live[seq_id] += n
            else:
                seq_id = 10000 + step
                n = random.randint(1, 40)
                avail_before = pool.avail_block_num()
                need = ceil_div(n, bs)
                assert pool.allocate(seq_id, n) == (need <= avail_before), f"step {step}: allocate 返回值不符合预期"
                if need <= avail_before:
                    live[seq_id] = n
            assert_invariants(pool)
