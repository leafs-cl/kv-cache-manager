class KVCacheBlockPool:
    """
    KV Cache Block Pool to manage kv-cache page
    Maintain block table for kv cache page
    Not responsible for real gpu memory allocation
    vllm will acquire gpu memory and distribute tensor for each attention layer at initialization
    Table will be shared for each attention layer (Only focus on full attention layer)
    """
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.seq_block_table : dict[int, list[int]] = dict()
        self.block_alloc_table : list[int] = list()
        self.seq_token_table : dict[int, int] = dict()
        self.available_block_num = num_blocks

    def allocate(self, seq_id, num_tokens) -> bool:
        f"""Allocate kv cache block for a new sequence with {num_tokens} tokens, remember that if allocation fails, don;t change any states of kv table"""
        if seq_id in self.seq_token_table:
            print(f"Sequence {seq_id} have already allocated kv cache, should not allocate again")
            return False
        require_block_size = num_tokens // self.block_size if num_tokens % self.block_size == 0 else num_tokens // self.block_size + 1
        if require_block_size > self.available_block_num:
            print(f"Sequence {seq_id} require {require_block_size} kv cache block, but available kv cache block {self.available_block_num} can't satisfy")
            return False
        # Allocate for a new sequence must use full block, otherwise, will affect kernel processing speed
        malloc_block = []
        for i in range(self.num_blocks):
            if i not in self.block_alloc_table:
                malloc_block.append(i)
            if len(malloc_block) * self.block_size >= num_tokens:
                self.block_alloc_table.extend(malloc_block)
                self.seq_block_table[seq_id] = malloc_block
                self.available_block_num -= len(malloc_block)
                self.seq_token_table[seq_id] = num_tokens
                break

        return True

    def free(self, seq_id):
        """
        Free kv cache block of sequence {seq_id}
        """
        if seq_id not in self.seq_block_table:
            print(f"Sequence {seq_id} have not allocated kv cache")
            return False
        for block in self.seq_block_table[seq_id]:
            self.block_alloc_table.remove(block)
        self.available_block_num += len(self.seq_block_table[seq_id])
        self.seq_token_table.pop(seq_id)
        self.seq_block_table.pop(seq_id)
        return True

    def block_table(self, seq_id):
        """
        Return kv cache block of sequence {seq_id}
        """
        return list(self.seq_block_table[seq_id])

    def avail_block_num(self):
        return self.available_block_num
