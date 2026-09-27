CREATE INDEX IF NOT EXISTS idx_support_episodes_embedding_hnsw
    ON support_episodes USING hnsw (embedding vector_cosine_ops)
    WHERE status = 'completed' AND outcome = 'resolved';
