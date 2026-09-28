CREATE INDEX IF NOT EXISTS support_procedures_embedding_hnsw
ON support_procedures USING hnsw (embedding vector_cosine_ops)
WHERE status = 'active';
