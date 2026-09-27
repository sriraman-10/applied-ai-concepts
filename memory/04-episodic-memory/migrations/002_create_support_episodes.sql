CREATE SEQUENCE IF NOT EXISTS support_case_number_seq;

CREATE TABLE IF NOT EXISTS support_episodes (
    id UUID PRIMARY KEY,
    episode_key TEXT NOT NULL UNIQUE,
    customer_id TEXT,
    issue_type TEXT NOT NULL,
    description TEXT NOT NULL,
    episode_summary TEXT,
    actions JSONB NOT NULL DEFAULT '[]'::jsonb,
    observations JSONB NOT NULL DEFAULT '[]'::jsonb,
    resolution TEXT,
    outcome TEXT CHECK (outcome IN ('resolved', 'escalated', 'failed', 'abandoned')),
    status TEXT NOT NULL CHECK (status IN ('open', 'completed')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    embedding VECTOR({{EMBEDDING_DIMENSIONS}}),
    CHECK (length(btrim(episode_key)) > 0),
    CHECK (length(btrim(issue_type)) > 0),
    CHECK (length(btrim(description)) > 0),
    CHECK (jsonb_typeof(actions) = 'array'),
    CHECK (jsonb_typeof(observations) = 'array'),
    CHECK (jsonb_typeof(metadata) = 'object'),
    CHECK (
        (
            status = 'open'
            AND episode_summary IS NULL
            AND resolution IS NULL
            AND outcome IS NULL
            AND completed_at IS NULL
            AND embedding IS NULL
        )
        OR
        (
            status = 'completed'
            AND episode_summary IS NOT NULL
            AND resolution IS NOT NULL
            AND outcome IS NOT NULL
            AND completed_at IS NOT NULL
            AND embedding IS NOT NULL
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_support_episodes_issue_type
    ON support_episodes (issue_type);
CREATE INDEX IF NOT EXISTS idx_support_episodes_customer
    ON support_episodes (customer_id) WHERE customer_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_support_episodes_status_outcome
    ON support_episodes (status, outcome);
