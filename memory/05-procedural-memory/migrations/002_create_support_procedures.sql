CREATE TABLE IF NOT EXISTS support_procedures (
    id UUID PRIMARY KEY,
    procedure_key TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    procedure_summary TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft', 'active', 'deprecated')),
    steps JSONB NOT NULL CHECK (jsonb_typeof(steps) = 'array'),
    preconditions JSONB NOT NULL CHECK (jsonb_typeof(preconditions) = 'array'),
    completion_conditions JSONB NOT NULL CHECK (jsonb_typeof(completion_conditions) = 'array'),
    escalation_rules JSONB NOT NULL CHECK (jsonb_typeof(escalation_rules) = 'array'),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object'),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    embedding VECTOR({{EMBEDDING_DIMENSIONS}}) NOT NULL,
    UNIQUE (procedure_key, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS support_procedures_one_active_key
ON support_procedures (procedure_key) WHERE status = 'active';
