CREATE TABLE IF NOT EXISTS procedure_executions (
    id UUID PRIMARY KEY,
    procedure_id UUID NOT NULL REFERENCES support_procedures(id) ON DELETE RESTRICT,
    procedure_key TEXT NOT NULL,
    procedure_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'escalated', 'failed')),
    current_step INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    outcome TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(metadata) = 'object')
);
CREATE TABLE IF NOT EXISTS procedure_step_executions (
    execution_id UUID NOT NULL REFERENCES procedure_executions(id) ON DELETE CASCADE,
    step_sequence INTEGER NOT NULL,
    action TEXT NOT NULL,
    tool_name TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'skipped')),
    result_summary TEXT,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    PRIMARY KEY (execution_id, step_sequence)
);
