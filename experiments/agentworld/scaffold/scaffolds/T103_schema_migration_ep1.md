## Scaffold: SQLite Schema Migration with Data Quality Handling

**1. Inspect First**
- Read `old_schema.sql`, `new_schema.sql`, and dump `test_data.db` (`.schema` + `SELECT *` from every table)
- Compare column sets, types, constraints (NOT NULL, UNIQUE, CHECK, FK), and defaults between schemas
- Catalog data quality issues: NULLs in NOT NULL columns, constraint violations, orphaned FK refs, duplicates, out-of-range values

**2. Decomposition — Ordered Sub-goals**
1. Backup source DB before any mutation
2. Identify column mappings (renames, splits, merges) between old→new
3. Design per-table transformation rules:
   - NULL/empty → default or sentinel value
   - Out-of-range → clamp or reject
   - Duplicate UNIQUE → deduplicate (suffix, merge, or drop)
   - Orphaned FK refs → create placeholder row or drop child
   - Invalid enum/check values → map to valid default
   - New columns without source → fill with schema defaults
4. Determine migration order (parents before children for FK integrity)
5. Compute derived columns (e.g., `total = qty * price`)

**3. Tool-Call Workflow**
- `sandbox_file_read` → both schema files
- `sandbox_shell_exec` → `sqlite3 <db> ".schema"` and `SELECT * FROM <table>` per table
- Write script to `sandbox_file_write`
- Run migration via `sandbox_shell_exec` (`python3 migrate_data.py`)
- Validate: dump migrated DB schema and row counts

**4. Failure Modes + Recovery**
- **Constraint violation on INSERT**: pre-validate each row; transform before insert
- **UNIQUE conflict**: track seen values in a set; generate suffix on collision
- **FK breakage**: resolve orphans in a dedicated pass before child inserts
- **SQLite executescript drops transactions**: use explicit `BEGIN/COMMIT` for atomicity
- **AUTOINCREMENT gaps**: explicitly set `id` on insert to preserve identity

**5. Verification**
- `SELECT COUNT(*)` matches per table (minus intentionally dropped rows)
- Schema dump matches `new_schema.sql` exactly
- Spot-check transformed rows: defaults applied, ranges valid, no NULLs in NOT NULL cols
- FK integrity: `PRAGMA foreign_key_check` returns empty
- Re-run on fresh copy produces identical output (idempotent)