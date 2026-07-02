## Scaffold: HTML Sanitizer Implementation

### 1. Decomposition
- **Inspect fixtures first**: List all attack/clean files, sample 3-5 of each to catalog XSS patterns
- **Classify vectors**: Group by type (script tags, event handlers, javascript: URLs, encoded payloads, dangerous embeds, CSS expressions, SVG/MathML injection)
- **Design multi-pass pipeline**: Pre-decode → structural parse → attribute sanitization → post-cleanup
- **Define safe allowlists**: Tags, attributes, URL schemes, CSS properties

### 2. Tool-Call Workflow
1. `ls` fixture directory to enumerate files
2. `cat` 2-3 attack files to identify patterns
3. `cat` 2-3 clean files to understand preservation requirements
4. Write filter.py with iterative testing against samples
5. Run against all 37 files, diff clean outputs for unintended changes

### 3. Failure Modes + Recovery
- **Over-sanitization**: Clean files lose content → check diffs, relax allowlists
- **Under-sanitization**: Attack files still contain vectors → add regex patterns, decode entities recursively
- **Entity encoding bypass**: `&#111;nerror`, `%3Cscript%3E` → pre-decode all entities before pattern matching
- **HTMLParser limitations**: Doesn't handle malformed HTML → add regex pre-processing
- **Style attribute injection**: `expression()`, `url(javascript:)` → sanitize style values

### 4. Verification
- Run filter on all attack files → grep for remaining XSS patterns (script, on*, javascript:, data:text/html)
- Run filter on all clean files → diff against originals (should be identical or minimal whitespace changes)
- Test edge cases: mixed case, nested encoding, comments, CDATA, SVG/MathML contexts
- Exit code check: confirm 0 on success