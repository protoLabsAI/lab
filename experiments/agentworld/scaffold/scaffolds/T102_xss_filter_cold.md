1. **Decomposition**
   - Define a strict allow-list of safe HTML tags, attributes, and URL schemes.
   - Implement a multi-pass sanitizer: decode encoded payloads first, then strip dangerous constructs, then re-validate.
   - Preserve document structure by only removing malicious nodes/attributes, not rewriting the tree.

2. **Tool-Call Workflow**
   - Inspect sample attack files to catalog diverse vectors (encoded, mixed-case, nested).
   - Inspect clean files to identify legitimate structures that must survive.
   - Draft `filter.py` using `html.parser` or regex-based stripping for standard library compliance.
   - Iterate: run against attacks (must be neutralized) and cleans (must be unchanged).

3. **Failure Modes + Recovery**
   - *Over-sanitization*: Legitimate content stripped. → Relax allow-list; verify clean file diffs.
   - *Under-sanitization*: Encoded XSS bypasses filter. → Add pre-decoding pass (URL decode, HTML entity decode) before sanitization.
   - *Structure corruption*: Valid HTML broken. → Use DOM-aware parsing instead of regex where possible.
   - *Performance*: Slow on large files. → Optimize regex patterns; avoid recursive deep copies.

4. **Verification**
   - Run filter on all 37 samples.
   - For attacks: diff output against original; confirm zero JS execution vectors remain (grep for `script`, `on*`, `javascript:`, `data:text/html`).
   - For cleans: diff output against original; confirm byte-identical or structurally equivalent output.
   - Exit code must be 0 on all runs.