**Scaffold: Binary Protocol Decoder**

1. **Decompose**
   - **Parse Spec**: Extract field layouts, magic bytes, endianness, and state transitions. Note ambiguities.
   - **Inspect Raw**: Sample first 512 bytes of `capture.bin`. Identify framing, headers, and delimiter patterns.
   - **Implement Parser**: Build incremental parser handling variable-length fields, padding, and error cases.
   - **Validate & Export**: Cross-check parsed fields against spec; emit JSONL.

2. **Tool Workflow**
   - **Read Spec** (`cat`/`head`): Prioritize field definitions, enums, and edge cases.
   - **Hex Dump** (`xxd`/`hexdump`): Verify magic bytes, endianness, and structural consistency in raw data.
   - **Iterative Coding**: Write parser in small chunks; test against known packet boundaries.
   - **Diff/Validate**: Compare decoded output against expected structure (if available) or internal consistency.

3. **Failure Modes + Recovery**
   - **Misaligned Parsing**: If checksums fail or fields look nonsensical, re-evaluate endianness or skip bytes.
   - **Ambiguous Spec**: Use empirical data (frequency analysis, known values) to resolve ambiguities.
   - **Truncated/Malformed Packets**: Decide on strict (abort) vs. lenient (skip) parsing; document decisions.
   - **Encoding Issues**: Ensure UTF-8/ASCII handling for string fields; handle null terminators.

4. **Verification**
   - **Round-trip Check**: Re-serialize decoded data and compare to original binary (if feasible).
   - **Statistical Sanity**: Check field value distributions (e.g., IDs sequential, lengths positive).
   - **Edge Cases**: Verify handling of first/last packets, zero-length payloads, and max-size fields.
   - **Output Format**: Confirm JSONL is valid JSON, one object per line, no trailing commas.