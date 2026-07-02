## Scaffold: Binary Protocol Parser

### 1. Decomposition
1. Read spec document; extract frame format (marker, fields, lengths, endianness, checksum algorithm).
2. Inspect binary capture: size, byte distribution, marker frequency, structural regularity.
3. Resolve ambiguities by cross-referencing spec claims against observed byte patterns (e.g., does checksum field match spec algorithm?).
4. Implement parser: scan for markers, parse fixed header, extract variable payload, validate checksum.
5. Emit one JSON object per frame to JSONL.

### 2. Tool-Call Workflow
- **First**: Read spec file fully (may be sparse or misleading).
- **Then**: `ls -la` fixtures; confirm capture.bin exists and has non-zero size. If missing, check for alternative data sources (test_data.json, setup scripts).
- **Inspect binary**: Use `od`/`hexdump` to sample first 200 bytes and scan for marker byte frequency. Look for repeating patterns indicating frame boundaries.
- **Validate checksum hypothesis**: Compute candidate checksums (SHA1[0], sum mod 256, XOR, CRC8) over observed frames; check which matches the trailing byte.
- **Code**: Write parser incrementally; test against known-good bytes from spec examples if available.

### 3. Failure Modes + Recovery
- **Missing capture.bin**: Check alternate paths, generate from test fixtures, or produce empty output with warning.
- **Spec says SHA1 but field is 1 byte**: Try truncation strategies (first byte, last byte, byte 10). If none work, reconsider: maybe it's not SHA1 at all — test simple sums/XORs.
- **Length field ambiguity** (payload-only vs. payload+checksum): Try both; the one producing valid checksums wins.
- **Malformed/truncated trailing frame**: Parse greedily, mark as error, don't abort.
- **Overlapping frames**: Prefer longest valid parse at each marker position.

### 4. Verification
- Run parser; confirm every line in JSONL is valid JSON.
- Spot-check first 3 frames manually against hex dump.
- Verify checksum validity rate is high (>90%) — low rate indicates wrong algorithm.
- Confirm no frames skipped: total parsed bytes ≈ file size (minus trailing garbage).
- Ensure JSONL has no trailing newline issues; each line terminates with `\n`.