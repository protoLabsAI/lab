### Scaffold: Binary Encoder Reverse-Engineering

**1. Decomposition**
1.  **Analyze Decoder**: Parse `decoder.py` to map input byte patterns to output characters/sequences. Identify the encoding scheme (e.g., Huffman, custom bit-packing, lookup table).
2.  **Analyze Target**: Compute target size and content statistics to estimate minimum encoded size.
3.  **Construct Payload**: Generate binary data that, when decoded, yields the target. If the scheme is reversible, invert the algorithm. If complex, use constraint solving or brute-force search with pruning.
4.  **Verify Size**: Ensure encoded size ≤ 60% of target size. If not, optimize encoding efficiency or check for redundant patterns.
5.  **Test**: Run decoder with encoded file and compare output to target.

**2. Tool-Call Workflow**
-   **Read** `decoder.py` fully. Use `cat` or text editor. Trace logic for edge cases.
-   **Read** `target.txt`. Use `cat` or `wc -c` for size.
-   **Write** `encoded.dat` using Python script or binary editor.
-   **Execute** `python decoder.py < encoded.dat` and capture stdout.
-   **Compare** output with `target.txt` using `diff` or `cmp`.

**3. Failure Modes + Recovery**
-   **Decoding Error**: Output differs from target. Re-examine decoder logic for off-by-one, endianness, or stateful errors. Add debug prints to decoder temporarily.
-   **Size Exceeded**: Encoded file too large. Check if encoding is optimal. Look for unused bits or redundant markers. Consider alternative encoding strategies if allowed.
-   **Ambiguous Encoding**: Multiple inputs map to same output. Ensure chosen input is valid and minimal.

**4. Verification**
-   **Exact Match**: Use `diff <(python decoder.py < encoded.dat) target.txt` — must return no output.
-   **Size Check**: `wc -c encoded.dat` and `wc -c target.txt`. Confirm ratio ≤ 0.6.
-   **Reproducibility**: Run test multiple times to ensure no randomness or environment dependency.