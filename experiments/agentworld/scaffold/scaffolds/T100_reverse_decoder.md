## Scaffold: Binary Encoding/Decoding Task

**1. Inspect**
- Read `decoder.py` to identify encoding scheme (zlib, base64, custom, etc.)
- Read `target.txt` to understand desired output format
- Note file sizes for constraint checking

**2. Analyze Constraints**
- Calculate max allowed size: `0.6 × target.txt size`
- Determine if decoder expects raw binary, hex, or specific format
- Check if decoder has error handling or strict parsing

**3. Generate Encoded Data**
- Apply inverse of decoder's encoding to `target.txt`
- If decoder uses compression (zlib, gzip, lzma), compress target
- If decoder expects specific format (base64, hex), encode accordingly
- Write result to `encoded.dat`

**4. Verify**
```bash
python decoder.py < encoded.dat | diff - target.txt
```
- Confirm exact match (no trailing whitespace, newlines)
- Check size: `wc -c encoded.dat` ≤ 60% of `wc -c target.txt`

**Failure Modes:**
- **Mismatch**: Decoder expects specific header/magic bytes → prepend if needed
- **Size overflow**: Use stronger compression or different algorithm
- **Encoding errors**: Check if decoder reads from `sys.stdin` vs `sys.stdin.buffer`
- **Trailing newlines**: Verify `diff` shows exact match, not just content

**Recovery:**
- If compression insufficient, try multiple algorithms (zlib vs gzip vs lzma)
- If format wrong, inspect decoder source for parsing logic
- Test incrementally: decode small sample first