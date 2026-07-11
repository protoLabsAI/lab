"""
Output parsing utilities for AgentWorldBench evaluation.
"""

import re
from typing import Dict, Any, List, Tuple, Optional

from .task_configs import TASK_CONFIGS


# =============================================================================
# Thinking Tags Removal
# =============================================================================

def _remove_thinking_tags(text: str, response_tag: str) -> str:
    """Remove <think> tags and their content from the entire text.
    
    This function is robust to various malformed or complex scenarios:
    1. Complete tags: <think>...</think>
    2. Multiple thinking blocks: Handles multiple disjoint blocks.
    3. Nested tags: Handles tags nested within each other (e.g., <think>...<think>...</think>...</think>).
    4. Left unclosed: Removes content before an orphaned </think>.
    5. Right unclosed: Removes content starting from an orphaned <think>.
       - If `response_tag` is provided, stops removal at the first occurrence of `response_tag` 
         to preserve the actual output.
       - Otherwise, removes everything to the end of the string.
    
    Args:
        text: Input text to clean
        response_tag: Optional response tag name to use as a boundary for right-unclosed tags.
                      This prevents deleting the valid response if the thinking tag is unclosed.
        
    Returns:
        Cleaned text with all thinking content removed.
    """
    if not text:
        return text
    
    # Find all <think> and </think> tags
    tags = []
    # Find opening tags
    for m in re.finditer(r"<think>", text, re.IGNORECASE):
        tags.append((m.start(), "open"))
    # Find closing tags
    for m in re.finditer(r"</think>", text, re.IGNORECASE):
        tags.append((m.start(), "close"))
    
    if not tags:
        return text
    
    # Sort by position
    tags.sort(key=lambda x: x[0])
    
    # Build list of ranges to remove
    ranges_to_remove = []
    used_indices = set()
    
    # First pass: match opening tags with corresponding closing tags
    for i, (pos, tag_type) in enumerate(tags):
        if i in used_indices:
            continue
        if tag_type == "open":
            # Find the next closing tag
            for j in range(i + 1, len(tags)):
                if j in used_indices:
                    continue
                j_pos, j_type = tags[j]
                if j_type == "close":
                    # Found matching pair
                    close_tag_len = len("</think>")
                    ranges_to_remove.append((pos, j_pos + close_tag_len))
                    used_indices.add(i)
                    used_indices.add(j)
                    break
    
    # Second pass: handle orphaned closing tags (left unclosed)
    # Remove everything from start to end of orphaned closing tag
    for i, (pos, tag_type) in enumerate(tags):
        if i in used_indices:
            continue
        if tag_type == "close":
            close_tag_len = len("</think>")
            ranges_to_remove.append((0, pos + close_tag_len))
            used_indices.add(i)
    
    # Third pass: handle orphaned opening tags (right unclosed)
    for i, (pos, tag_type) in enumerate(tags):
        if i in used_indices:
            continue
        if tag_type == "open":
            # For right unclosed, we need to be careful not to remove the actual response
            # if it comes after this tag.
            # Strategy: Look for response_tag AFTER this unclosed thinking tag.
            end_pos = len(text)
            if response_tag:
                # Find first occurrence of response_tag AFTER the unclosed think tag
                response_pattern = rf"<{re.escape(response_tag)}>"
                response_match = re.search(response_pattern, text[pos:], re.IGNORECASE)
                if response_match:
                    # Found a response tag, stop removal there
                    end_pos = pos + response_match.start()
            
            ranges_to_remove.append((pos, end_pos))
            used_indices.add(i)
    
    # Merge overlapping ranges and sort
    if not ranges_to_remove:
        return text
    
    ranges_to_remove.sort()
    merged_ranges = []
    for start, end in ranges_to_remove:
        if merged_ranges and start <= merged_ranges[-1][1]:
            # Overlapping, extend the previous range
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))
    
    # Build cleaned text
    result_parts = []
    prev_end = 0
    for start, end in merged_ranges:
        if start > prev_end:
            result_parts.append(text[prev_end:start])
        prev_end = end
    if prev_end < len(text):
        result_parts.append(text[prev_end:])
    
    return "".join(result_parts).strip()


# =============================================================================
# Output Parsing
# =============================================================================

def parse_model_output(raw_output: str, response_tag: str) -> str:
    """Parse and extract predicted output from model generation.
    
    The extraction process is designed to be robust against "thinking models" that might
    output fake response tags during their reasoning process.
    
    Steps:
    1. Remove all thinking tags (and their content) using `_remove_thinking_tags`.
    2. Extract content from the **LAST** valid response tag block found in the cleaned text.
       - Prioritizing the last block ensures we capture the final answer, ignoring any 
         fake or intermediate blocks that might have survived thinking removal (though unlikely).
    
    Args:
        raw_output: Raw model generation text
        response_tag: Tag name to extract content from (e.g., "predicted_observation")
        
    Returns:
        Extracted output string, or cleaned text if no tag found
    """
    if not raw_output:
        return "No output"
    
    # Step 1: Remove thinking tags
    cleaned_text = _remove_thinking_tags(raw_output, response_tag)
    
    # Step 2: Extract content from response tag
    # We want the LAST valid block.
    
    # Find all start tags
    start_pattern = rf"<{re.escape(response_tag)}>"
    start_matches = list(re.finditer(start_pattern, cleaned_text, re.IGNORECASE))
    
    if not start_matches:
        # Fallback: return the cleaned text as-is (model may not use XML tags)
        return cleaned_text.strip() if cleaned_text.strip() else "No output"
    
    # Take the last start tag
    last_start_match = start_matches[-1]
    start_pos = last_start_match.end()
    
    # Look for a closing tag AFTER the start tag
    close_pattern = rf"</{re.escape(response_tag)}>"
    close_match = re.search(close_pattern, cleaned_text[start_pos:], re.IGNORECASE)
    
    if close_match:
        # Found closing tag
        end_pos = start_pos + close_match.start()
        return cleaned_text[start_pos:end_pos].strip()
    else:
        # No closing tag, return everything until end
        return cleaned_text[start_pos:].strip()

def clean_response_marker(text: str, subtask: str) -> str:
    """Remove response markers (e.g., **Environment Observation:**) from text.
    
    Args:
        text: Text to clean
        subtask: Subtask name to get the correct marker
        
    Returns:
        Cleaned text with marker removed
    """
    marker = TASK_CONFIGS.get(subtask, {}).get("response_marker", "")
    if marker and text.startswith(marker):
        return text[len(marker):].strip()
    return text.replace(marker, "").strip() if marker else text.strip()
