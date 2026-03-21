"""Creative writing and roleplay graders — LLM-as-judge scoring.

Dimensions:
- Narrative quality: prose quality, pacing, descriptive richness
- Character voice: consistency, distinctiveness, emotional depth
- World building: atmospheric detail, internal consistency
- Engagement: hook, tension, reader investment
- Instruction following: adherence to prompt constraints (tone, length, style)
"""

from __future__ import annotations

from graders.llm_judge import LLMJudge

NARRATIVE_RUBRIC = """You are a literary critic evaluating AI-generated creative writing.

## Prompt Given
{input}

## Text to Evaluate
{output}

## Scoring — Narrative Quality
- 1.0: Publication-quality prose. Vivid imagery, varied sentence structure, strong voice.
- 0.75: Good writing. Clear, engaging, occasional strong moments.
- 0.5: Adequate. Readable but generic, lacks distinctive voice.
- 0.25: Weak. Clichéd, repetitive, flat descriptions.
- 0.0: Poor. Incoherent, grammatically broken, or empty.

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""

CHARACTER_VOICE_RUBRIC = """You are evaluating AI-generated character dialogue and roleplay.

## Character/Scene Description
{input}

## AI Output
{output}

## Expected Behavior (if available)
{expected}

## Scoring — Character Voice
- 1.0: Distinctive, consistent voice. Dialogue feels natural and reveals personality. Emotional depth.
- 0.75: Good characterization. Voice is mostly consistent with occasional generic moments.
- 0.5: Adequate. Character is recognizable but could be anyone. Surface-level personality.
- 0.25: Weak. Generic dialogue, inconsistent personality, breaks character.
- 0.0: No characterization. Robot-like or completely off-character.

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""

WORLD_BUILDING_RUBRIC = """You are evaluating world-building and atmospheric detail in AI-generated text.

## Prompt
{input}

## AI Output
{output}

## Scoring — World Building
- 1.0: Rich, immersive world. Sensory details, internal consistency, lived-in feeling.
- 0.75: Good atmosphere. Setting feels real with occasional vague areas.
- 0.5: Adequate. Basic setting described but lacks depth or sensory detail.
- 0.25: Thin. Generic fantasy/sci-fi tropes, no unique world elements.
- 0.0: No world building. Setting is absent or contradictory.

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""

ENGAGEMENT_RUBRIC = """You are evaluating how engaging and compelling AI-generated text is.

## Prompt
{input}

## AI Output
{output}

## Scoring — Engagement
- 1.0: Compelling. Strong hook, maintained tension, want to read more.
- 0.75: Engaging. Interesting premise, mostly sustains interest.
- 0.5: Adequate. Readable but doesn't pull you in.
- 0.25: Dull. Predictable, no tension, skimmable.
- 0.0: Unengaging. Would stop reading immediately.

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""

GM_QUALITY_RUBRIC = """You are evaluating an AI Game Master's response in a tabletop RPG session.

## Player Action / Situation
{input}

## GM Response
{output}

## Expected GM Behavior (if available)
{expected}

## Scoring — GM Quality
Evaluate across these sub-dimensions:
- Does the GM advance the narrative (not stall or loop)?
- Does the GM present meaningful choices (not railroad)?
- Does the GM use the game mechanics appropriately (dice, skills, consequences)?
- Does the GM maintain world consistency?
- Does the GM create tension and dramatic moments?

- 1.0: Expert GM. Advances plot, presents choices, uses mechanics, creates drama.
- 0.75: Good GM. Mostly hits the marks, occasional missed opportunities.
- 0.5: Adequate. Functional but mechanical, lacks dramatic flair.
- 0.25: Weak. Railroads, ignores mechanics, or breaks immersion.
- 0.0: Bad GM. Contradicts world, ignores player agency, or stalls.

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""

RESEARCH_SYNTHESIS_RUBRIC = """You are evaluating the quality of an AI-generated research synthesis.

## Research Question
{input}

## AI Synthesis
{output}

## Source Material (if available)
{expected}

## Scoring — Research Synthesis Quality
- 1.0: Expert-level synthesis. Multiple perspectives, nuanced analysis, proper attribution, identifies gaps.
- 0.75: Good synthesis. Covers key points, some analysis depth, mostly attributed.
- 0.5: Adequate. Summarizes sources but lacks synthesis or critical analysis.
- 0.25: Weak. Superficial, misses key points, or misrepresents sources.
- 0.0: Poor. Hallucinated content, no source grounding, or off-topic.

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}
"""


def narrative_judge(**kwargs) -> LLMJudge:
    return LLMJudge(dimension="narrative_quality", rubric=NARRATIVE_RUBRIC, **kwargs)

def character_voice_judge(**kwargs) -> LLMJudge:
    return LLMJudge(dimension="character_voice", rubric=CHARACTER_VOICE_RUBRIC, **kwargs)

def world_building_judge(**kwargs) -> LLMJudge:
    return LLMJudge(dimension="world_building", rubric=WORLD_BUILDING_RUBRIC, **kwargs)

def engagement_judge(**kwargs) -> LLMJudge:
    return LLMJudge(dimension="engagement", rubric=ENGAGEMENT_RUBRIC, **kwargs)

def gm_quality_judge(**kwargs) -> LLMJudge:
    return LLMJudge(dimension="gm_quality", rubric=GM_QUALITY_RUBRIC, **kwargs)

def research_synthesis_judge(**kwargs) -> LLMJudge:
    return LLMJudge(dimension="research_synthesis", rubric=RESEARCH_SYNTHESIS_RUBRIC, **kwargs)


def all_creative_judges(**kwargs) -> list[LLMJudge]:
    """All creative writing judges."""
    return [narrative_judge(**kwargs), character_voice_judge(**kwargs),
            world_building_judge(**kwargs), engagement_judge(**kwargs)]

def all_roleplay_judges(**kwargs) -> list[LLMJudge]:
    """All roleplay/GM judges."""
    return [character_voice_judge(**kwargs), world_building_judge(**kwargs),
            engagement_judge(**kwargs), gm_quality_judge(**kwargs)]
