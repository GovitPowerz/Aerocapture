# Gap-Closure Seed Pool Eviction Design

## Problem

The adaptive seed pool's eviction strategy (`evict_redundant()` in `seed_pool.py`) always drops the easiest seed. Over many generations, this ratchets the pool toward exclusively hard scenarios -- adversarial curriculum collapse. The GA over-optimizes for edge cases and average-case performance degrades.

## Solution

Replace "evict easiest" with gap-closure eviction: evict the seed that is most similar in difficulty to another seed, preserving coverage across the full difficulty spectrum.

## Changes

### `seed_pool.py` -- `evict_redundant()` method only

**Current behavior:** Drop the seed with the lowest difficulty score.

**New behavior:**
1. Sort seeds by difficulty score
2. Find the pair of adjacent seeds with the smallest difficulty gap
3. From that pair, evict the seed that was added more recently (tiebreak: keep the older seed)
4. Repeat until `len(seeds) <= max_size`

### No other changes

- `stress_test()`: unchanged. Injected hard seeds play by the same eviction rules.
- `add_seeds()`, `score_difficulty()`, `evaluate_population()`: unchanged.
- Serialization (`to_dict`/`from_dict`): unchanged -- no new state.
- `evict_redundant()` call sites: unchanged -- same interface.

## Edge Cases

- Seeds with no difficulty score get `difficulty.get(s, 0.0)` -- they cluster at 0.0 and are candidates for gap-closure eviction against each other. Correct: unscored seeds are genuinely redundant.
- Pool with 0 or 1 seeds: no eviction needed (loop condition handles this).
- All seeds have identical difficulty: all gaps are 0, eviction picks the most recently added from any adjacent pair. Degrades gracefully.

## Tests

- Existing tests: verify they still pass (eviction behavior changes but API doesn't)
- New test: pool with known difficulty scores, verify eviction preserves difficulty spectrum endpoints (min/max survive)
- New test: verify that when two seeds have the same difficulty, the newer one is evicted
- New test: pool below max_size -- no eviction occurs
