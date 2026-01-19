# Agent prompt template for confusion sets and weights

Copy/paste this prompt into ChatGPT or Gemini to generate a suite spec fragment. The agent should return valid JSON only.

```
You are designing semantic confusion clusters and test weights for a QuickDraw stroke-sequence capability suite.

Input format:
- classes: <comma-separated class list>
- base_tests: provided separately in default_spec.json

Output JSON keys:
- confusion_sets: array of {name, classes, pairs}
- scoring: {weights: {<test_name>: weight}}

Constraints:
- Keep weights normalized to a max of 1.0.
- Choose 2-4 clusters; include at least one pair per cluster.
- Do not include free-form text outside JSON.
```

Example output:

```
{
  "confusion_sets": [
    {"name": "containers", "classes": ["cup", "mug", "bottle", "bowl"], "pairs": [["cup","mug"], ["bowl","cup"]]},
    {"name": "strings", "classes": ["violin","guitar","banjo"], "pairs": [["violin","guitar"]]}
  ],
  "scoring": {
    "weights": {
      "resample_uniform": 1.0,
      "stroke_dropout": 1.0,
      "confusion_sets": 1.0
    }
  }
}
```
