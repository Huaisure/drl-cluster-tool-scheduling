---
status: accepted
---

# Separate policy intents from auditable events

Let the policy choose parameterized intents, while the kernel expands each accepted intent into its complete boundary bundle and auditable schedule events. Pick and Place use multi-boundary transport bundles; Pump, Vent, and Clean remain explicit in the internal schedule; Process is an explicit interval created automatically by the kernel rather than a policy action. This preserves a compact, general model interface without hiding physical transitions from validation.

## Consequences

An intent is not synonymous with one event, and immediate enablement is weaker than commitment of the full bundle. Validators operate on expanded events and intervals, while training data may retain both the chosen intent and its deterministic expansion.
