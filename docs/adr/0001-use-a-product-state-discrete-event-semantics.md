---
status: accepted
---

# Use a product-state discrete-event semantics

Represent scheduling as a unified discrete-event state transition system whose state is the product of orthogonal equipment, wafer, resource, and obligation variables. Do not encode the domain as a giant enumerated machine: for example, load-lock pressure transition and wafer cooling are independent axes that may advance concurrently. This keeps new constraints expressible through the same guards, claims, effects, triggers, and obligations instead of adding model-specific branches.

## Consequences

Every transition must be attributable to an explicit event boundary, and a decision may be requested only after same-tick consequences reach a stable fixed point. The policy consumes the generic state and intent interface; the semantic kernel remains the authority for legality and state changes.
