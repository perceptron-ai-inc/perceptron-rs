# Perceptron Rust SDK

Rust SDK for Perceptron AI vision models.

## Conventions

- **Builder pattern**: Request structs use `new()` + chained setters (e.g. `.reasoning(true).temperature(0.7)`)
- **Generation params**: Flat fields per request struct with a `generation_param_setters!()` macro for shared setters
- **Dependencies**: Use 3-part versions (e.g. `"0.28.0"` not `"0.28"`) for predictable resolution
- **Minimal dependencies**: Only add crates when they provide clear value over manual code
- **No panics**: SDK code must not panic; use `Result` or compile-time safety (enums) instead. If a failure is truly impossible, use `.expect("reason")` over `.unwrap()`
- **Enums over strings**: Prefer enums for closed sets to make invalid states unrepresentable
- **strum**: Use `strum::Display` and `strum::EnumString` on enums that need string conversion (e.g. for MCP compatibility)

## Testing

- Run: `cargo test`
- All integration tests use `wiremock` to mock the chat completions API
- Shared helpers live in `tests/common.rs` — use `common::setup()`, `common::response()`, and `common::mock_response()` instead of duplicating boilerplate
- Every mock must include a request matcher (via `body_partial_json`) — no unverified mocks
- Each endpoint should have at least one test for every `Media` variant
- Extract shared response content and assertion helpers into per-file functions to reduce repetition
- Keep test names descriptive of the behavior being tested (e.g. `base64_media`, `with_reasoning`, `point_format`)

## Version Bumps

- Version is in `Cargo.toml` under `[package] version`
- Version bumps must be a **standalone commit** (no other changes)
- To determine the bump level, diff from the last version bump commit to HEAD:
  - **Patch** (0.x.Y): Bug fixes, internal refactors, doc changes
  - **Minor** (0.X.0): New features, new public types/methods, breaking API changes, new dependencies
