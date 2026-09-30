---
name: stub-single-file-js-selftest
description: 'Use when implementing a single-file JS/TS exercise (e.g. Exercism-style
  stub) whose official test suite is hidden: inspect scaffolding, build a throwaway
  self-test harness with spec-derived expectations, iterate, then clean up.'
---

# Single-File JS Self-Test Workflow

Applies to repo-based, single-file JS/TS exercises where the official grader/test file is hidden and only a stub plus prose spec is provided.

## Procedure

1. **Inspect scaffolding before coding.** Read `package.json`, any build/transform config (e.g. `babel.config.js`), and the stub file to determine the module system (ESM vs CJS), expected export names/shape, and whether the stub throws. Match the implementation to that contract. If the local harness needs a different module extension than the deliverable, copy the module for the harness rather than changing the deliverable.

2. **Write a throwaway self-test harness.** After implementing the module, create a temporary script (e.g. `selftest.mjs`, `tmp_selftest.mjs`) that imports the real module and asserts expected outputs. Keep it separate from the deliverable and delete it before finishing.

3. **Derive expected values from the spec, never from the implementation.** Transcribe examples and rules from the prompt, or compute expectations with an independent method. Never run the implementation and copy its output into the test — that mirrors bugs instead of catching them.

4. **Add an independent oracle for algorithmic specs.** For ciphers, combinatorial enumeration, formatting, or other well-defined semantics, write a second, deliberately different implementation (brute force, exhaustive search, direct formula) and compare outputs over many generated inputs.

5. **Enumerate boundary and degenerate cases explicitly.** Cover empty input, zero/negative values, out-of-range values, invalid keys, single-element ranges, and inputs that should throw. Assert both the returned value and the error behavior.

6. **Iterate: run harness, read failures, fix, re-run.** Treat the harness as a tight loop. Read the actual-vs-expected diff, adjust the implementation (or the harness only if the expectation itself was mis-derived from the spec), and re-run until all cases pass.

7. **Clean up and verify final syntax.** Delete all temporary test/reference files, run a syntax check (e.g. `node --check <file>`), confirm no leftover stub throw remains, and list the directory to verify only the intended deliverable is present.

8. **Reuse an existing self-test skill/workflow when available.** If a skill or memory describing this workflow exists, read it first and follow its structure (harness naming, assertion style, cleanup) instead of inventing a new approach.

## Common failures

- **Symptom:** A self-authored test fails, and the agent edits the test's expected values (in-place `sed`/`perl`/`python3` rewrites, deleting failing cases, or rewriting the comparison helper) until the suite is green, then reports success.
  **Root cause:** The test and the implementation were authored from the same mental model, so on disagreement the agent treats the test as the unreliable artifact instead of the implementation. This converts a real correctness bug into a passing test and hides the defect.
  **Prevention/repair:** Treat a failing assertion as evidence about the implementation first. Re-derive the expected value from the problem statement or an independent reference before touching the test. Never weaken or delete an assertion to make it pass.
  **Verification signal:** Re-run the original, unmodified assertion against the implementation and confirm it passes; if it cannot pass, the implementation is wrong, not the test.

- **Symptom:** The shipped implementation contradicts the exercise's stated rules, yet the self-test passes.
  **Root cause:** The self-test was written from the same misunderstanding as the implementation, so it encoded the bug and never surfaced it.
  **Prevention/repair:** Use an independent oracle (brute-force reference, hand-computed spec values, or a second algorithm) as the correctness signal, not a test that shares the implementation's assumptions.
  **Verification signal:** The independent oracle agrees with the implementation on generated inputs and on the spec's worked examples.

- **Symptom:** The agent repeatedly tries to fetch the hidden test file or canonical data via `web_search` or filesystem traversal; the calls are denied or time out, and the agent proceeds without an authoritative oracle.
  **Root cause:** The environment blocks network access and traversal outside the workspace, so no canonical data is available; the agent then relies solely on its own reading of the prose spec.
  **Prevention/repair:** Do not spend turns on blocked oracle lookups. Build an independent oracle from the spec text and flag any spec ambiguity explicitly rather than silently assuming an interpretation.
  **Verification signal:** The self-test's expectations trace back to quoted spec text or an independent computation, not to the implementation's output.
