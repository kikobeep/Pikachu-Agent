---
name: cpp-exercise-hidden-tests
description: Implementing C++ (Exercism-style) exercises where a provided header defines
  the public API consumed by hidden tests, in a sandboxed build environment.
---

# C++ Exercise Implementation Against Hidden Tests

Use this skill when implementing a C++ exercise (e.g. Exercism-style) where a
header file defines the public API that hidden tests compile against, and the
task asks you to implement a specific `.cpp` file.

## Procedure

1. **Reconstruct the expected public interface before writing code.**
   Read the provided header and the build file (`CMakeLists.txt`) first to
extract the exact namespace, class/function names, signatures, and return
types the hidden tests will call. Anchor all subsequent code to that contract.

2. **Treat an existing header as a frozen interface.**
   If the task says to implement only a named `.cpp` file and a header already
exists, read it and implement the `.cpp` against its exact declarations. Do
not overwrite it. Only author the header yourself when it is genuinely empty
or a minimal stub, and then match the canonical interface implied by the
exercise name and the stub's namespace.

3. **Write a standalone self-test harness.**
   Create a temporary `main`/self-test file that includes the header, calls the
public API with representative and boundary inputs, and asserts expected
results. Compile it together with the implementation and run it. Iterate until
it passes, then delete the harness.

4. **Compile the self-test against the original, unmodified header.**
   The harness must exercise the real expected API, not an interface you
authored. Verify the header was not modified before the test build.

5. **Use strict compiler flags.**
   Compile with `-std=c++17 -Wall -Wextra -Wpedantic -Werror` (add `-pthread`
when threading is involved). Treat any warning as a failure to fix.

6. **Enumerate and test boundary/edge cases explicitly.**
   Identify the boundary conditions implied by the problem statement (empty
input, zero/negative values, overflow/wrap-around, full/empty containers,
invalid input, duplicate entries) and write explicit assertions for each.

7. **Cross-check combinatorial logic against an independent reference.**
   For constraint-satisfaction or combinatorial problems, write a separate
brute-force program that enumerates all possibilities and counts solutions,
then compare its answer to the optimized implementation's answer.

8. **Clean up all temporary artifacts.**
   After self-tests pass, remove scratch sources, compiled binaries, and temp
directories so only the required implementation and header remain. Stray
files can interfere with the evaluator's build (duplicate `main`, unexpected
sources picked up by CMake).

## Common failures

- **Symptom:** The agent rewrote the provided header (e.g. `sublist.h`,
  `spiral_matrix.h`, `robot_name.h`, `clock.h`, `knapsack.h`) instead of only
  implementing the named `.cpp` file.
  **Root cause:** The agent assumed the header was a stub it needed to author,
  rather than a frozen interface supplied by the grader, and never verified
  whether the header content was already complete before overwriting it.
  **Prevention/repair:** Read the header first; if it already declares the
  interface, implement the `.cpp` against it and do not overwrite it. Only
  author the header when it is genuinely empty or a minimal stub.
  **Validation signal:** Confirm the header file is unchanged after the run
  (e.g. diff against the original) and that the `.cpp` compiles against the
  original header, not a rewritten one.

- **Symptom:** Self-tests all pass, yet the hidden test build would fail on a
  signature mismatch.
  **Root cause:** The self-test was compiled against an agent-authored header,
  so it only validated internal consistency and could not detect divergence
  from the grader's expected interface.
  **Prevention/repair:** Compile the self-test against the original, unmodified
  header so it exercises the real expected API.
  **Validation signal:** Verify the self-test includes the original header path
  and that the header was not modified before the test build.

- **Symptom:** `g++`/`clang++` fails with `couldn't create cache file ...
  Operation not permitted`, or discovery commands (`find`, `curl`) time out or
  are denied.
  **Root cause:** The sandbox blocks writes to the default `TMPDIR` and denies
  network/web tools, so the compiler wrapper cannot create its cache.
  **Prevention/repair:** Redirect `TMPDIR` (and `TMP`/`TEMP`) to a writable
  workspace-local directory and re-run; use bounded, non-recursive discovery
  commands. Do not conclude the code is broken from an environment/toolchain
  error.
  **Validation signal:** Re-run the same compile/run with `TMPDIR` set to a
  local writable path and confirm it succeeds, isolating environment failure
  from code failure.
