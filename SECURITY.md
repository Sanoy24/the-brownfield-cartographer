# Security policy

## Supported versions

Brownfield Cartographer is pre-1.0. Security fixes are made on `main` and
included in the next release; older releases are not patched.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/Sanoy24/the-brownfield-cartographer/security/advisories/new)
for this repository. Include the affected version or commit, the steps or a
minimal repository that reproduces the problem, and the impact you observed.

You should receive an acknowledgement within 7 days. Please allow time for a
fix to be released before disclosing details publicly.

## Scope

Cartographer reads repositories that may be untrusted. The following are in
scope:

- Reading, following, or reporting files outside the analyzed repository root
  (path traversal, symlinks, junctions).
- Executing code from an analyzed repository. Local analysis must never do so.
- Disclosing excluded files (for example `.env` or key files) through artifacts,
  CLI output, or MCP tools.
- Resource exhaustion that bypasses the documented scanner limits.
- Injection through analyzed content into generated artifacts or MCP responses.
- Any network access or source upload during local analysis.

Out of scope: findings that require an attacker to already control the local
machine, secrets embedded in ordinary source files (which are not detected; see
the README), and issues limited to the optional legacy `llm` extra's upstream
model providers.
