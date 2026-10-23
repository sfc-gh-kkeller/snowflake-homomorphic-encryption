#!/usr/bin/env bash
# Pre-publication scan: fails if tracked/staged files contain key material, local config, or account identifiers.
#
#   scripts/scan_secrets.sh                 # pattern scan of files git would commit
#   SCAN_GITLEAKS=1 scripts/scan_secrets.sh # additionally run gitleaks (via Docker) on the repo
#
# Extra identifiers to forbid (account names, emails, connection names) can be passed in HE_FORBIDDEN as a
# '|'-separated regex, e.g. HE_FORBIDDEN='MYORG-MYACCOUNT|my_connection'.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0
files=$(git ls-files --cached --others --exclude-standard)

# 1. forbidden paths
bad_paths=$(echo "$files" | grep -E '(^|/)(keys/|config\.toml$|.*\.key$|.*\.sk$|\.pixi/)' || true)
if [ -n "$bad_paths" ]; then echo "FORBIDDEN PATHS:"; echo "$bad_paths"; fail=1; fi

# 2. secret-looking content (tokens, private keys, passwords)
patterns='(-----BEGIN [A-Z ]*PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|(password|passwd|token|secret)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"'$<{]{8,})'
hits=$(echo "$files" | grep -vE '^vendor/' | xargs grep -nIiE "$patterns" 2>/dev/null || true)
if [ -n "$hits" ]; then echo "SECRET-LIKE CONTENT:"; echo "$hits"; fail=1; fi

# 3. caller-supplied identifiers (account locators, org names, emails, connection names)
if [ -n "${HE_FORBIDDEN:-}" ]; then
  hits=$(echo "$files" | xargs grep -nIiE "$HE_FORBIDDEN" 2>/dev/null || true)
  if [ -n "$hits" ]; then echo "FORBIDDEN IDENTIFIERS:"; echo "$hits" | cut -c1-200; fail=1; fi
fi

# 4. optional gitleaks
if [ "${SCAN_GITLEAKS:-0}" = "1" ]; then
  docker run --rm -v "$PWD":/repo ghcr.io/gitleaks/gitleaks:latest dir /repo --no-banner --redact || fail=1
fi

if [ "$fail" = "0" ]; then echo "scan_secrets: clean ($(echo "$files" | wc -l | tr -d ' ') files)"; fi
exit $fail
