#!/usr/bin/env bash
# Push HEAD to GitHub with the repo-local PAT.
#
# The DSH sandbox cannot initialise any git TLS backend (schannel:
# SEC_E_NO_CREDENTIALS, openssl: reported unavailable), so this script only
# works when the caller elevates (sandbox_permissions=danger-full-access).
# The token is read from .agents/github_token (gitignored) and never written to
# git config or a remote.
set -euo pipefail
cd "$(dirname "$0")/.."
branch="${1:-master}"
token=$(tr -d '\r\n' < .agents/github_token)
if [ -z "$token" ]; then echo "empty .agents/github_token" >&2; exit 1; fi
case "$token" in
  https://*) url="$token" ;;
  *:*)       url="https://${token}@github.com/Stevenkerman27/Dpanel.git" ;;
  *)         url="https://x-access-token:${token}@github.com/Stevenkerman27/Dpanel.git" ;;
esac
git push "$url" "HEAD:${branch}"
