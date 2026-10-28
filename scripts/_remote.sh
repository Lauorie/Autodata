# Sourced by the sync scripts. Loads the run-box settings and defines
# remote_rsync. Settings come from $REMOTE_ENV_FILE, default
# .secrets/remote.env (see .env.example):
#   REMOTE_HOST, REMOTE_PORT, REMOTE_USER   the run box
#   REMOTE_PASSWORD                         optional; empty means key auth
#   REMOTE_REPO_DIR                         the repo copy on the box
#   REMOTE_OUTPUTS_DIR                      paths.outputs_root on the box
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REMOTE_ENV_FILE:-$ROOT/.secrets/remote.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "missing $ENV_FILE (copy .env.example)" >&2; exit 1
fi
set -a; . "$ENV_FILE"; set +a
: "${REMOTE_HOST:?set REMOTE_HOST in $ENV_FILE}"
: "${REMOTE_USER:?set REMOTE_USER in $ENV_FILE}"
: "${REMOTE_REPO_DIR:?set REMOTE_REPO_DIR in $ENV_FILE}"
: "${REMOTE_OUTPUTS_DIR:?set REMOTE_OUTPUTS_DIR in $ENV_FILE}"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
SSH_CMD="ssh -p ${REMOTE_PORT:-22} -o StrictHostKeyChecking=accept-new"

# sshpass reads the password from the environment (-e), not the command line.
remote_rsync() {
  if [ -n "${REMOTE_PASSWORD:-}" ]; then
    SSHPASS="$REMOTE_PASSWORD" sshpass -e rsync -e "$SSH_CMD" "$@"
  else
    rsync -e "$SSH_CMD" "$@"
  fi
}
