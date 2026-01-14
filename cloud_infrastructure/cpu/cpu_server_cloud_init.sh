#!/usr/bin/env bash
set -euo pipefail

# Generate download_pdb.sh
cat > ~/download_pdb.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

apt update -y
apt install -y openjdk-8-jdk s3cmd zip

# Variables
PDB_PATH="pdb_mirror"
ARCHIVE_NAME="pdb_mirror.tar.gz"
VENV_DIR=".atomworks-venv"

# Install uv if not installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    source $HOME/.local/bin/env
fi

# Create venv with Python 3.12 if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python 3.12 venv in $VENV_DIR..."
    uv venv --python=3.12 "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install atomworks with extras
echo "Installing atomworks with uv..."
uv pip install "atomworks[ml,openbabel,dev]"

# Sync PDB
echo "Syncing PDB to $PDB_PATH ..."
atomworks pdb sync "$PDB_PATH"

# Compress folder
echo "Compressing $PDB_PATH to $ARCHIVE_NAME ..."
tar -czf "$ARCHIVE_NAME" -C "$(dirname "$PDB_PATH")" "$(basename "$PDB_PATH")"

echo "Done. Archive created: $ARCHIVE_NAME"
EOF

chmod +x ~/download_pdb.sh

# Generate upload_s3.sh
cat > ~/upload_s3.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Prompt for credentials
read -rp "Enter your access key: " ACCESS_KEY
read -rsp "Enter your secret key: " SECRET_KEY
echo ""
read -rp "Enter your bucket name: " BUCKET_NAME
read -rp "Enter destination folder in bucket (e.g. pdb_backups): " DEST_FOLDER
read -rp "Enter archive file to upload: " BACKUP_FILE

# Create s3cfg
cat > ~/.s3cfg <<EOCFG
[default]
access_key = $ACCESS_KEY
secret_key = $SECRET_KEY
use_https = True
host_base = tor1.digitaloceanspaces.com
host_bucket = %(bucket)s.tor1.digitaloceanspaces.com
EOCFG

# Upload
echo "Uploading $BACKUP_FILE to s3://$BUCKET_NAME/$DEST_FOLDER/ ..."
s3cmd put "$BACKUP_FILE" "s3://$BUCKET_NAME/$DEST_FOLDER/"

echo "Upload complete."
EOF

chmod +x ~/upload_s3.sh

echo "Scripts created: download_pdb.sh and upload_s3.sh"