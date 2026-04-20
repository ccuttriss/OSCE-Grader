# Server deployment

This application can run on a Linux VM (systemd) or a container platform
(Docker / Kubernetes / OpenShift). The same artifact works on either —
only env vars differ.

## Env vars

See `.env.example` for the full list. **Server mode is the default** —
you only need to set `OSCE_SERVER_MODE=0` to opt into the local-dev
workstation path (auto-admin, key-file fallback, auto-generated master
key). Minimum required for a working server-mode deployment:

- `OSCE_DATA_DIR=/var/lib/osce-grader` (VM) or `/data` (container)
- `OSCE_ADMIN_EMAILS=comma,separated,list` — emails allowed to reach the
  admin tabs. Without this, no one can reach Models & Keys.
- `OSCE_SECRET_KEY=<Fernet key>` — base64-encoded 32-byte key used to
  encrypt API keys at rest. Generate with
  `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.
  Without it, the admin UI warns and stored keys remain in plaintext.
- At least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`
  (optional — admins can also add keys at runtime via the Models & Keys tab).

## Linux VM (systemd)

```ini
# /etc/systemd/system/osce-grader.service
[Unit]
Description=OSCE Grader
After=network.target

[Service]
User=osce
Group=osce
EnvironmentFile=/etc/osce-grader.env
WorkingDirectory=/opt/osce-grader
ExecStart=/opt/osce-grader/venv/bin/streamlit run app.py \
    --server.port=8501 --server.address=127.0.0.1
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Put a reverse proxy (NGINX / Apache) in front of `127.0.0.1:8501` for TLS
termination. Audit data lives at `$OSCE_DATA_DIR/osce_grader.db`; back it up.

## Container

```bash
docker build -t osce-grader .
docker run -d \
  --name osce-grader \
  -p 8501:8501 \
  --env-file /etc/osce-grader.env \
  -v /srv/osce-grader-data:/data \
  osce-grader
```

## Backup and retention

- Database: nightly `sqlite3 osce_grader.db .backup backup.db`.
- Materials: `$OSCE_STORAGE_DIR/materials` — rsync / object-store sync.
- Retention sweeps run automatically in server mode (startup + daily).
