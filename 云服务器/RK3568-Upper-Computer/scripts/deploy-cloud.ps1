param(
    [ValidateSet("Canary", "Upload", "Activate")]
    [string]$Action = "Canary",
    [string]$CloudHost = "203.0.113.10",
    [string]$CloudUser = "root",
    [string]$IdentityFile = "runtime/mediamtx_ed25519",
    [string]$KnownHostsFile = "runtime/mediamtx_known_hosts",
    [string]$RenderedDir = "runtime/rendered/cloud",
    [ValidateSet("staging", "production")]
    [string]$CertificateMode = "staging",
    [switch]$ConfirmProduction
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$identity = (Resolve-Path (Join-Path $repo $IdentityFile)).Path
$knownHosts = (Resolve-Path (Join-Path $repo $KnownHostsFile)).Path
$rendered = (Resolve-Path (Join-Path $repo $RenderedDir)).Path
$target = "$CloudUser@$CloudHost"
$sshOptions = @(
    "-i", $identity,
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "UserKnownHostsFile=$knownHosts",
    "-o", "ConnectTimeout=8"
)

function Invoke-Ssh {
    param([Parameter(Mandatory)][string]$Command)
    & ssh @sshOptions $target $Command
    if ($LASTEXITCODE -ne 0) {
        throw "remote command failed with exit code $LASTEXITCODE"
    }
}

function Copy-Remote {
    param(
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$Destination
    )
    & scp @sshOptions $Source "${target}:$Destination"
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed with exit code $LASTEXITCODE"
    }
}

$required = @(
    "mediamtx.yml",
    "nginx.conf",
    "turnserver.conf",
    "rk3568-cloud.service",
    "rk3568-ip-cert.service",
    "rk3568-ip-cert.timer",
    "rk3568-renew-ip-cert",
    "cloud.env",
    "cert.env"
)
foreach ($name in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $rendered $name))) {
        throw "missing rendered file: $name; run scripts/render-deployment.py first"
    }
}

$sha = (& git -C $repo rev-parse --short=12 HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or -not $sha) {
    throw "unable to resolve Git commit"
}

if ($Action -eq "Canary") {
    $canary = Join-Path $repo "runtime/mediamtx-canary.yml"
    $text = Get-Content -Raw -Encoding utf8 (
        Join-Path $rendered "mediamtx.yml"
    )
    $text = $text.Replace("127.0.0.1:9997", "127.0.0.1:19997")
    $text = $text.Replace("127.0.0.1:9998", "127.0.0.1:19998")
    $text = $text.Replace("127.0.0.1:8888", "127.0.0.1:18888")
    $text = $text.Replace("127.0.0.1:8889", "127.0.0.1:18889")
    $text = $text.Replace(":49199", ":49399")
    $text = $text.Replace(":49200", ":49400")
    $text = $text.Replace(":8890", ":18890")
    [System.IO.File]::WriteAllText(
        $canary,
        $text,
        (New-Object System.Text.UTF8Encoding($false))
    )
    try {
        Copy-Remote $canary "/tmp/rk3568-mediamtx-canary.yml"
        Invoke-Ssh @'
set -eu
log=/tmp/rk3568-mediamtx-canary.log
rm -f "$log"
set +e
timeout 6 /usr/local/bin/mediamtx /tmp/rk3568-mediamtx-canary.yml >"$log" 2>&1
code=$?
set -e
if [ "$code" -ne 124 ] && [ "$code" -ne 143 ]; then
  sed -n '1,120p' "$log"
  exit "$code"
fi
grep -q 'started with listener' "$log"
rm -f /tmp/rk3568-mediamtx-canary.yml "$log"
'@
    } finally {
        Remove-Item -LiteralPath $canary -Force -ErrorAction SilentlyContinue
    }
    Write-Host "MediaMTX continuity_canary configuration passed."
    exit 0
}

$runtimeDir = Join-Path $repo "runtime"
$sourceArchive = Join-Path $runtimeDir "rk3568-manager-$sha.tar.gz"
$frontendArchive = Join-Path $runtimeDir "rk3568-frontend-$sha.tar.gz"
if ($Action -in @("Upload", "Activate")) {
    if (-not (Test-Path (Join-Path $repo "frontend/dist/index.html"))) {
        throw "frontend/dist is missing; run npm run build --prefix frontend"
    }
    & git -C $repo archive --format=tar.gz -o $sourceArchive HEAD
    if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
    & tar -czf $frontendArchive -C (Join-Path $repo "frontend/dist") .
    if ($LASTEXITCODE -ne 0) { throw "frontend archive failed" }
    Invoke-Ssh "install -d -m 0700 /var/tmp/rk3568-deploy/$sha"
    Copy-Remote $sourceArchive "/var/tmp/rk3568-deploy/$sha/source.tar.gz"
    Copy-Remote $frontendArchive "/var/tmp/rk3568-deploy/$sha/frontend.tar.gz"
    foreach ($name in $required) {
        Copy-Remote (
            Join-Path $rendered $name
        ) "/var/tmp/rk3568-deploy/$sha/$name"
    }
    Write-Host "Uploaded Git commit $sha and protected rendered configuration."
}

if ($Action -eq "Upload") {
    exit 0
}

if (-not $ConfirmProduction) {
    throw "Activate requires -ConfirmProduction after Canary succeeds"
}

$stagingFlag = if ($CertificateMode -eq "staging") {
    "--staging"
} else {
    "--force-renewal"
}
$activateScript = @'
set -eu
sha='__SHA__'
stage="/var/tmp/rk3568-deploy/$sha"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup="/var/backups/rk3568-manager/$stamp"
install -d -m 0700 "$backup"
for path in \
  /etc/mediamtx.yml \
  /etc/turnserver.conf \
  /etc/nginx/sites-available/rk3568-manager \
  /etc/systemd/system/rk3568-cloud.service \
  /etc/systemd/system/rk3568-ip-cert.service \
  /etc/systemd/system/rk3568-ip-cert.timer
do
  if [ -e "$path" ]; then
    cp -a "$path" "$backup/"
  fi
done
if [ -L /opt/rk3568-manager/current ]; then
  readlink /opt/rk3568-manager/current >"$backup/previous-release"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  nginx python3-venv python3-pip ca-certificates openssl

if ! getent passwd rkmanager >/dev/null; then
  useradd --system --home /var/lib/rk3568-manager \
    --shell /usr/sbin/nologin rkmanager
fi
install -d -o rkmanager -g rkmanager -m 0750 \
  /var/lib/rk3568-manager /var/lib/rk3568-manager/events
install -d -m 0750 /etc/rk3568-manager
install -d -m 0755 /opt/rk3568-manager/releases

release="/opt/rk3568-manager/releases/$sha"
rm -rf "$release"
install -d -m 0755 "$release/frontend/dist"
tar -xzf "$stage/source.tar.gz" -C "$release"
tar -xzf "$stage/frontend.tar.gz" -C "$release/frontend/dist"
python3 -m venv "$release/.venv"
"$release/.venv/bin/pip" install --no-cache-dir \
  -r "$release/requirements-cloud.txt"
chown -R root:root "$release"
ln -sfn "$release" /opt/rk3568-manager/current

install -m 0600 "$stage/cloud.env" /etc/rk3568-manager/cloud.env
install -m 0600 "$stage/cert.env" /etc/rk3568-manager/cert.env
install -m 0640 -o root -g mediamtx \
  "$stage/mediamtx.yml" /etc/mediamtx.yml.next
install -m 0640 -o root -g turnserver \
  "$stage/turnserver.conf" /etc/turnserver.conf.next

install -d -m 0755 /var/www/acme
cat >/etc/nginx/sites-available/rk3568-bootstrap <<'BOOTSTRAP'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    location ^~ /.well-known/acme-challenge/ {
        root /var/www/acme;
    }
    location / {
        return 503;
    }
}
BOOTSTRAP
rm -f /etc/nginx/sites-enabled/default
rm -f /etc/nginx/sites-enabled/rk3568-manager
ln -sfn /etc/nginx/sites-available/rk3568-bootstrap \
  /etc/nginx/sites-enabled/rk3568-bootstrap
nginx -t
systemctl enable --now nginx
systemctl reload nginx

if [ ! -x /opt/certbot/bin/certbot ]; then
  python3 -m venv /opt/certbot
  /opt/certbot/bin/pip install --no-cache-dir 'certbot>=5.4,<6'
fi
set -a
. /etc/rk3568-manager/cert.env
set +a
/opt/certbot/bin/certbot certonly \
  --non-interactive \
  --agree-tos \
  --email "$ACME_EMAIL" \
  --preferred-profile shortlived \
  --cert-name "$PUBLIC_IP" \
  --webroot \
  --webroot-path /var/www/acme \
  --ip-address "$PUBLIC_IP" \
  __STAGING_FLAG__

install -d -m 0750 -o root -g turnserver /etc/turnserver/certs
install -m 0640 -o root -g turnserver \
  "/etc/letsencrypt/live/$PUBLIC_IP/fullchain.pem" \
  /etc/turnserver/certs/fullchain.pem
install -m 0640 -o root -g turnserver \
  "/etc/letsencrypt/live/$PUBLIC_IP/privkey.pem" \
  /etc/turnserver/certs/privkey.pem

install -m 0644 "$stage/nginx.conf" \
  /etc/nginx/sites-available/rk3568-manager
rm -f /etc/nginx/sites-enabled/rk3568-bootstrap
ln -sfn /etc/nginx/sites-available/rk3568-manager \
  /etc/nginx/sites-enabled/rk3568-manager
nginx -t

mv /etc/mediamtx.yml.next /etc/mediamtx.yml
mv /etc/turnserver.conf.next /etc/turnserver.conf
install -m 0644 "$stage/rk3568-cloud.service" \
  /etc/systemd/system/rk3568-cloud.service
install -m 0644 "$stage/rk3568-ip-cert.service" \
  /etc/systemd/system/rk3568-ip-cert.service
install -m 0644 "$stage/rk3568-ip-cert.timer" \
  /etc/systemd/system/rk3568-ip-cert.timer
install -m 0755 "$stage/rk3568-renew-ip-cert" \
  /usr/local/sbin/rk3568-renew-ip-cert

ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8890/udp
ufw allow 49199/udp
ufw allow 49200/tcp
ufw allow 3478/tcp
ufw allow 3478/udp
ufw allow 5349/tcp
ufw allow 49160:49190/udp

systemctl daemon-reload
systemctl enable mediamtx coturn rk3568-cloud rk3568-ip-cert.timer
systemctl restart mediamtx coturn rk3568-cloud nginx
systemctl start rk3568-ip-cert.timer
systemctl --no-pager --full status \
  mediamtx coturn rk3568-cloud nginx | sed -n '1,100p'
echo "backup=$backup"
'@
$activateScript = $activateScript.Replace("__SHA__", $sha)
$activateScript = $activateScript.Replace("__STAGING_FLAG__", $stagingFlag)
Invoke-Ssh $activateScript
Write-Host "Activated commit $sha with $CertificateMode IP certificate."
