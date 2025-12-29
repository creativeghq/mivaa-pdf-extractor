# Restart MIVAA PDF Extractor service on production server

Write-Host "🔄 Restarting MIVAA PDF Extractor Service" -ForegroundColor Yellow
Write-Host "============================================================"
Write-Host ""

$server = "root@v1api.materialshub.gr"

Write-Host "📋 Service Information:" -ForegroundColor Cyan
Write-Host "   Server: $server" -ForegroundColor White
Write-Host "   Service: mivaa-pdf-extractor.service" -ForegroundColor White
Write-Host "   Path: /var/www/mivaa-pdf-extractor" -ForegroundColor White
Write-Host ""

Write-Host "🔄 Restarting service..." -ForegroundColor Yellow

ssh $server @"
echo '🛑 Stopping service...'
sudo systemctl stop mivaa-pdf-extractor

echo ''
echo '⏳ Waiting 2 seconds...'
sleep 2

echo ''
echo '🚀 Starting service...'
sudo systemctl start mivaa-pdf-extractor

echo ''
echo '✅ Checking service status...'
sudo systemctl status mivaa-pdf-extractor --no-pager | head -20

echo ''
echo '📋 Recent logs (last 30 lines):'
sudo journalctl -u mivaa-pdf-extractor -n 30 --no-pager
"@

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "✅ Service restarted successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Next Steps:" -ForegroundColor Yellow
    Write-Host "   1. Test the SigLIP2 endpoint" -ForegroundColor White
    Write-Host "   2. Verify 1152D embeddings are generated" -ForegroundColor White
    Write-Host "   3. Monitor logs for any errors" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Service restart failed" -ForegroundColor Red
    Write-Host "   Check the logs above for errors" -ForegroundColor Yellow
    exit 1
}

