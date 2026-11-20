# Run in the project root: powershell -ExecutionPolicy Bypass -File .\VERIFY_WINDOWS.ps1
$ErrorActionPreference = 'Stop'
Write-Host '=== Python tests (MuJoCo 4.0 integration test should run here) ==='
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed. Stop and inspect the traceback.' }
Write-Host '=== First viewpoint with gravity ==='
python robot_simulation.py --cyberknife --cyberknife-nodes 1
if ($LASTEXITCODE -ne 0) { throw 'First node failed. Inspect results\cyberknife_motion\multinode_summary.json.' }
Write-Host '=== All seven viewpoints with gravity ==='
python robot_simulation.py --cyberknife
if ($LASTEXITCODE -ne 0) { throw 'Multi-node run failed. Do not claim success.' }
Write-Host '=== All checks passed: visualization is optional ==='
Write-Host 'Run: python robot_simulation.py --cyberknife-view'
