# prereg42: start run_chain_4090.sh as an independent hidden process (survives the Claude session).
#   powershell -ExecutionPolicy Bypass -File results\prereg42\explore\launch_4090.ps1 V2 V4 V3
# ASCII only (Windows PowerShell 5.1 reads scripts without a BOM as ANSI). Checks nvidia-smi first: driver 616.92 and
# no other compute process; otherwise it does not start. Log: data\prereg42\chain_log.txt.
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$bash = "C:\Program Files\Git\bin\bash.exe"
if ($args.Count -lt 1) { Write-Output "usage: launch_4090.ps1 V2 V4 [V3]"; exit 2 }
$drv = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader).Trim()
if ($drv -ne "616.92") { Write-Output "driver is $drv, not 616.92: not starting"; exit 3 }
$apps = (& nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader) | Where-Object { $_ -and $_.Trim() }
if ($apps) { Write-Output "another compute process is on the GPU: $apps"; exit 3 }
$script = Join-Path $PSScriptRoot "run_chain_4090.sh"
$argline = '"' + $script.Replace('\', '/') + '" ' + ($args -join ' ')
$p = Start-Process -FilePath $bash -ArgumentList $argline -WorkingDirectory $repo -WindowStyle Hidden -PassThru
Write-Output ("started pid " + $p.Id + ": " + $argline)
