# TopoSegNet queue watchdog.
#
# Why: desktop-app auto-updates killed the training queue four times in a
# week (2026-10-02..08), because anything launched from an IDE session lives in
# the app's process tree. This script is run hourly by a Windows scheduled task
# (Task Scheduler, outside the IDE app), so app updates cannot kill it.
#
# Each run, in order:
#   1. experiments\_logs\PAUSE exists          -> do nothing (you stopped it on purpose)
#   2. a python.exe running run_queue.py or
#      train.py already exists                 -> do nothing (healthy)
#   3. newest queue log ends "QUEUE COMPLETE"  -> do nothing (finished)
#   4. otherwise                               -> start run_queue.py, which resumes
#                                                 from the last saved checkpoint
# run_queue.py also takes an OS-level lock, so even a race cannot start two copies.
#
# Remove:  schtasks /delete /tn TopoSegNet-watchdog /f
# Pause:   create an empty file experiments\_logs\PAUSE   (delete it to resume)

$Root = 'C:\Users\prash\toposegnet'
$Logs = Join-Path $Root 'experiments\_logs'
$Py   = 'C:\Users\prash\anaconda3\envs\toposegnet\python.exe'
$WdLog = Join-Path $Logs 'watchdog.log'

function Log($msg) { Add-Content -Path $WdLog -Value ("[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg) }

if (Test-Path (Join-Path $Logs 'PAUSE')) { Log 'PAUSE file present - not touching the queue'; exit 0 }

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -match 'run_queue\.py|train\.py' }
if ($running) { Log ("healthy - running PIDs: " + (($running | ForEach-Object ProcessId) -join ', ')); exit 0 }

$latest = Get-ChildItem (Join-Path $Logs 'queue_*.out.log') -ErrorAction SilentlyContinue |
          Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest -and ((Get-Content $latest.FullName -Tail 3) -match 'QUEUE COMPLETE')) { Log 'queue complete - nothing to do'; exit 0 }

$ts  = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $Logs "queue_resume_wd_$ts.out.log"
$err = Join-Path $Logs "queue_resume_wd_$ts.err.log"
$p = Start-Process -FilePath $Py -ArgumentList '-u','run_queue.py' -WorkingDirectory $Root `
     -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
Log ("queue was DOWN - restarted run_queue.py as PID {0}; log {1}" -f $p.Id, (Split-Path $out -Leaf))
