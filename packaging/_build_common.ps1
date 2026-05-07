# Shared bundle builder (PowerShell). Not meant to be run directly — use one
# of the build_<platform>_<annotator_set>.bat wrappers, which set:
#
#   $env:PLATFORM        windows | unix
#   $env:ANNOTATOR_SET   single  | multi
#
# Output: packaging\dist\digital-registrar-annotator-<platform>-<set>.<ext>
#
# Host: Windows. Requires an activated dev venv with dspy installed (the host
# Python must be able to import digital_registrar_research.models.modellist and
# digital_registrar_research.schemas.pydantic._builder — precompute_section_groups.py
# uses those to precompute the section-groups map that runtime ships with).
#
# Windows bundles cross-install win_amd64 wheels. Unix bundles cross-install
# manylinux wheels into a python-build-standalone CPython (Linux x86_64).

$ErrorActionPreference = 'Stop'

# ── Wrapper contract ──────────────────────────────────────────────────────────
$Platform     = $env:PLATFORM
$AnnotatorSet = $env:ANNOTATOR_SET

if ([string]::IsNullOrEmpty($Platform)) {
    throw "PLATFORM env var must be set (windows|unix) by the wrapper."
}
if ([string]::IsNullOrEmpty($AnnotatorSet)) {
    throw "ANNOTATOR_SET env var must be set (single|multi) by the wrapper."
}
if ($Platform -notin @('windows','unix','macos')) {
    throw "unknown PLATFORM='$Platform' (expected windows|unix|macos)"
}
if ($AnnotatorSet -notin @('single','single_kpc','multi')) {
    throw "unknown ANNOTATOR_SET='$AnnotatorSet' (expected single|single_kpc|multi)"
}

# ── Pins ──────────────────────────────────────────────────────────────────────
$PythonVersion   = '3.12.8'
$PythonShort     = '3.12'
$PythonTag       = 'cp312'
$RuntimeReqs     = @('streamlit>=1.35,<2', 'pydantic>=2,<3')

# python-build-standalone pinned release (install_only tarballs).
# 20250115 is the release that ships cpython 3.12.8 for both Linux x86_64 and
# macOS arm64. The project moved from indygreg/ to astral-sh/ — both URLs work
# (GitHub redirects), but use the canonical org.
$PbsRelease      = '20250115'
$PbsLinuxTarball = "cpython-${PythonVersion}+${PbsRelease}-x86_64-unknown-linux-gnu-install_only.tar.gz"
$PbsMacArmTarball = "cpython-${PythonVersion}+${PbsRelease}-aarch64-apple-darwin-install_only.tar.gz"

# ── Paths ─────────────────────────────────────────────────────────────────────
$PkgDir     = Split-Path -Parent $PSCommandPath
$RepoRoot   = Split-Path -Parent $PkgDir
$DlDir      = Join-Path $PkgDir '_downloads'
$BuildDir   = Join-Path $PkgDir 'build'
$DistDir    = Join-Path $PkgDir 'dist'
$BundleName = "digital-registrar-annotator-${Platform}-${AnnotatorSet}"
$BundleDir  = Join-Path $BuildDir $BundleName

$PyHost = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

# ── Helpers ───────────────────────────────────────────────────────────────────
function Say($msg)    { Write-Host "==> $msg" -ForegroundColor Cyan }
function Die($msg)    { throw $msg }

function Assert-Exe($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Die "$name is required but not on PATH"
    }
}

function Invoke-Native {
    # Run a native exe; throw if its exit code is non-zero.
    # Usage: Invoke-Native <exe> <args...>
    $exe = $args[0]
    $rest = $args[1..($args.Length - 1)]
    & $exe @rest
    if ($LASTEXITCODE -ne 0) {
        Die "$exe exited with code $LASTEXITCODE"
    }
}

function Invoke-Robocopy {
    # Robocopy exit codes 0-7 are success; 8+ are real failures.
    param([string]$Src, [string]$Dst, [string[]]$ExtraArgs = @())
    $rcArgs = @($Src, $Dst, '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/R:1', '/W:1') + $ExtraArgs
    & robocopy @rcArgs | Out-Null
    if ($LASTEXITCODE -ge 8) {
        Die "robocopy $Src -> $Dst failed (exit $LASTEXITCODE)"
    }
    $script:LASTEXITCODE = 0
}

function Remove-PyCruft($root) {
    if (-not (Test-Path $root)) { return }
    Get-ChildItem -Path $root -Recurse -Directory -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('__pycache__', 'tests', 'test') } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $root -Recurse -File -Force -Filter '*.pyc' -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

# ── Preconditions ─────────────────────────────────────────────────────────────
Assert-Exe $PyHost
Assert-Exe 'curl.exe'
Assert-Exe 'tar.exe'

if (-not (Test-Path (Join-Path $RepoRoot 'src\digital_registrar_research\annotation'))) {
    Die 'annotation package not found — run from the repo root with dev venv active'
}
if (-not (Test-Path (Join-Path $RepoRoot 'dummy'))) {
    Die 'dummy/ not found; nothing to bundle'
}
if (-not (Test-Path (Join-Path $RepoRoot 'workspace'))) {
    Die 'workspace/ not found; create it first (Copy-Item -Recurse dummy workspace)'
}

# Host Python must import the dspy-backed modules (used by precompute_section_groups.py).
$pyProbe = @"
import digital_registrar_research.models.modellist
import digital_registrar_research.schemas.pydantic._builder
"@
& $PyHost -c $pyProbe
if ($LASTEXITCODE -ne 0) {
    Die 'host python cannot import dspy-backed modules — activate the dev venv first'
}

# ── Clean build dir (keep downloads cached) ───────────────────────────────────
Say "Preparing build tree for $BundleName"
if (Test-Path $BundleDir) { Remove-Item -Recurse -Force $BundleDir }
foreach ($d in @($DlDir, $DistDir, $BundleDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# ── Stage portable Python + install wheels (platform-specific) ────────────────
$PythonDir = Join-Path $BundleDir 'python'
New-Item -ItemType Directory -Path $PythonDir | Out-Null

if ($Platform -eq 'windows') {
    $embedUrl  = "https://www.python.org/ftp/python/${PythonVersion}/python-${PythonVersion}-embed-amd64.zip"
    $embedPath = Join-Path $DlDir "python-${PythonVersion}-embed-amd64.zip"

    if (-not (Test-Path $embedPath)) {
        Say "Downloading Windows embeddable Python $PythonVersion"
        Invoke-Native 'curl.exe' '-fLsS' '--retry' '3' '-o' "$embedPath.part" $embedUrl
        Move-Item "$embedPath.part" $embedPath
    } else {
        Say "Using cached $(Split-Path $embedPath -Leaf)"
    }

    Say 'Extracting embedded Python'
    Expand-Archive -Path $embedPath -DestinationPath $PythonDir -Force

    # Enable `import site` so pip-installed packages are picked up.
    $pthFile = Get-ChildItem -Path $PythonDir -Filter 'python*._pth' | Select-Object -First 1
    if (-not $pthFile) { Die 'python*._pth not found in embedded distribution' }
    Say "Patching $($pthFile.Name)"
    # Paths are relative to python.exe (python/python.exe), so ..\app points
    # to <bundle>/app where our slim digital_registrar_research package lives.
    # ._pth disables PYTHONPATH env var, so this must be listed here or the
    # package won't be importable at runtime.
    @"
python312.zip
.
..\app
Lib\site-packages
import site
"@ | Set-Content -Path $pthFile.FullName -Encoding ASCII

    $SitePackages = Join-Path $PythonDir 'Lib\site-packages'
    New-Item -ItemType Directory -Path $SitePackages -Force | Out-Null

    Say 'Installing Windows wheels into embedded site-packages'
    $pipArgs = @(
        '-m', 'pip', 'install',
        '--disable-pip-version-check',
        '--target', $SitePackages,
        '--platform', 'win_amd64',
        '--python-version', $PythonShort,
        '--implementation', 'cp',
        '--abi', $PythonTag,
        '--only-binary=:all:',
        '--upgrade'
    ) + $RuntimeReqs
    Invoke-Native $PyHost @pipArgs

} else {
    # Unix / macOS: download a python-build-standalone install_only tarball,
    # extract it (top-level `python/` directory), then cross-install wheels
    # for the target platform tag.
    if ($Platform -eq 'unix') {
        $pbsTarball  = $PbsLinuxTarball
        $pipPlatform = 'manylinux2014_x86_64'
        $wheelLabel  = 'Unix (manylinux)'
    } else {
        # macos = Apple Silicon (Mac mini M1/M2/M3+ etc.)
        $pbsTarball  = $PbsMacArmTarball
        $pipPlatform = 'macosx_11_0_arm64'
        $wheelLabel  = 'macOS arm64'
    }

    $pbsUrl  = "https://github.com/astral-sh/python-build-standalone/releases/download/${PbsRelease}/${pbsTarball}"
    $pbsPath = Join-Path $DlDir $pbsTarball

    if (-not (Test-Path $pbsPath)) {
        Say "Downloading python-build-standalone $PythonVersion ($PbsRelease) for $Platform"
        Invoke-Native 'curl.exe' '-fLsS' '--retry' '3' '-o' "$pbsPath.part" $pbsUrl
        Move-Item "$pbsPath.part" $pbsPath
    } else {
        Say "Using cached $(Split-Path $pbsPath -Leaf)"
    }

    Say 'Extracting python-build-standalone'
    # install_only tarball unpacks a top-level `python/` directory into $BundleDir.
    Invoke-Native 'tar.exe' '-xzf' $pbsPath '-C' $BundleDir
    if (-not (Test-Path (Join-Path $PythonDir 'bin\python3'))) {
        Die 'extracted python-build-standalone layout unexpected'
    }

    $SitePackages = Join-Path $PythonDir "lib\python${PythonShort}\site-packages"
    New-Item -ItemType Directory -Path $SitePackages -Force | Out-Null

    Say "Cross-installing $wheelLabel wheels into bundled site-packages"
    $pipArgs = @(
        '-m', 'pip', 'install',
        '--disable-pip-version-check',
        '--target', $SitePackages,
        '--platform', $pipPlatform,
        '--python-version', $PythonShort,
        '--implementation', 'cp',
        '--abi', $PythonTag,
        '--only-binary=:all:',
        '--upgrade'
    ) + $RuntimeReqs
    Invoke-Native $PyHost @pipArgs
}

# ── Trim size: drop tests/ and __pycache__ from site-packages ─────────────────
Say 'Trimming site-packages'
Remove-PyCruft $SitePackages
$binDir = Join-Path $SitePackages 'bin'
if (Test-Path $binDir) { Remove-Item -Recurse -Force $binDir }

# ── Assemble slim digital_registrar_research package ─────────────────────────
Say 'Copying annotation package'
$AppRoot = Join-Path $BundleDir 'app'
$DrrDest = Join-Path $AppRoot 'digital_registrar_research'
New-Item -ItemType Directory -Path $DrrDest -Force | Out-Null

$srcDrr = Join-Path $RepoRoot 'src\digital_registrar_research'
Copy-Item (Join-Path $srcDrr '__init__.py') (Join-Path $DrrDest '__init__.py')
Copy-Item (Join-Path $srcDrr 'paths.py')    (Join-Path $DrrDest 'paths.py')

# annotation subtree (robocopy with excludes)
$annotationDest = Join-Path $DrrDest 'annotation'
New-Item -ItemType Directory -Path $annotationDest -Force | Out-Null
Invoke-Robocopy (Join-Path $srcDrr 'annotation') $annotationDest @(
    '/XD', '__pycache__', 'dummy_data',
    '/XF', '*.pyc', 'generate_dummy_data.py', 'annotators.json'
)

# schemas/data only (runtime reads the JSON files)
$schemasDest = Join-Path $DrrDest 'schemas'
$schemasData = Join-Path $schemasDest 'data'
New-Item -ItemType Directory -Path $schemasData -Force | Out-Null
New-Item -ItemType File      -Path (Join-Path $schemasDest '__init__.py') -Force | Out-Null
Invoke-Robocopy (Join-Path $srcDrr 'schemas\data') $schemasData @('/XD','__pycache__','/XF','*.pyc')

Say 'Generating precomputed section-groups map'
Invoke-Native $PyHost (Join-Path $PkgDir 'precompute_section_groups.py') `
    (Join-Path $annotationDest '_section_groups.json')

# ── Write annotators.json baked to the chosen annotator set ──────────────────
Say "Baking annotators.json ($AnnotatorSet)"
$annotatorsJson = switch ($AnnotatorSet) {
    'single' {
@'
{
  "annotators": [
    {"name": "Nan-Haw Chow", "suffix": "nhc"}
  ]
}
'@
    }
    'single_kpc' {
@'
{
  "annotators": [
    {"name": "Kai-Po Chang", "suffix": "kpc"}
  ]
}
'@
    }
    'multi' {
@'
{
  "annotators": [
    {"name": "Nan-Haw Chow", "suffix": "nhc"},
    {"name": "Kai-Po Chang", "suffix": "kpc"}
  ]
}
'@
    }
}
# Write UTF-8 *without* BOM. PS 5.1's `Set-Content -Encoding UTF8` emits a BOM,
# which makes Python's json.loads() raise JSONDecodeError — the runtime loader
# would then silently fall back to DEFAULTS (which includes KPC), breaking the
# single-annotator lock. Use .NET directly to force no-BOM.
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    (Join-Path $annotationDest 'annotators.json'),
    $annotatorsJson,
    $utf8NoBom
)

# ── Copy dummy dataset + workspace skeleton ──────────────────────────────────
Say 'Copying dummy dataset'
$dummyDest = Join-Path $BundleDir 'dummy'
New-Item -ItemType Directory -Path $dummyDest -Force | Out-Null
Invoke-Robocopy (Join-Path $RepoRoot 'dummy') $dummyDest @(
    '/XD', 'results', '__pycache__',
    '/XF', '.DS_Store'
)

Say 'Copying workspace skeleton'
$workspaceDest = Join-Path $BundleDir 'workspace'
New-Item -ItemType Directory -Path $workspaceDest -Force | Out-Null
Invoke-Robocopy (Join-Path $RepoRoot 'workspace') $workspaceDest @(
    '/XD', 'results', '__pycache__',
    '/XF', '.DS_Store'
)

# ── Launchers + README ────────────────────────────────────────────────────────
Say 'Installing launchers'
if ($Platform -eq 'windows') {
    Copy-Item (Join-Path $PkgDir "run_workspace_locked_${AnnotatorSet}.bat") (Join-Path $BundleDir 'run.bat')
    Copy-Item (Join-Path $PkgDir "run_dummy_unlocked_${AnnotatorSet}.bat")   (Join-Path $BundleDir 'run_demo.bat')
} else {
    # macOS launchers also strip the Gatekeeper quarantine bit from the bundled
    # Python tree on first run; otherwise the unsigned python binary is blocked.
    $launcherSuffix = if ($Platform -eq 'macos') { '_macos' } else { '' }
    Copy-Item (Join-Path $PkgDir "run_workspace_locked_${AnnotatorSet}${launcherSuffix}.sh") (Join-Path $BundleDir 'run.sh')
    Copy-Item (Join-Path $PkgDir "run_dummy_unlocked_${AnnotatorSet}${launcherSuffix}.sh")   (Join-Path $BundleDir 'run_demo.sh')
    # Windows filesystems don't carry POSIX exec bits; tar will add them via --mode below.
}
Copy-Item (Join-Path $PkgDir 'README.txt') (Join-Path $BundleDir 'README.txt')

# ── Final pycache sweep ───────────────────────────────────────────────────────
Remove-PyCruft $BundleDir

$bundleSize = (Get-ChildItem $BundleDir -Recurse -File | Measure-Object Length -Sum).Sum
$bundleMiB  = "{0:N1} MiB" -f ($bundleSize / 1MB)
Say "Bundle tree assembled: $BundleDir ($bundleMiB)"

# ── Archive ───────────────────────────────────────────────────────────────────
if ($Platform -eq 'windows') {
    $archivePath = Join-Path $DistDir "$BundleName.zip"
    if (Test-Path $archivePath) { Remove-Item -Force $archivePath }
    Say 'Zipping'
    # Use tar.exe which ships with Windows 10+; its zip backend handles large trees
    # better than Compress-Archive and produces a standard ZIP.
    Push-Location $BuildDir
    try {
        Invoke-Native 'tar.exe' '-a' '-cf' $archivePath $BundleName
    } finally { Pop-Location }
} else {
    $archivePath = Join-Path $DistDir "$BundleName.tar.gz"
    if (Test-Path $archivePath) { Remove-Item -Force $archivePath }
    Say 'Tarring'
    # Windows' bsdtar doesn't support `--mode=...`, so we use Python's tarfile to
    # write the archive with explicit POSIX modes: 0755 for directories, .sh
    # launchers, and the bundled python binary tree (so `./run.sh` works and the
    # bundled python is executable on the target); 0644 for everything else.
    $tarScript = @"
import os, sys, tarfile
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
def fix(ti):
    ti.uid = 0; ti.gid = 0
    ti.uname = ''; ti.gname = ''
    needs_exec = (
        ti.isdir()
        or ti.name.endswith('.sh')
        or '/python/bin/' in (ti.name + '/')
    )
    ti.mode = 0o755 if needs_exec else 0o644
    return ti
os.chdir(src)
with tarfile.open(dst, 'w:gz') as tf:
    tf.add(name, recursive=True, filter=fix)
"@
    Invoke-Native $PyHost '-c' $tarScript $BuildDir $archivePath $BundleName
}
$archiveSize = (Get-Item $archivePath).Length
$archiveMiB  = "{0:N1} MiB" -f ($archiveSize / 1MB)

Say 'DONE'
Write-Host ("   bundle:  {0}  ({1} unpacked)" -f $archivePath, $bundleMiB)
Write-Host ("   archive: {0}" -f $archiveMiB)
Write-Host ''
if ($Platform -eq 'windows') {
    Write-Host "Next: copy $archivePath to the Windows machine, unzip, double-click run.bat."
} elseif ($Platform -eq 'macos') {
    Write-Host "Next: copy $archivePath to the Mac (Apple Silicon), 'tar -xzf' it, then in Terminal: cd <bundle> && ./run.sh"
} else {
    Write-Host "Next: copy $archivePath to the Linux machine, tar -xzf it, ./run.sh"
}
