#!powershell
#
# powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
#
# gcloud auth application-default login

$ErrorActionPreference = "Stop"

function checkEnv() {
    $script:ARCH = $Env:PROCESSOR_ARCHITECTURE.ToLower()
    $script:TARGET_ARCH=$Env:PROCESSOR_ARCHITECTURE.ToLower()
    Write-host "Building for ${script:TARGET_ARCH}"
    write-host "Locating required tools and paths"
    $script:SRC_DIR=$PWD
    if (!$env:VCToolsRedistDir) {
        $MSVC_INSTALL=(Get-CimInstance MSFT_VSInstance -Namespace root/cimv2/vs)[0].InstallLocation
        $env:VCToolsRedistDir=(get-item "${MSVC_INSTALL}\VC\Redist\MSVC\*")[0]
    }
    # Locate CUDA versions
    # Note: this assumes every version found will be built
    $cudaList=(get-item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\" -ea 'silentlycontinue')
    if ($cudaList.length -eq 0) {
        $d=(get-command -ea 'silentlycontinue' nvcc).path
        if ($null -ne $d) {
            $script:CUDA_DIRS=@($d| split-path -parent)
        }
    } else {
        $script:CUDA_DIRS=$cudaList
    }
    
    $script:INNO_SETUP_DIR=(get-item "C:\Program Files*\Inno Setup*\")[0]

    $script:DEPS_DIR="${script:SRC_DIR}\dist\windows-${script:TARGET_ARCH}"
    $env:CGO_ENABLED="1"
    Write-Output "Checking version"
    if (!$env:VERSION) {
        $data=(git describe --tags --first-parent --abbrev=7 --long --dirty --always)
        $pattern="v(.+)"
        if ($data -match $pattern) {
            $script:VERSION=$matches[1]
        }
    } else {
        $script:VERSION=$env:VERSION
    }
    $pattern = "(\d+[.]\d+[.]\d+).*"
    if ($script:VERSION -match $pattern) {
        $script:PKG_VERSION=$matches[1]
    } else {
        $script:PKG_VERSION="0.0.0"
    }
    write-host "Building Ollama $script:VERSION with package version $script:PKG_VERSION"

    # Note: Windows Kits 10 signtool crashes with GCP's plugin
    if ($null -eq $env:SIGN_TOOL) {
        ${script:SignTool}="C:\Program Files (x86)\Windows Kits\8.1\bin\x64\signtool.exe"
    } else {
        ${script:SignTool}=${env:SIGN_TOOL}
    }
    if ("${env:KEY_CONTAINER}") {
        ${script:OLLAMA_CERT}=$(resolve-path "${script:SRC_DIR}\ollama_inc.crt")
        Write-host "Code signing enabled"
    } else {
        write-host "Code signing disabled - please set KEY_CONTAINERS to sign and copy ollama_inc.crt to the top of the source tree"
    }

}


function buildOllama() {
    if ($null -eq ${env:OLLAMA_SKIP_GENERATE}) {
        write-host "Building ollama runners"
        Remove-Item -ea 0 -recurse -force -path "${script:SRC_DIR}\dist\windows-${script:ARCH}"
        if ($null -eq ${env:OLLAMA_NEW_RUNNERS}) {
            # Start by skipping CUDA to build everything else
            pwsh -Command { $env:OLLAMA_SKIP_CUDA_GENERATE="1"; & go generate ./... }
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}    

            # Then skip everyhting else and build all the CUDA variants
            foreach ($env:CUDA_LIB_DIR in $script:CUDA_DIRS) {
                write-host "Building CUDA ${env:CUDA_LIB_DIR}"

                if ($env:CUDA_LIB_DIR.Contains("v12")) {
                    pwsh -Command {
                        $env:OLLAMA_SKIP_CUDA_GENERATE=""
                        $env:OLLAMA_SKIP_STATIC_GENERATE="1"
                        $env:OLLAMA_SKIP_CPU_GENERATE="1"
                        $env:OLLAMA_SKIP_ONEAPI_GENERATE="1"
                        $env:OLLAMA_SKIP_ROCM_GENERATE="1"
                        $env:CMAKE_CUDA_ARCHITECTURES="60;61;62;70;72;75;80;86;87;89;90;90a"
                        $env:OLLAMA_CUSTOM_CUDA_DEFS="-DGGML_CUDA_USE_GRAPHS=on"
                        $env:CUDA_PATH=split-path -path $env:CUDA_LIB_DIR -parent
                        $env:PATH="$envs:CUDA_LIB_DIR;$env:PATH"
                        & go generate ./...
                    }
                } else {
                    pwsh -Command {
                        $env:OLLAMA_SKIP_CUDA_GENERATE=""
                        $env:OLLAMA_SKIP_STATIC_GENERATE="1"
                        $env:OLLAMA_SKIP_CPU_GENERATE="1"
                        $env:OLLAMA_SKIP_ONEAPI_GENERATE="1"
                        $env:OLLAMA_SKIP_ROCM_GENERATE="1"
                        $env:CMAKE_CUDA_ARCHITECTURES="50;52;53;60;61;62;70;72;75;80;86"
                        $env:OLLAMA_CUSTOM_CUDA_DEFS=""
                        $env:CUDA_PATH=split-path -path $env:CUDA_LIB_DIR -parent
                        $env:PATH="$envs:CUDA_LIB_DIR;$env:PATH"
                        & go generate ./...
                    }
                }
                if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
            }
        } else {
            & make -C llama -j 12
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        }
        
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}    
    } else {
        write-host "Skipping generate step with OLLAMA_SKIP_GENERATE set"
    }
    write-host "Building ollama CLI"
    & go build -trimpath -ldflags "-s -w -X=github.com/ollama/ollama/version.Version=$script:VERSION -X=github.com/ollama/ollama/server.mode=release" .
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    if ("${env:KEY_CONTAINER}") {
        & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
            /csp "Google Cloud KMS Provider" /kc ${env:KEY_CONTAINER} ollama.exe
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    }
    New-Item -ItemType Directory -Path .\dist\windows-${script:TARGET_ARCH}\bin\ -Force
    cp .\ollama.exe .\dist\windows-${script:TARGET_ARCH}\bin\
}

function buildApp() {
    write-host "Building Ollama App"
    cd "${script:SRC_DIR}\app"
    & windres -l 0 -o ollama.syso ollama.rc
    & go build -trimpath -ldflags "-s -w -H windowsgui -X=github.com/ollama/ollama/version.Version=$script:VERSION -X=github.com/ollama/ollama/server.mode=release" .
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    if ("${env:KEY_CONTAINER}") {
        & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
            /csp "Google Cloud KMS Provider" /kc ${env:KEY_CONTAINER} app.exe
        if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
    }
}

function gatherDependencies() {
    write-host "Gathering runtime dependencies"
    cd "${script:SRC_DIR}"
    md "${script:DEPS_DIR}\lib\ollama" -ea 0 > $null

    # TODO - this varies based on host build system and MSVC version - drive from dumpbin output
    # currently works for Win11 + MSVC 2019 + Cuda V11
    cp "${env:VCToolsRedistDir}\x64\Microsoft.VC*.CRT\msvcp140*.dll" "${script:DEPS_DIR}\lib\ollama\"
    cp "${env:VCToolsRedistDir}\x64\Microsoft.VC*.CRT\vcruntime140.dll" "${script:DEPS_DIR}\lib\ollama\"
    cp "${env:VCToolsRedistDir}\x64\Microsoft.VC*.CRT\vcruntime140_1.dll" "${script:DEPS_DIR}\lib\ollama\"
    foreach ($part in $("runtime", "stdio", "filesystem", "math", "convert", "heap", "string", "time", "locale", "environment")) {
        cp "$env:VCToolsRedistDir\..\..\..\Tools\Llvm\x64\bin\api-ms-win-crt-${part}*.dll" "${script:DEPS_DIR}\lib\ollama\"
    }


    cp "${script:SRC_DIR}\app\ollama_welcome.ps1" "${script:SRC_DIR}\dist\"
    if ("${env:KEY_CONTAINER}") {
        write-host "about to sign"
        foreach ($file in (get-childitem "${script:DEPS_DIR}\lib\ollama\cu*.dll") + @("${script:SRC_DIR}\dist\ollama_welcome.ps1")){
            write-host "signing $file"
            & "${script:SignTool}" sign /v /fd sha256 /t http://timestamp.digicert.com /f "${script:OLLAMA_CERT}" `
                /csp "Google Cloud KMS Provider" /kc ${env:KEY_CONTAINER} $file
            if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
        }
    }
}

function buildInstaller() {
    write-host "Building Ollama Installer"
    cd "${script:SRC_DIR}\app"
    $env:PKG_VERSION=$script:PKG_VERSION
    if ("${env:KEY_CONTAINER}") {
        & "${script:INNO_SETUP_DIR}\ISCC.exe" /DARCH=$script:TARGET_ARCH /SMySignTool="${script:SignTool} sign /fd sha256 /t http://timestamp.digicert.com /f ${script:OLLAMA_CERT} /csp `$qGoogle Cloud KMS Provider`$q /kc ${env:KEY_CONTAINER} `$f" .\ollama.iss
    } else {
        & "${script:INNO_SETUP_DIR}\ISCC.exe" /DARCH=$script:TARGET_ARCH .\ollama.iss
    }
    if ($LASTEXITCODE -ne 0) { exit($LASTEXITCODE)}
}

function distZip() {
    write-host "Generating stand-alone distribution zip file ${script:SRC_DIR}\dist\ollama-windows-${script:TARGET_ARCH}.zip"
    Compress-Archive -Path "${script:SRC_DIR}\dist\windows-${script:TARGET_ARCH}\*" -DestinationPath "${script:SRC_DIR}\dist\ollama-windows-${script:TARGET_ARCH}.zip" -Force
}

try {
    checkEnv
    buildOllama
    buildApp
    gatherDependencies
    buildInstaller
    distZip
} catch {
    write-host "Build Failed"
    write-host $_
} finally {
    set-location $script:SRC_DIR
    $env:PKG_VERSION=""
}
