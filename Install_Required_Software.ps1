if (!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) 
{
    Start-Process powershell.exe -WindowStyle Maximized -Verb RunAs -ArgumentList ('-noprofile -file "{0}" -elevated' -f ($myinvocation.MyCommand.Definition))    
    exit
}

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new() 

function RestartScript()
{
    Write-Output "Restarting Script"
    Sleep 3
    Start-Process powershell.exe -WindowStyle Maximized -Verb RunAs -ArgumentList ('-noprofile -file "{0}" -elevated' -f ($myinvocation.MyCommand.Definition))    
    Exit    
}

function InstallIfNotInstalled  
{
    param ([string]$app)

    if ( ((winget list $app) -match '^\p{L}' | Measure-Object -Line).Lines -lt 2 ) 
    {
        winget install $app --scope machine
    }
}


winget list --accept-source-agreements | Out-Null

InstallIfNotInstalled("Python.Python.3.11")
InstallIfNotInstalled("Kitware.CMake")

python.exe -m pip install --upgrade pip
pip install opencv-python
pip install cmake

pip list --outdated | Select-Object -Skip 2 | ForEach-Object { $_.Split(' ')[0] } | ForEach-Object { pip install --upgrade $_ }

try
{
    cmake --version
}
catch
{
    RestartScript
}

# Step 2: Install Visual Studio Build Tools
#winget install Microsoft.VisualStudio.2022.BuildTools --scope machine
winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools" --silent --accept-package-agreements --scope machine



# Step 3: Download and Install Boost Library
$boostUrl = "https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.zip"
$boostZip = "$env:TEMP\boost_1_76_0.zip"
$boostDir = "$env:ProgramFiles\boost_1_76_0"

Invoke-WebRequest -Uri $boostUrl -OutFile $boostZip
Expand-Archive -Path $boostZip -DestinationPath $boostDir

### Set BOOST_ROOT environment variable
[System.Environment]::SetEnvironmentVariable("BOOST_ROOT", 
$boostDir, [System.EnvironmentVariableTarget]::Machine)


# Step 4: Install dlib using pip
pip install dlib


pause




# Install Visual Studio Build Tools with Desktop development with C++ workload
winget install Microsoft.VisualStudio.2022.BuildTools "--add Microsoft.VisualStudio.Workload.VCTools"

winget search Microsoft.VisualStudio.2022.BuildTools help

winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools"




