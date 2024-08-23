if (!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) 
{
    Start-Process powershell.exe -WindowStyle Maximized -Verb RunAs -ArgumentList ('-noprofile -file "{0}" -elevated' -f ($myinvocation.MyCommand.Definition))    
    exit
}

function RestartScript()
{
    Write-Output "Restarting Script"
    Sleep 3
    Start-Process powershell.exe -WindowStyle Maximized -Verb RunAs -ArgumentList ('-noprofile -file "{0}" -elevated' -f ($myinvocation.MyCommand.Definition))    
    Exit    
}

winget list --accept-source-agreements | Out-Null

#windows prompt:

winget install Python.Python.3.11 --scope machine
winget install Kitware.CMake --scope machine
python.exe -m pip install --upgrade pip
pip list --outdated | Select-Object -Skip 2 | ForEach-Object { $_.Split(' ')[0] } | ForEach-Object { pip install --upgrade $_ }

try
{
    cmake
}
catch
{
    RestartScript
}

pip install opencv-python dlib

pause