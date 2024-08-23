if (!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) 
{
    Start-Process powershell.exe -WindowStyle Maximized -Verb RunAs -ArgumentList ('-noprofile -file "{0}" -elevated' -f ($myinvocation.MyCommand.Definition))    
    exit
}

winget list --accept-source-agreements | Out-Null

#windows prompt:

winget install Python.Python.3.9 --scope machine
winget install Kitware.CMake --scope machine

pip install opencv-python dlib
