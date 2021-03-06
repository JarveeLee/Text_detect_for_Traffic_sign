@echo off
set img_base_dir="C:\qxr\face_a"
for /f "delims=" %%i in (%img_base_dir%) do (
    set route=%%i
    set filename=
    call :separate
)
pause
goto :eof

:separate
if not "%route:~-1%"=="\" (
    set filename=%route:~-1%%filename%
    set route=%route:~0,-1%
    goto separate
    ) 
echo %filename% 
if not exist %filename%.xml (imglab.exe -c %filename%.xml %img_base_dir%) 
imglab.exe %filename%.xml
goto :eof
