@echo off
setlocal

SET conda=C:\Users\%username%\anaconda3\Scripts\activate.bat 
SET CODE="C:\Users\%username%\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd"
SET PATH=.
start cmd /k "cd /d %~dp0 && %conda% activate tensorflowenv310 && %CODE% %PATH% && exit"
exit