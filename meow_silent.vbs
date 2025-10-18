Set shell = CreateObject("WScript.Shell")
Set args = WScript.Arguments
If args.Count = 0 Then WScript.Quit 1

nekoFile = """" & args(0) & """"
cmd = "cmd.exe /c ""C:\Users\nykan\AppData\Local\Programs\meow\meow.bat"" " & nekoFile

' 0 = hidden window, True = wait until finished
shell.Run cmd, 0, True
