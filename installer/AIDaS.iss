; AIDaS per-user installer. Keep AppId, install mode, and architecture stable
; across every release so Inno Setup performs an in-place upgrade.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif
#ifndef MyAppFileVersion
  #define MyAppFileVersion "0.0.0.0"
#endif

#define MyAppName "AIDaS"
#define MyAppPublisher "Machine Vision and Pattern Recognition Lab, Wayne State University"
#define MyAppURL "https://github.com/Hejrati/AIDaS"
#define MyAppExeName "AIDaS.exe"

[Setup]
AppId={{5E514B02-7E97-4F86-8902-DC6EA73A7CB2}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Programs\AIDaS
DefaultGroupName=AIDaS
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\release
OutputBaseFilename=AIDaS-Setup-{#MyAppVersion}
SetupIconFile=..\assets\aidas.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
CloseApplicationsFilter={#MyAppExeName}
RestartApplications=no
AppMutex=Local\MVPRL.AIDaS.DesktopApplication
UsePreviousAppDir=yes
UsePreviousGroup=yes
Uninstallable=yes
VersionInfoVersion={#MyAppFileVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription=AIDaS installer
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppFileVersion}
VersionInfoProductTextVersion={#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; AIDaS.exe is private to this application. ignoreversion deliberately replaces
; the prior executable while leaving every user-created file and directory alone.
Source: "..\dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\AIDaS"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\AIDaS"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
; Interactive first installs show the normal launch checkbox. Updates started by
; AIDaS are silent and use the second entry to reopen only after files are ready.
Filename: "{app}\{#MyAppExeName}"; Description: "Launch AIDaS"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent; Check: not IsAppUpdate
Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Flags: nowait; Check: IsAppUpdate

[Code]
function IsAppUpdate: Boolean;
begin
  Result := ExpandConstant('{param:UPDATEFROMAPP|0}') = '1';
end;
