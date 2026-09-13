#ifndef PackageDir
  #error PackageDir must point to a clean portable release folder.
#endif

[Setup]
AppId={{A6A052C5-0C9D-485B-A228-5629AB44C79F}
AppName=WAVELY
AppVersion=0.1.1
AppPublisher=WAVELY
DefaultDirName={localappdata}\Programs\WAVELY
DefaultGroupName=WAVELY
UninstallDisplayIcon={app}\Wavely.exe
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
SetupIconFile={#PackageDir}\assets\wavely_icon.ico
OutputBaseFilename=WAVELY-Setup
Compression=lzma2
SolidCompression=yes
CloseApplications=yes
RestartApplications=no

[Files]
Source: "{#PackageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Icons]
Name: "{group}\WAVELY"; Filename: "{app}\Wavely.exe"
Name: "{autodesktop}\WAVELY"; Filename: "{app}\Wavely.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Wavely.exe"; Description: "Launch WAVELY"; Flags: nowait postinstall skipifsilent

[Code]
var
  RemoveUserData: Boolean;

function InitializeUninstall(): Boolean;
var
  Choice: Integer;
begin
  RemoveUserData := False;
  if UninstallSilent then
  begin
    Result := True;
    Exit;
  end;

  Choice := TaskDialogMsgBox(
    'Uninstall WAVELY',
    'Choose whether to keep your Wavely user data, or remove all Wavely user data and settings.' + #13#10 +
    'User data includes configuration, Home Assistant mappings, face images, trained models, cached state, backups, and logs.',
    mbConfirmation, MB_YESNOCANCEL, ['Uninstall and keep my data',
    'Remove all Wavely user data and settings', 'Cancel'], 0);
  case Choice of
    IDYES: Result := True;
    IDNO:
      begin
        RemoveUserData := True;
        Result := True;
      end;
  else
    Result := False;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;

  DeleteFile(ExpandConstant('{userstartup}\WAVELY Vision.cmd'));
  if RemoveUserData then
  begin
    DataDir := ExpandConstant('{localappdata}\Wavely');
    if DirExists(DataDir) and (not DelTree(DataDir, True, True, True)) then
      MsgBox('WAVELY was removed, but some personal data could not be deleted from ' +
        DataDir + '. Please close any WAVELY processes and remove that folder manually.',
        mbError, MB_OK);
  end;
end;
