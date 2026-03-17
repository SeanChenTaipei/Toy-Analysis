param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [Parameter(Mandatory = $false)]
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"

$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
if (-not $OutputPath -or [string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = [System.IO.Path]::ChangeExtension($resolvedInput, ".csv")
}

$ext = [System.IO.Path]::GetExtension($resolvedInput).ToLowerInvariant()
if ($ext -ne ".xls" -and $ext -ne ".xlsx") {
    throw "Only .xls or .xlsx is supported. input=$resolvedInput"
}

if ($ext -eq ".xlsx") {
    $props = "Excel 12.0 Xml;HDR=YES;IMEX=1"
}
else {
    $props = "Excel 8.0;HDR=YES;IMEX=1"
}

$connStr = "Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$resolvedInput;Extended Properties='$props';"
$conn = New-Object System.Data.OleDb.OleDbConnection($connStr)
$conn.Open()

try {
    $schema = $conn.GetOleDbSchemaTable([System.Data.OleDb.OleDbSchemaGuid]::Tables, $null)
    $sheet = $schema |
        Where-Object { $_.TABLE_NAME -like '*$' -or $_.TABLE_NAME -like "*$'" } |
        Select-Object -First 1 -ExpandProperty TABLE_NAME

    if (-not $sheet) {
        throw "No worksheet found."
    }

    $cmd = $conn.CreateCommand()
    $cmd.CommandText = "SELECT * FROM [$sheet]"

    $adapter = New-Object System.Data.OleDb.OleDbDataAdapter($cmd)
    $dt = New-Object System.Data.DataTable
    [void]$adapter.Fill($dt)

    $dt | Export-Csv -Path $OutputPath -NoTypeInformation -Encoding UTF8

    Write-Output "Input: $resolvedInput"
    Write-Output "Sheet: $sheet"
    Write-Output "Rows: $($dt.Rows.Count)"
    Write-Output "Cols: $($dt.Columns.Count)"
    Write-Output "Output: $OutputPath"
}
finally {
    $conn.Close()
}
