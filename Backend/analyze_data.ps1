# PowerShell script to analyze exoplanet data files

function Test-JsonStructure {
    param([string]$FilePath)

    Write-Host "Analyzing: $FilePath" -ForegroundColor Green
    Write-Host "=================================================="

    try {
        # Check if file exists
        if (-not (Test-Path $FilePath)) {
            Write-Host "Error: File not found - $FilePath" -ForegroundColor Red
            return
        }

        # Read and parse JSON
        $jsonContent = Get-Content $FilePath -Raw -ErrorAction Stop
        $data = $jsonContent | ConvertFrom-Json

        Write-Host "Total records: $($data.Count)"
        Write-Host ""

        # Find records with actual data
        $dataRecords = @()
        foreach ($record in $data) {
            if ($record -is [PSCustomObject]) {
                $hasData = $false
                foreach ($property in $record.PSObject.Properties) {
                    # Skip metadata and empty fields
                    if ($property.Name -notlike '*#*' -and
                        $property.Name -ne '' -and
                        $property.Name -notlike 'FIELD*' -and
                        $null -ne $property.Value -and
                        $property.Value -ne '' -and
                        $property.Value -ne '[]' -and
                        $property.Value -ne 'N/A' -and
                        $property.Value -ne 'null') {
                        $hasData = $true
                        break
                    }
                }
                if ($hasData) {
                    $dataRecords += $record
                }
            }
        }

        Write-Host "Records with data: $($dataRecords.Count)"
        Write-Host ""

        if ($dataRecords.Count -gt 0) {
            $sampleRecord = $dataRecords[0]

            Write-Host "Available fields in data records:"
            $fieldCount = 0
            foreach ($property in $sampleRecord.PSObject.Properties) {
                if ($property.Name -notlike '*#*' -and $property.Name -ne '' -and $property.Name -notlike 'FIELD*') {
                    $value = $property.Value
                    $valueType = if ($null -eq $value) { "null" } else { $value.GetType().Name }
                    $valueStr = if ($null -eq $value) { "null" } else { $value.ToString() }
                    if ($valueStr.Length -gt 50) { $valueStr = $valueStr.Substring(0, 50) + "..." }
                    Write-Host "  $($property.Name): $valueType = $valueStr"
                    $fieldCount++
                    if ($fieldCount -ge 15) {
                        Write-Host "  ... (showing first 15 fields)"
                        break
                    }
                }
            }
            Write-Host ""

            # Look for classification fields
            $classificationFields = @('disposition', 'default_flag', 'tfopwg_disp', 'pl_name', 'pl_status', 'solution_type', 'pl_status')
            Write-Host "Classification-relevant fields:"
            foreach ($field in $classificationFields) {
                if ($sampleRecord.PSObject.Properties.Name -contains $field) {
                    $value = $sampleRecord.$field
                    Write-Host "  $field`: $value"
                }
            }
            Write-Host ""

            # Show unique values for disposition field if it exists
            if ($sampleRecord.PSObject.Properties.Name -contains 'disposition') {
                $dispositions = $dataRecords | Where-Object { $null -ne $_.disposition -and $_.disposition -ne '' } | Select-Object -ExpandProperty disposition -Unique
                Write-Host "Unique disposition values: $($dispositions -join ', ')"
                Write-Host ""
            }

            # Show data statistics
            Write-Host "Data Statistics:"
            $numericFields = @()
            foreach ($property in $sampleRecord.PSObject.Properties) {
                if ($property.Name -notlike '*#*' -and $property.Value -is [double] -or $property.Value -is [int]) {
                    $numericFields += $property.Name
                }
            }

            if ($numericFields.Count -gt 0) {
                Write-Host "  Numeric fields found: $($numericFields.Count)"
                Write-Host "  Sample numeric field: $($numericFields[0])"
            }

            $totalSize = $dataRecords.Count
            Write-Host "  Total valid records: $totalSize"
            Write-Host ""
        } else {
            Write-Host "No valid data records found in this file."
            Write-Host ""
        }

    } catch {
        Write-Host "Error analyzing $FilePath`: $($_.Exception.Message)" -ForegroundColor Red
    }

    Write-Host ""
}

# Main execution
Write-Host "Exoplanet Dataset Analysis" -ForegroundColor Cyan
Write-Host "=========================="
Write-Host ""

# Define file paths (check current directory and subdirectories)
$dataFiles = @(
    "k2pandc_2025.10.04_07.10.02.json",
    "TOI_2025.10.04_07.06.07.json",
    "..\Downloads\cumulative_2025.10.04_06.25.10.json"
)

Write-Host "Looking for the following dataset files:" -ForegroundColor Yellow
foreach ($file in $dataFiles) {
    Write-Host "  - $file"
}
Write-Host ""

# Analyze each file
foreach ($file in $dataFiles) {
    if (Test-Path $file) {
        Test-JsonStructure $file
    } else {
        Write-Host "Warning: File not found - $file" -ForegroundColor Yellow
        # Try alternative locations
        $alternativePaths = @(
            ".\$file",
            "..\data\$file",
            "data\$file"
        )

        $found = $false
        foreach ($altPath in $alternativePaths) {
            if (Test-Path $altPath) {
                Write-Host "Found at alternative location: $altPath" -ForegroundColor Green
                Test-JsonStructure $altPath
                $found = $true
                break
            }
        }

        if (-not $found) {
            Write-Host "Could not locate file: $file" -ForegroundColor Red
        }
    }
}

Write-Host "Analysis complete!" -ForegroundColor Cyan