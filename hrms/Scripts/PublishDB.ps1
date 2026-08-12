param(
    [bool] $IsSchemaOnly = 1,
    [string] $hash = "version",
    [string] $env = "PROD")

function PublishDbToServer {
    param(
        [Parameter(Position = 0, Mandatory = $True)]
        [string] $tmpFolder,
        [string] $env
    )
    
    # $commonPath = Resolve-Path $PSScriptRoot
    # Write-host "CommonPath = $($commonPath)"
    # cd "$commonPath"
    if ( $env -eq "STG") {
        $TargetServerConnString = "`"Data Source=kh-z10.aw02.pbodia.com;user=sa;password=P@ssword;Pooling=False`""
    }
    elseif ( $env -eq "UAT" ) {
        $TargetServerConnString = "`"Data Source=kh-x10.aw02.pbodia.com;user=sa;password=P@ssword;Pooling=False`""
    }
    elseif ($env -eq "PROD") {

        $encryptedData = 'bqIVIoLT06gTXWwM8Z3x9WhTtEpBpiNx3uwuo2HgY9e9/b3hzBCehbF083Aqk+QqXnQm6HFg60HSSY42YXnfxHoh0u/sDxAEXzy5lA4tjFXh5yqi38N5ABSRPUqji0yNhLzbk8UMWA9HGcMXWFkvkEHHNah9KC8Fs8vy880JKd3qzdY1782VPky7fB6bZqv4deJwMH15qfVCG/1c2MT96PvKdJKVOETHKi0s7uGMVUgSMz6qfbJBMYcTR7W6usN59okDkQ5tdnVoJGQ3H8jbECIv5kv/8/2FqGbIVSc7rhwIlZYzRY6dWHgtep1ba+La9SfjTdb21UJ1wWfD4aIcVw=='
        $rsaPrivateKey = New-Object System.Security.Cryptography.RSACryptoServiceProvider
        $rsaPrivateKey.FromXmlString('<RSAKeyValue><Modulus>px8Nk6a4bTsEOdaOtlsOJl7ZjEHaXsBhSn3xxiU/pELEEZOObE9Dz3KfaOIUYBvAmx01lG3h2S3nO7e/vkqZa36P6XMZSB1w+FPCj2KOtJsqWrRTwCf+kqdlBoQdZQBFiKw7nOEYpn8K15oFNhjr68cdyWKTpdXG3YTyfuDUSqwoFJCDoglQxqa6WYnSu2YkdQjtf3ws9inXylMWlJKtT58qLyvXtwY1OW+wfHvZBFHpACWtL8VOwa2CH8lQcIh+A1KQgxtGV7EQgT38g5HUXvk8Mnn0uXfsMJQVzEmC6d51iUelS2SWD9vvdXgA2sM4DRbgn6sV5OfiN3vPADnLGQ==</Modulus><Exponent>AQAB</Exponent><P>1ILIOOh3aE+B6Xr1anmr4BsNLH2ulRbgfoBVewNdY+Dr3QXsftcL84brd+wxPe3CaZR7bnm1Xybl/U0yv39NIBNLg6n2W42dvgif9PVsGujTVORgMz07YXcyst9Dqe+NZiFtLVJZiMIkAhoeEcfYWF2m8gydVTxyCkS19cMV1bU=</P><Q>yVJcIYhQEzv7SIg+yhiZ0sc/O1dBaxMqTNk+8In7vnxVYn9a+iWd8stauh73DlAn0nCx/A8VXxko0z1H5X5mh2gbuUhCwZjWAnDSh/VdqNfnTVwlrs2YdgefH48HDlpsSTIof0yFj/fbXhWT67TuWJmN3FnZMVTC8sdeTtJYPlU=</Q><DP>DDOyMWpPqOKfz/sqakGwN771468XviHBJCJURirsStbYVCmJ6cKJQiNmE82xCiBDwHIxICfG3T7Lh97N4ib6E7Rn6phwt9MK0qWGIbeRzH2/3Kj8SRTj+vWwKIsfRHunv8x9i1OmJm8XSDOdtGGoL3K5Kj7FLea5mZNWa1UXRN0=</DP><DQ>eQ9GiX9/r2FFRKNxJnPOo/neYx+gHfQjXpzQhTJkKHJ5ocY1ffB631I7V2phY8D9gUT1Mj4butjVPNk/z5fHrasD6NG0Gth98G/JmaoJTQuvckxz+H53LJp+rCqEsrPbI9S+l3X7ZsxDpHrTPUzKeoqzzIpW2Zw/smACrOE94RU=</DQ><InverseQ>LQX6+s4er9alJutrojnksGNsBdmcIMe2Xeu84ntHsWYLu4yibxcCMMs7dWA+RLM65LGYCivr1sco6ySYmOEO4OkGM2aoKgXNme8YL/q72U003eRU3e28hL2ReIn1nxB2Pp2Jl7SJTBcLhLFlx98toaet8al65l1Z9tRvFUdyf78=</InverseQ><D>JB5d/M9HhLQAkc+BQIAlar2NHvcMjvXdERBvNHfQbVvQrEvLuDTZXvhS44QZCzx0QBHsBoxJB3sYQpvP4PyWc9kCmV261Q8n30ObBQoz9Wyjlr7qatk599Ad0W60O6P3YzZ2G826WiGS4k3zWmHAe1Cpa2DWsDynojCdiDZnK0v4XQSvUmn7/7FWOfJ/IDxWz3PXH6jz9ZmwQp2ysPkKwp7xnI4yuuxAUU1yemOoRSkbBZKqNRRqLtlxX+eEvmiphO6jPOjkJjQuG8ErboWuscSt2ND7SKtFnbVygdoTSMNU4z33lLG1CN2GSXomOCXfvCcD8UEeyOQdC2hxTeuBQQ==</D></RSAKeyValue>')
        $byte =  [Convert]::FromBase64String($encryptedData)
        $decryptedData = $rsaPrivateKey.Decrypt($byte, $true)
        $decryptedString = [System.Text.Encoding]::UTF8.GetString($decryptedData) 
        $TargetServerConnString = $decryptedString
    }

    $jobs = @()
    
    $projects | % {
        $jobs += Start-Job $ScriptBlock -ArgumentList $_, $tmpFolder, $TargetServerConnString, $env
    }
  
    Wait-Job -Job $jobs | Receive-Job
        
    write-host "Job finished."
    $errorCode = 0
    $jobs | ? { $_.State -eq 'Failed' } | % {
        Write-Host ($_.ChildJobs[0].JobStateInfo.Reason.Message) -ForegroundColor Red
        $errorCode = 1
    }
    exit $errorCode
}

$ScriptBlock = {
    param($project, $tmpFolder, $TargetServerConnString, $env) 
    Write-host (get-Date).ToString("HH:mm:ss fff")
    set-Alias mSqlpackage "C:\Program Files (x86)\Microsoft SQL Server\140\DAC\bin\SqlPackage.exe"
    cd "$env:CI_PROJECT_DIR\HRMS\"
    cd ..
    $path = $project.path
    write-host "Project Path => $($path)"
    #cd $path
    # $sqlProj=$project.sqlproj

    $folder = 'F:\DBRelease\' + $tmpFolder

    if ("" + $project.dacpac -eq "") {
        $dacPath = $folder + '\' + $path + ".dacpac";
    }
    write-output "dacpac path  $dacPath"
    $dacPath = $dacPath.replace(' ', '')

    $newPublishFile = "$env:CI_PROJECT_DIR\$($project.path)\staging.publish.xml"

    $publishFile = "staging.schemaOnly.publish.xml"

    $dbName = "HRMSDB"

    if ($env -eq "PROD") {
        $dbName = "HRMSDB_Production"
    }

    (cat "$env:CI_PROJECT_DIR\$($project.path)\$($publishFile)" ).replace("{user}", "sa").replace("{password}", "P@ss0wrd").replace("{dbName}", $dbName) | out-file $newPublishFile -Encoding ascii

    $retry = 1
    
    while ($retry -lt 3) {
        write-output "Invoke-Expression mSqlpackage /SourceFile:$dacPath /Action:publish /tcs:$TargetServerConnString /Profile:$newPublishFile /Quiet:True /p:ScriptDatabaseOptions=False /p:IgnoreColumnOrder=True /v : Environment = $env -ErrorVariable ErrorOutput -Verbose:$false"
        Invoke-Expression "mSqlpackage /SourceFile:$dacPath /Action:publish /tcs:$TargetServerConnString /Profile:$newPublishFile /Quiet:True /p:ScriptDatabaseOptions=False /p:IgnoreColumnOrder=True /v:Environment=$env"  -ErrorVariable ErrorOutput -Verbose:$false
        if ($LASTEXITCODE -ne 0) {
            $retry++
        }
        else {
            break;
        }
    }
    
    if ($LASTEXITCODE -ne 0) {
        throw "Job failed. The error was: project={0},Error={1}" -f ($project.name), ([string] $ErrorOutput)
    }
    
    Write-host (get-Date).ToString("HH:mm:ss fff")
}

$projects = (
    @{name = "HRMS"; path = "HRMSDB"; sqlproj = "HRMSDB.sqlproj" }
)

write-host "preparing to publish db schema env : " + $env

Publishdbtoserver -tmpfolder $hash -env $env