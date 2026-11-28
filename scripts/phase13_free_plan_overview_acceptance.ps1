$ErrorActionPreference = "Stop"

$baseUrl = "http://127.0.0.1:8000"
$videoId = "VKrtiTgBeng"
$maxWaitSeconds = 3900

$question = @'
Summarize the entire video chronologically in 7 points.

Ensure the summary covers the beginning, middle, and final portions
of the transcript. Include important developments from later
sections, not just the opening half.

Preserve the sequence of events, important arguments, explanations,
people, conclusions, and relevant details.

Use only transcript-supported information. Do not invent transitions
or combine unrelated events. Include relevant timestamps and
clearly qualify anything the transcript does not establish.
'@

$body = @{
    video_id = $videoId
    question = $question
    conversation_history = @()
} | ConvertTo-Json -Depth 10

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

try {
    Write-Host "Submitting whole-video acceptance request..."
    $accepted = Invoke-RestMethod `
        -Method Post `
        -Uri "$baseUrl/chat" `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 120

    if ($accepted.job_id) {
        $jobId = [string]$accepted.job_id
        $jobUri = "$baseUrl/chat/jobs/$jobId"
        $deadline = [DateTimeOffset]::UtcNow.AddSeconds($maxWaitSeconds)
        $pollDelay = 5

        Write-Host "Accepted as durable job: $jobId"
        Write-Host "Polling $jobUri"

        while ([DateTimeOffset]::UtcNow -lt $deadline) {
            Start-Sleep -Seconds $pollDelay
            $job = Invoke-RestMethod `
                -Method Get `
                -Uri $jobUri `
                -TimeoutSec 60

            if ($job.status -eq "completed") {
                $response = $job.result
                break
            }

            if ($job.status -eq "failed") {
                $message = if ($job.error_message) {
                    [string]$job.error_message
                } else {
                    "The whole-video summary job failed."
                }

                if ($job.retry_after_seconds) {
                    $message += " Provider retry-after: $($job.retry_after_seconds) seconds."
                }

                throw $message
            }

            if ($job.status -notin @("queued", "running")) {
                throw "Unexpected chat job status: $($job.status)"
            }

            Write-Host ("Job status: {0}; next check in {1}s." -f $job.status, $pollDelay)
            $pollDelay = [math]::Min(10, [math]::Ceiling($pollDelay * 1.4))
        }

        if (-not $response) {
            throw "The summary job did not complete within $maxWaitSeconds seconds. Its durable job remains available for a later status check."
        }
    } else {
        # Preserve compatibility if the backend returns a synchronous ChatResponse.
        $response = $accepted
    }

    $stopwatch.Stop()

    if (-not $response.answer) {
        throw "The completed response did not contain an answer."
    }

    $overview = $response.observability.retrieval.overview
    $terminal = $overview.terminal_tail_scan
    $endCap = $terminal.end_cap_scan
    $coverage = $overview.verified_terminal_tail_coverage
    $audit = $response.observability.generation.answer_evidence_verification

    Write-Host "`n===== FINAL ANSWER ====="
    $response.answer

    $requiredFindingIds = @()
    foreach ($finding in $endCap.finding_source_chunk_ids) {
        foreach ($chunkId in $finding) {
            $requiredFindingIds += [string]$chunkId
        }
    }
    $requiredFindingIds = @($requiredFindingIds | Select-Object -Unique)

    $verifiedEndingIds = @(
        $coverage.end_cap_supporting_chunk_ids |
        ForEach-Object { [string]$_ }
    )

    $matchedFindingIds = @(
        $requiredFindingIds |
        Where-Object { $verifiedEndingIds -contains $_ }
    )

    Write-Host "`n===== ACCEPTANCE DIAGNOSTICS ====="
    [PSCustomObject]@{
        ElapsedSeconds          = [math]::Round($stopwatch.Elapsed.TotalSeconds, 2)
        IndexedChunks           = $overview.indexed_document_count
        SelectedChunks          = $overview.selected_document_count
        SectionCount            = $overview.section_count
        MapCalls                = $overview.map_calls
        VerifiedPoints          = $overview.distinct_verified_point_count
        TimelineCoverage        = $overview.verified_timeline_coverage.coverage_fraction
        AuditStatus             = $audit.status
        TerminalScanStatus      = $terminal.status
        EndCapScanStatus        = $endCap.status
        EndCapFindingCount      = $endCap.finding_count
        EndCapFindingChunkIds   = ($requiredFindingIds -join ",")
        VerifiedEndCap          = $coverage.end_cap_covered
        VerifiedEndCapIds       = ($verifiedEndingIds -join ",")
        MatchedFindingIds       = ($matchedFindingIds -join ",")
        ReserveVerification     = $audit.overview_verification.reserve_verification_attempted
        PrimaryEndCapCovered    = $audit.overview_verification.primary_end_cap_covered
    } | Format-List

    if ($audit.status -ne "completed") {
        throw "Acceptance failed: evidence verification did not complete."
    }

    if ($overview.verified_timeline_coverage.coverage_fraction -lt 1.0) {
        throw "Acceptance failed: verified coverage does not span all timeline buckets."
    }

    if ($overview.distinct_verified_point_count -lt 7) {
        throw "Acceptance failed: fewer than seven distinct verified points were retained."
    }

    if ($endCap.finding_count -gt 0) {
        if ($coverage.end_cap_covered -ne $true) {
            throw "Acceptance failed: the selected verified answer does not cover the ending."
        }

        if ($requiredFindingIds.Count -gt 0 -and $matchedFindingIds.Count -eq 0) {
            throw "Acceptance failed: the generated ending candidate was not retained in the selected answer."
        }
    }

    Write-Host "`nPHASE 13 WHOLE-VIDEO ACCEPTANCE GATES PASSED."
}
catch {
    $stopwatch.Stop()
    Write-Host "`nAcceptance run failed after $([math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds."

    if ($_.Exception.Response) {
        try {
            Write-Host "HTTP status: $([int]$_.Exception.Response.StatusCode)"
        }
        catch {}
    }

    if ($_.ErrorDetails.Message) {
        Write-Host $_.ErrorDetails.Message
    }

    Write-Host $_.Exception.Message
    throw
}
