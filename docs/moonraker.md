# Moonraker printer connectivity

Manifold can upload generated G-code, request immediate printing, and monitor a
Moonraker/Klipper printer. Networking is optional and separate from slicing.

## CLI

```sh
# Existing offline slicing: no printer access.
manifold model.stl -o part.gcode

# Upload only. This does not intentionally start the file.
manifold model.stl -o part.gcode --printer-url http://printer.local:7125 --upload

# One combined upload-and-start request, then monitor that canonical file.
manifold model.stl -o part.gcode --printer-url http://printer.local:7125 --print --monitor

# Attach to the existing active job: no mesh required, no file written/uploaded.
manifold --printer-url http://printer.local:7125 --monitor
```

Add `--printer-api-key KEY` if required. Command-line arguments may be visible in
process listings/shell history; avoid sharing captured commands with credentials.
Use HTTPS on untrusted networks. TLS certificate verification is not bypassed.
Reverse-proxy directory URLs such as `https://printer.example/moonraker` work with
or without a final slash. Do not embed credentials, query strings or fragments in
the endpoint; redirects are rejected rather than forwarding the API key.

- A URL alone **never uploads**. Mesh input plus a URL alone still slices offline.
- `--monitor` without upload/print attaches without processing any mesh inputs.
- Upload/print requires mesh inputs and a URL. Invalid intent is rejected before
  slicing. `--upload --monitor` requires `--print`; otherwise use monitor-only.
- `--upload --print` means the same single upload-and-start as `--print`.
- Upload reports the canonical stored filename and **Uploaded**, **Started**,
  **Queued**, or **StartNotConfirmed**. A stored/queued file is not a confirmed
  immediate print; `--print` returns nonzero unless Started is confirmed. No extra
  start request is sent to compensate for an ambiguous response.
- Monitoring names the exact file. Print+monitor waits to observe it active before
  accepting completion, rather than accepting an old completed job. A very short
  job whose active transition was missed may time out instead of claiming success.
- Print errors, cancellation, terminal authentication errors and timeouts return
  nonzero. Initial readiness, job observation, and continuous disconnection waits
  are bounded at 30 seconds. There is no total time limit on a healthy active print.
- Ctrl-C exits monitoring with nonzero status and **does not cancel the print**.
  Inspect the printer if an upload/control request was interrupted after transmission.

## GUI

Expand the **Printer** panel, enter **Draft URL** and the masked optional API key,
and choose whether this profile should auto-connect when loaded. **Save connection
settings** applies a validated draft without contacting the printer; **Save Profile**
serializes the visible validated draft as well. API keys are stored as plaintext
inside the optional profile configuration. Existing profiles without Moonraker work.

**Connect draft** deliberately opens that target. The **Active target** label is
immutable for that session; editing draft settings never silently retargets active
controls. Every profile load retires the previous session, including profiles with
no Moonraker settings or with auto-connect off. **Reconnect active** retries the
existing target; use Connect draft for corrected credentials or a different target.

After slicing, use **Upload to Printer** or **Upload & Print** in the toolbar.
Starts are guarded by fresh Klipper readiness/job queries and local single-flight
admission. Active/paused jobs, unknown state and printer errors block starting.
Cold heaters alone are not a blocker: normal print-start G-code performs heating.
Manifold rejects overwriting the active file; current uploads accept a basename,
not a directory path. The server is still authoritative for races with other clients.

Operation IDs show pending work, upload bytes read, server outcomes and actionable
errors. **Upload progress is client bytes read, not server completion or print
progress.** Ordinary errors remain until acknowledged; emergency history uses the bounded summary described below. A pending start can reserve upload
admission until the corresponding active job is observed; if it stays uncertain,
inspect Moonraker before explicitly reconnecting to reconcile.

Telemetry updates repaint the UI at bounded intervals, including when the panel is
collapsed. Stale data is labeled. The job view includes filename, print progress,
thermals, Klipper error message, elapsed time, optional reported layers and Z, and
**approximate progress-based remaining time**. Remaining time is unknown for invalid/
zero progress, pauses or stale data. Klipper's toolhead motion clock is not a job
ETA, and Z never implies planar layer numbers in this non-planar slicer.

Pause/Resume are ordinary ordered controls. Cancel asks for confirmation tied to
the current session/job. **Disconnect is not cancellation** and cannot undo a
request already transmitted. Dropping/retiring a session cancels local owned work,
rejects unsent commands and stops reconnecting without blocking the UI on a join.
Profile replacement and Connect draft retain completed/interrupted results separately
from the retired worker, tagged with endpoint, session and operation ID, even when
the new profile has no connection. The last eight retired sessions retain details;
older results become explicit success/failure/unknown counters (old target details
are compacted). Profile resets never acknowledge these warnings. Acknowledge results
deliberately after inspecting the affected printer; no old worker is held for history.

**Emergency Stop** uses a separate bounded HTTP path and does not wait for a WS
handshake, held upload, full normal queue or retained ordinary failures. One stop
may be in flight at a time. Once it completes, another deliberate stop can dispatch
without acknowledging unrelated results, even if the previous stop failed. The
latest stop retains its detailed result; older stop results become bounded visible
success/failure/unknown counters until deliberately acknowledged. No stop is replayed
automatically. Its result is visible; an interrupted
request can have an unknown remote outcome. It is best-effort network control,
**not a hardware safety guarantee**. Use the printer's physical safety mechanisms
when needed. Do not treat UI disconnection or a failed request as evidence that
motion/heaters have stopped.

## Validation and optional live verification

Automated regressions use loopback HTTP/WebSocket servers, temporary meshes/files,
CLI subprocesses and headless egui frames. They do not contact saved endpoints,
launch the GUI event loop or validate physical printer/visual behavior. Run:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace --release
```

Live testing is optional and must be explicitly authorized by the printer operator:

1. Use a disposable profile and confirm the exact endpoint/credentials with its
   operator. Start with a read-only `server/info` request or a GUI connection/
   subscription. Check readiness, filename, thermals and stale/disconnect behavior;
   **do not click upload, print, cancel or emergency stop** as a connectivity probe.
2. If an existing safe job is already running, opt into monitor-only and verify
   its exact filename and updates. Exit monitoring; confirm the print continues.
3. Only with separate operator authorization, a cleared/prepared machine, reviewed
   G-code and physical supervision, consider upload/print/control testing. Use a
   disposable distinct filename. These actions can move/hot-start/stop hardware;
   they are never automatic acceptance steps.
4. Record Moonraker/Klipper versions, transport/proxy/auth mode and observed results,
   excluding credentials. Local mock success is not a claim of live-device support
   verification for a particular installation.

Current API references:

- <https://moonraker.readthedocs.io/en/latest/external_api/file_manager/#file-upload>
- <https://moonraker.readthedocs.io/en/latest/external_api/server/#identify-connection>
- <https://www.klipper3d.org/Status_Reference.html>
