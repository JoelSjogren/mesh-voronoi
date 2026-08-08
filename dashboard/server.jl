using HTTP
using HTTP.WebSockets
using GLMakie

include(joinpath(@__DIR__, "cases.jl"))

const IMAGES_DIR = joinpath(@__DIR__, "images")
mkpath(IMAGES_DIR)

"""
Shared, mutable dashboard state: the latest known result for each case
(`nothing` until it's run at least once), the id of the most recently
*started* run, and every currently-connected WebSocket (each new result
gets pushed to all of them as soon as it's ready, which is what makes
results "arrive one at a time" live in the browser instead of all at once
at the end).
"""
mutable struct DashboardState
    results::Dict{String,Any}     # name -> (pass, run_id, detail, image_url)
    run_id::Int
    sockets::Vector{Any}
    lock::ReentrantLock
end

const STATE = DashboardState(Dict{String,Any}(), 0, [], ReentrantLock())

function broadcast_json(msg::String)
    lock(STATE.lock) do
        dead = Int[]
        for (i, ws) in enumerate(STATE.sockets)
            try
                WebSockets.send(ws, msg)
            catch
                push!(dead, i)
            end
        end
        deleteat!(STATE.sockets, dead)
    end
end

json_string(s::AbstractString) = "\"" * replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""

function result_json(name::String, r)
    pass, run_id, detail, image_url = r
    case = only(c for c in CASES if c.name == name)
    return """{"type":"result","name":$(json_string(name)),"description":$(json_string(case.description)),"group":$(json_string(case.group)),"pass":$pass,"run_id":$run_id,"detail":$(json_string(detail)),"image_url":$(json_string(image_url))}"""
end

"""
Run every case in `CASES`, one at a time, saving its figure to `images/`
and broadcasting the result to every connected browser as soon as it's
ready -- this is the "one test case result at a time as they pass one by
one" live-update behavior.

A case's second return value is either a Makie `Figure` (the common case:
`run_all_cases!` saves it itself) or a `String` giving a path to an image
the case already saved on its own -- needed for cases that run the
dimension-generic redesign (`mesh-voronoi-nd`) as a subprocess, since a
Makie figure built in a different package's own process/environment can't
be handed back as a live object.
"""
function run_all_cases!()
    run_id = lock(STATE.lock) do
        STATE.run_id += 1
        STATE.run_id
    end
    # Tell every connected browser a fresh run has *started* right away, so
    # cards can grey out immediately rather than staying "fresh" until each
    # one's new result happens to arrive.
    broadcast_json("""{"type":"run_started","run_id":$run_id}""")
    for case in CASES
        pass, detail = false, "crashed before producing a result"
        image_url = ""
        try
            ok, fig, d = case.run()
            pass, detail = ok, d
            if fig isa AbstractString
                image_url = fig
            else
                image_path = joinpath(IMAGES_DIR, "$(case.name)_$(run_id).png")
                save(image_path, fig)
                image_url = "/images/$(case.name)_$(run_id).png"
            end
        catch e
            detail = "crashed: " * sprint(showerror, e)
        end
        lock(STATE.lock) do
            STATE.results[case.name] = (pass, run_id, detail, image_url)
        end
        broadcast_json(result_json(case.name, STATE.results[case.name]))
    end
end

"""
Static, one-off investigation report pages -- separate from the live
test-case dashboard (`/`), reusing only its HTTP server + image serving +
cloudflare tunnel. `(key, file, label, first_published)`: `key` maps a
route under `/reports/` to a static HTML file under `public/`; `label` is
what the cross-page nav dropdown (`nav_html`) shows for it;
`first_published` (`"YYYY-MM-DD"`, or `"unknown"` when no reliable evidence
of the original date survives -- filesystem timestamps get overwritten by
routine edits, so aren't trustworthy provenance -- never guess a plausible-
looking one instead) is shown via `date_html`. Add an entry here for each
new report -- order here is the order it appears in the dropdown.
"""
const REPORT_LIST = [
    ("poster", "poster.html", "Algorithm poster", "2026-07-31"),
    ("multiseg-overlap", "report_multiseg_overlap.html", "Multi-segment overlap", "unknown"),
    ("curved-twice", "report_curved_twice.html", "Curved bisector crossing twice", "unknown"),
    ("self-intersect", "report_self_intersect.html", "Self-intersecting boundary", "2026-07-28"),
    ("hole-topology", "report_hole_topology.html", "Cell territory with a hole", "2026-07-30"),
    ("triangulation-difficulties", "report_triangulation_difficulties.html", "Triangulation difficulties (historical)", "2026-07-30"),
    ("infinity-layer", "report_infinity_layer.html", "Layer at infinity (planning)", "2026-08-04"),
    ("nonmanifold-cap", "report_nonmanifold_cap.html", "3D: a genuine tie point (3 segments)", "2026-08-07"),
    ("tie-gallery", "report_tie_gallery.html", "Every k-way tie locus, n=2 and n=3", "2026-08-08"),
]
const REPORTS = Dict(key => file for (key, file, _, _) in REPORT_LIST)

"""
The cross-page `<select>` nav (dashboard + every report), injected in place
of a literal `<!--NAV-->` marker in each static HTML file at load time --
the single source of truth for the report list is `REPORT_LIST` above, not
each page's own markup, so adding a report never means hand-editing
navigation into every other existing page. `current_key` (`""` for the
dashboard itself) picks which option starts pre-selected.
"""
function nav_html(current_key::String)
    opt(value, key, label) = """<option value="$value"$(key == current_key ? " selected" : "")>$label</option>"""
    options = [opt("/", "", "Dashboard")]
    for (key, _, label, _) in REPORT_LIST
        push!(options, opt("/reports/$key", key, label))
    end
    select_style = "background:var(--card-bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:0.35rem 0.6rem;font-size:0.85rem;font-family:inherit;"
    return """<select onchange="if(this.value) location.href=this.value;" style="$select_style">""" *
           join(options) * "</select>"
end

"""
The "first published" annotation injected in place of a literal
`<!--DATE-->` marker in each report's own header. `"unknown"` is rendered
honestly as unknown, not silently omitted or backfilled with today's date
-- provenance that can't be verified shouldn't look verified.
"""
function date_html(first_published::String)
    text = first_published == "unknown" ? "First published: date unknown (predates this project's tracked history)" : "First published: $first_published"
    return """<div style="color:var(--muted);font-size:0.85rem;">$text</div>"""
end

const INDEX_HTML = replace(read(joinpath(@__DIR__, "public", "index.html"), String), "<!--NAV-->" => nav_html(""))
const REPORT_HTML = Dict(key => replace(read(joinpath(@__DIR__, "public", file), String),
    "<!--NAV-->" => nav_html(key), "<!--DATE-->" => date_html(date)) for (key, file, _, date) in REPORT_LIST)

"""
Route `req` to a plain `(status::Int, headers::Vector{Pair}, body)` tuple
-- deliberately *not* an `HTTP.Response`, whose `.body` is a wrapped wire
type (`HTTP.BytesBody` in this HTTP.jl version) that isn't itself directly
`write`-able to an `HTTP.Stream`; a plain `String`/`Vector{UInt8}` is, so
the stream-writing code in `start_server` writes `body` straight through.
"""
function handle(req::HTTP.Request)
    target = HTTP.URI(req.target).path

    if target == "/" || target == "/index.html"
        return 200, ["Content-Type" => "text/html"], INDEX_HTML
    elseif startswith(target, "/reports/")
        key = target[length("/reports/")+1:end]
        haskey(REPORT_HTML, key) || return 404, [], "no such report: $key"
        return 200, ["Content-Type" => "text/html"], REPORT_HTML[key]
    elseif startswith(target, "/images/")
        path = joinpath(IMAGES_DIR, basename(target))
        isfile(path) || return 404, [], "not found"
        return 200, ["Content-Type" => "image/png"], read(path)
    elseif target == "/cases" && req.method == "GET"
        lock(STATE.lock) do
            entries = String[]
            for case in CASES
                if haskey(STATE.results, case.name)
                    push!(entries, result_json(case.name, STATE.results[case.name]))
                else
                    push!(entries, """{"type":"result","name":$(json_string(case.name)),"description":$(json_string(case.description)),"group":$(json_string(case.group)),"pass":null,"run_id":0,"detail":"not yet run","image_url":""}""")
                end
            end
            body = """{"current_run_id":$(STATE.run_id),"cases":[""" * join(entries, ",") * "]}"
            return 200, ["Content-Type" => "application/json"], body
        end
    elseif target == "/rerun" && req.method == "POST"
        @async run_all_cases!()
        return 200, [], "started"
    else
        return 404, [], "not found"
    end
end

"""
Start the dashboard's HTTP server (serving `/`, `/cases`, `/images/*`,
`POST /rerun`) with a `/ws` WebSocket upgrade for live push, run every case
once up front, and return the running `HTTP.Server` (call `close` on it to
stop). Does not itself start a `cloudflared` tunnel -- see `run.jl`.
"""
function start_server(; port=8765)
    server = HTTP.listen!(port) do http::HTTP.Stream
        try
            if HTTP.WebSockets.isupgrade(http.message)
                HTTP.WebSockets.upgrade(http) do ws
                    lock(STATE.lock) do
                        push!(STATE.sockets, ws)
                    end
                    try
                        for _ in ws
                            # the client never sends anything meaningful; just
                            # keep the socket open until it disconnects
                        end
                    catch
                    end
                end
            else
                req = http.message
                read(http)   # drain any request body -- none of our routes need its content
                HTTP.closeread(http)
                status, headers, body = handle(req)
                HTTP.setstatus(http, status)
                for h in headers
                    HTTP.setheader(http, h)
                end
                HTTP.startwrite(http)
                write(http, body)
                HTTP.closewrite(http)
            end
        catch e
            @error "dashboard server handler error" exception = (e, catch_backtrace())
        end
    end
    @async run_all_cases!()
    return server
end
