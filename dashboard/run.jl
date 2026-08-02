include(joinpath(@__DIR__, "server.jl"))

const PORT = 8765

println("Starting dashboard server on http://localhost:$PORT ...")
server = start_server(; port=PORT)

println("Launching cloudflared tunnel...")
tunnel_log = joinpath(@__DIR__, "cloudflared.log")
proc = run(pipeline(`cloudflared tunnel --url http://localhost:$PORT`; stdout=tunnel_log, stderr=tunnel_log); wait=false)

public_url = nothing
for _ in 1:60
    sleep(1)
    if isfile(tunnel_log)
        m = match(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", read(tunnel_log, String))
        if m !== nothing
            global public_url = m.match
            break
        end
    end
end

println()
if public_url !== nothing
    println("Dashboard is live at: $public_url")
else
    println("Could not detect the tunnel URL yet -- check $tunnel_log")
end
println("(local: http://localhost:$PORT)")
println("Press Ctrl+C to stop.")

try
    wait(server)
catch e
    e isa InterruptException || rethrow()
finally
    process_running(proc) && kill(proc)
    close(server)
end
