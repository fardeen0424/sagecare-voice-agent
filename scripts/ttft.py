"""Measures LLM time-to-first-token (TTFT) over N streaming requests."""
import http.client, json, statistics, sys, time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SYSTEM = "You are a friendly scheduling assistant for a clinic. Keep replies to one or two sentences."
body = json.dumps({
    "model": "qwen3.8-27b", "stream": True, "max_tokens": 40,
    "messages": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Hi, I'd like to book an appointment."}],
})

times = []
for i in range(N):
    conn = http.client.HTTPConnection("127.0.0.1", 8000)
    start = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    for line in resp:
        if line.startswith(b"data: ") and b'"content"' in line:
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if delta:
                times.append((time.perf_counter() - start) * 1000)
                break
    conn.close()

times.sort()
print(f"requests={N}  p50={statistics.median(times):.0f}ms  "
      f"p95={times[int(len(times)*0.95)-1]:.0f}ms  min={times[0]:.0f}ms")