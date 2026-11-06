#!/bin/bash
# Usage: server.sh start|stop|restart|status   (MODEL env overrides model path)
ROOT=$HOME/finops-eval
MODEL=${MODEL:-$ROOT/models/qwen3.8-27b-mlx-8bit}
PIDF=$ROOT/logs/server.pid
start() {
  source $ROOT/.venv/bin/activate
  echo "=== start $(date) model=$MODEL ===" >> $ROOT/logs/server.log
  nohup python -m mlx_lm server --model "$MODEL" --host 127.0.0.1 --port 8080 \
     --max-tokens 16384 --prompt-cache-size ${PROMPT_CACHE_SIZE:-10} --log-level INFO >> $ROOT/logs/server.log 2>&1 &
  echo $! > $PIDF
  nohup caffeinate -is -w $(cat $PIDF) >/dev/null 2>&1 &   # prevent idle/system sleep only while the server runs (no settings changed)
  for i in $(seq 1 120); do curl -s -m 5 http://127.0.0.1:8080/v1/models >/dev/null && { echo "server up (pid $(cat $PIDF))"; return 0; }; sleep 3; done
  echo "server failed to start"; return 1
}
stop() { [ -f $PIDF ] && kill $(cat $PIDF) 2>/dev/null; pkill -f "mlx_lm server --model" 2>/dev/null; sleep 3; rm -f $PIDF; echo stopped; }
case "$1" in
  start) start;; stop) stop;; restart) stop; start;;
  status) curl -s -m 5 http://127.0.0.1:8080/v1/models && echo || echo down;;
esac
