#!/usr/bin/env bash
set -u

# Minimal 100x22 Kitty dashboard shared by every Codexa training launcher.
run_name="${1:?run name required}"
total_steps="${2:?total steps required}"
run_dir="${3:?run directory required}"
checkpoint="${4:?checkpoint path required}"
pid="${5:-}"
title="${6:-TRAINING}"
metrics="$run_dir/train_metrics.jsonl"
started=$(date +%s)

cyan=$'\033[38;5;81m'; white=$'\033[1;97m'; muted=$'\033[38;5;245m'
green=$'\033[38;5;120m'; yellow=$'\033[38;5;221m'; reset=$'\033[0m'

while :; do
  steps=0; loss='--'; val='--'; speed='--'; tokens='--'; eta='--:--:--'
  if [[ -f "$metrics" ]]; then
    steps=$(wc -l < "$metrics")
    last=$(tail -n 1 "$metrics")
    field() { printf '%s' "$last" | sed -n "s/.*\"$1\": \([^,}]*\).*/\1/p"; }
    loss=$(field training_loss); val=$(field validation_loss)
    speed=$(field tokens_per_second); tokens=$(field total_tokens_seen)
    [[ -n "$loss" && "$loss" != null ]] || loss='--'
    [[ -n "$val" && "$val" != null ]] || val='--'
    [[ -n "$speed" && "$speed" != null ]] || speed='--'
    [[ -n "$tokens" && "$tokens" != null ]] || tokens='--'
  fi
  [[ "$speed" == '--' ]] || speed=$(awk -v v="$speed" 'BEGIN{printf "%.0f",v}')
  [[ "$tokens" == '--' ]] || tokens=$(awk -v v="$tokens" 'BEGIN{printf "%.1fM",v/1000000}')
  [[ "$loss" == '--' ]] || loss=$(awk -v v="$loss" 'BEGIN{printf "%.3f",v}')
  [[ "$val" == '--' ]] || val=$(awk -v v="$val" 'BEGIN{printf "%.3f",v}')
  alive=0; [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=1
  if (( steps >= total_steps )); then status='COMPLETE'; pct=100; eta='00:00:00'; status_color=$green
  elif (( alive )); then status='RUNNING'; pct=$((steps*100/total_steps)); status_color=$cyan
  else status='STOPPED'; pct=$((steps*100/total_steps)); status_color=$yellow; fi
  (( pct > 100 )) && pct=100
  if [[ "$speed" != '--' && "$steps" -lt "$total_steps" ]]; then
    seconds=$(awk -v r="$((total_steps-steps))" -v s="$speed" 'BEGIN{if(s>0)printf "%d",r*65536/s;else print 0}')
    eta=$(printf '%02d:%02d:%02d' $((seconds/3600)) $(((seconds/60)%60)) $((seconds%60)))
  fi
  filled=$((pct*50/100)); empty=$((50-filled))
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#'); rest=$(printf '%*s' "$empty" '' | tr ' ' '-')
  now=$(date +%s); elapsed=$((now-started))

  printf '\033[2J\033[H'
  printf '%s  CODEXA V1%s   %s%s%s\n' "$cyan" "$reset" "$white" "$title" "$reset"
  printf '%s  ------------------------------------------------------------------------%s\n\n' "$muted" "$reset"
  printf '  %s%s%s  %s%3d%%%s    ETA %s%s%s    %s%s%s\n\n' "$status_color" "$bar$rest" "$reset" "$white" "$pct" "$reset" "$white" "$eta" "$reset" "$status_color" "$status" "$reset"
  printf '  %sSTEPS%s       %s%s%s\n' "$muted" "$reset" "$white" "$steps / $total_steps" "$reset"
  printf '  %sSPEED%s       %s%s tok/s%s\n' "$muted" "$reset" "$white" "$speed" "$reset"
  printf '  %sTRAIN LOSS%s  %s%s%s\n' "$muted" "$reset" "$white" "$loss" "$reset"
  printf '  %sVAL LOSS%s    %s%s%s\n' "$muted" "$reset" "$white" "$val" "$reset"
  printf '  %sTOKENS%s      %s%s%s\n' "$muted" "$reset" "$white" "$tokens" "$reset"
  printf '  %sELAPSED%s     %s%02d:%02d:%02d%s\n\n' "$muted" "$reset" "$white" $((elapsed/3600)) $(((elapsed/60)%60)) $((elapsed%60)) "$reset"
  printf '  %sCHECKPOINT%s\n  %s%s%s\n\n' "$cyan" "$reset" "$muted" "$checkpoint" "$reset"
  printf '  %sLOG%s\n  %s%s%s\n\n' "$cyan" "$reset" "$muted" "$metrics" "$reset"
  printf '  %sBF16  |  CUDA  |  refresh 2s%s\n' "$muted" "$reset"
  if [[ "$status" != RUNNING ]]; then printf '\n  %sRun is %s. Press Enter to close.%s' "$status_color" "$status" "$reset"; read -r _; break; fi
  sleep 2
done
