# Codexa 100M Training Viewer

Open the live training viewer:

```bash
CODEXA_TRAINING_RUN=codexa-100m-native-v1 ./scripts/open_100m_training_viewer.sh
```

The viewer attaches to the running `codexa-100m-native-v1` process and displays
the current step, loss, tokens/second, token count, elapsed time, progress bar,
and checkpoint directory.

To open it in a new fixed-size Kitty terminal:

```bash
kitty --override remember_window_size=no \
  --override initial_window_width=100c \
  --override initial_window_height=22c \
  --title 'Codexa 100M Base Training' \
  zsh -lc 'CODEXA_TRAINING_RUN=codexa-100m-native-v1 exec ./scripts/open_100m_training_viewer.sh'
```

The viewer is read-only. Closing it does not stop training.
