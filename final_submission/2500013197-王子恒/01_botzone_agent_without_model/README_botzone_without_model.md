# Botzone agent without model

This directory contains the Botzone-runnable agent package without model
weights.

Files:

- `mahjong_v7_call150_friendstyle_storage_bot.zip`: upload this as the Botzone
  bot code package.
- `botzone_final_source/`: unpacked source for inspection.

Required model storage file on Botzone:

- Upload `mahjong_v7_calibrated_best.pkl` to Botzone storage `/data`.
- Runtime path expected by the bot: `data/mahjong_v7_calibrated_best.pkl`.

The no-model package intentionally does not include `.pkl` or `.pt` files.
