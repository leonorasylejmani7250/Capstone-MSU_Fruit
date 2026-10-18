# Step 5: score the fine-tuned model on the same clip, with the same boxes.
FINETUNED = "SAM 2.1 tiny · fine-tuned (demo_run) · box"
board = F.benchmark({FINETUNED: F.SAM2Prompted(ckpt=ckpt, prompt="box")})
