FINETUNED = "SAM 2.1 tiny · fine-tuned (demo_run) · box"
board = F.benchmark({FINETUNED: F.SAM2Prompted(ckpt=ckpt, prompt="box")})
