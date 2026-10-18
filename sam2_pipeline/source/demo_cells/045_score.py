# Step 3: score pretrained SAM 2.1 on the held-out clip. It gets one box per cluster and must follow each one.
PRETRAINED = "SAM 2.1 tiny · pretrained · box"
F.clear_leaderboard()
board = F.benchmark({PRETRAINED: F.SAM2Prompted("tiny", prompt="box")})
