PRETRAINED = "SAM 2.1 tiny · pretrained · box"
F.clear_leaderboard()
board = F.benchmark({PRETRAINED: F.SAM2Prompted("tiny", prompt="box")})
