# Step 4: fine-tune SAM 2.1 for 20 steps with Meta's training code, then plot the training loss.
ckpt = F.finetune("demo_run", size="tiny", steps=20)
F.plot_training("demo_run")
