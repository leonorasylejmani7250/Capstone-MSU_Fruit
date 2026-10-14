#@title Step 5 · Fine-tune SAM 2.1 on your labelled training clips  { display-mode: "form" }
#@markdown Uses Meta's own training code. `TRAIN_STEPS` is the main dial: more steps = more adaptation and more
#@markdown risk of memorising a small dataset, so always compare with the pretrained model on the val clips.
RUN_NAME      = "grapes_v1"   #@param {type:"string"}
TRAIN_STEPS   = 1500          #@param {type:"integer"}
LEARNING_RATE = 5e-6          #@param {type:"number"}
TRAIN_ENCODER = "auto"        #@param ["auto", "no", "yes"]
#@markdown `TRAIN_ENCODER`: "no" freezes the image backbone and trains only the mask decoder and memory
#@markdown (fits a 16 GB T4). "yes" trains everything (needs roughly 40 GB or more). "auto" picks by GPU size.
#@markdown Long runs fit better in an HPCC batch job (`hpcc/train.sb`). Give it the same RUN_NAME and this cell
#@markdown just picks up the finished weights from the project's `models` folder.

if DEMO:
    TRAIN_STEPS = min(TRAIN_STEPS, 150)       # enough to prove the plumbing, not enough to judge fine-tuning
FINETUNED = finetune(RUN_NAME, steps=TRAIN_STEPS, lr=LEARNING_RATE, encoder=TRAIN_ENCODER)
plot_training(RUN_NAME)
