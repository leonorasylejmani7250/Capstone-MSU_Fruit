# Step 5: count metrics for the detector alone and for detector → SAM 2.1 (same photos, same hand counts).
def count_metrics(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    err = pred - truth
    return {
        "photos": len(pred),
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "bias (predicted − true)": err.mean(),
        "exact count (%)": (err == 0).mean() * 100,
        "clusters predicted": pred.sum(),
        "clusters by hand": truth.sum(),
    }

ok = df[df["status"] == "OK"]
sam2_name = f"SAM 2.1 fine-tuned ({FINETUNED.stem.split(F.SAM2_SIZE + '_', 1)[-1]})" if FINETUNED else "SAM 2.1 pretrained"
metrics = pd.DataFrame({
    "YOLO26 boxes": count_metrics(ok["yolo"], ok["truth"]),
    f"YOLO26 → {sam2_name}": count_metrics(ok["sam2"], ok["truth"]),
}).round(3)
metrics.to_csv(F.RESULTS / "count_metrics.csv")
display(metrics)
if len(ok) < len(df):
    display(df[df["status"] != "OK"][["image", "status"]])
