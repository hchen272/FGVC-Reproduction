#!/usr/bin/env python3
"""Author P2P-Net CUB single-GPU adaptation: preflight, 50/300-epoch runs, official test, timings.
Preserves author algorithm/recipe. Main checkpoint is fixed final epoch.
No CLIP/GBDD/bbox masks, no replacement losses. All artifacts stay in the project.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, os, sys, json, time, hashlib, csv, random, traceback, gc

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def save_pt(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def status(args, state, stage, **kwargs):
    obj = dict(run_id=args.out.name, display_name=f"P2P-Net RN50 · seed{args.seed} · {args.epochs} epochs · single GPU",
               model="P2P-Net / ResNet50", dataset="CUB-200-2011", input_mode="full",
               image_size=448, seed=args.seed, state=state, stage=stage,
               epochs=args.epochs, updated_at=now(), **kwargs)
    atomic(args.out / "status.json", obj)

def component_floats(values):
    return {k: float(v.detach().item()) if torch.is_tensor(v) else float(v) for k, v in values.items()}

def optimizer_for(model):
    return torch.optim.SGD([
        {"params": [p for n,p in model.named_parameters() if "backbone" not in n], "lr": .002},
        {"params": model.backbone.parameters(), "lr": .0002}],
        lr=.002, momentum=.9, weight_decay=.0005)

def evaluate(args, author, model, dataset, rows, history, epoch, export_logits=False):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                  num_workers=args.workers, pin_memory=True, drop_last=False)
    all_logits, all_labels = [], []
    head_correct = [0] * 4
    started = time.monotonic()
    with torch.no_grad():
        for b, (images, labels) in enumerate(tqdm(loader, desc=f"test epoch{epoch}", dynamic_ncols=True), 1):
            x, y = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(x, is_train=False)
            heads = outputs[:4]
            logits = sum(heads)
            if not torch.isfinite(logits).all(): raise FloatingPointError("Nonfinite test logits")
            for k, head in enumerate(heads):
                head_correct[k] += int((head.argmax(1) == y).sum())
            all_logits.append(logits.cpu().numpy()); all_labels.append(y.cpu().numpy())
            if b == 1 or b % 10 == 0 or b == len(loader):
                status(args, "running", "test", epoch=epoch, completed_epochs=epoch,
                       batch=b, batches=len(loader), history=history,
                       metrics={"elapsed_seconds": time.monotonic()-args.start,
                                "samples_per_second": min(b*args.eval_batch_size,len(dataset))/max(time.monotonic()-started,.01)})
    logits = np.concatenate(all_logits); labels = np.concatenate(all_labels)
    assert len(labels) == 5794 and np.array_equal(labels, np.array([r["label"] for r in rows]))
    order = np.argsort(-logits, axis=1, kind="stable")
    gt_rank = 1 + np.argmax(order == labels[:, None], axis=1)
    pred = order[:, 0]
    support = np.bincount(labels, minlength=200)
    cm = np.zeros((200,200),dtype=np.int64); np.add.at(cm,(labels,pred),1)
    c = np.diag(cm).astype(float)
    class_acc = c/support
    denom = support+cm.sum(0)
    mf1 = np.mean(np.divide(2*c,denom,out=np.zeros_like(c),where=denom>0))
    z = logits.astype(np.float64); maximum = z.max(1)
    ce = maximum + np.log(np.exp(z-maximum[:,None]).sum(1)) - z[np.arange(len(labels)),labels]
    result = {"loss": float(ce.mean()), "top1": float((gt_rank == 1).mean()),
              "top3": float((gt_rank <= 3).mean()), "top5": float((gt_rank <= 5).mean()),
              "macro_f1": float(mf1), "mean_per_class_accuracy": float(class_acc.mean()),
              "samples": len(labels), "head_top1": [x/len(labels) for x in head_correct],
              "seconds": time.monotonic()-started}
    if export_logits:
        np.savez_compressed(args.out/"test_logits.npz", logits=logits, labels=labels,
            cub_image_ids=np.array([r["cub_image_id"] for r in rows]), sample_ids=np.array([r["sample_id"] for r in rows]),
            classes=np.array(args.classes))
        np.savez_compressed(args.out/"test_confusion.npz", confusion=cm, classes=np.array(args.classes))
        with (args.out/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["cub_image_id","sample_id","image","gt","pred","gt_rank"])
            for j,r in enumerate(rows):
                w.writerow([r["cub_image_id"],r["sample_id"],r["image"],args.classes[int(labels[j])],
                            args.classes[int(pred[j])],int(gt_rank[j])])
        atomic(args.out/"classification_report.json", [
            {"species": name,"test_samples":int(support[i]),"correct":int(c[i]),"top1":float(class_acc[i])}
            for i,name in enumerate(args.classes)])
    return result

def checkpoint(model, optimizer, generator, epoch, history, identity):
    return {"schema":"p2p_author_single_gpu_v1", "run_identity":identity, "epoch":epoch, "model":model.state_dict(),
            "optimizer":optimizer.state_dict(), "pose_state":runtime.pose_state(model),
            "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
            "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
            "loader_generator":generator.get_state(), "history":history}

def preflight(args, author, trainset, source_hash):
    seed_all(args.seed)
    model = runtime.make_model(author,args.weights,"cuda")
    optimizer = optimizer_for(model)
    g = torch.Generator().manual_seed(args.seed)
    loader=torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,
          num_workers=args.workers,pin_memory=True,drop_last=True,generator=g)
    torch.cuda.reset_peak_memory_stats()
    timings, losses, gradient_checks = [], [], {}
    for j,(images,labels) in enumerate(tqdm(loader,desc="P2P train-only preflight",dynamic_ncols=True),1):
        if j > args.preflight_steps: break
        t=time.monotonic()
        model.train();optimizer.zero_grad(set_to_none=True)
        x,y=images.cuda(non_blocking=True),labels.cuda(non_blocking=True)
        loss, components, logits=runtime.forward_loss(author,model,x,y)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite preflight loss")
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("Nonfinite preflight gradients")
        watched=["backbone.conv1.weight","proposal_net.down1.weight","reg_mlp1.0.weight","classifier_concat.1.weight"]
        named=dict(model.named_parameters())
        for n in watched:
            p=named[n]
            gradient_checks[n]=gradient_checks.get(n,False) or (p.grad is not None and float(p.grad.norm())>0)
        optimizer.step();torch.cuda.synchronize()
        timings.append(time.monotonic()-t);losses.append(float(loss))
        status(args,"running","preflight",epoch=0,completed_epochs=0,batch=j,batches=args.preflight_steps,
          metrics={"batch_loss":float(loss),"gpu_memory_mb":torch.cuda.max_memory_allocated()/1048576,
                   "elapsed_seconds":time.monotonic()-args.start})
    if len(timings) != args.preflight_steps or not all(gradient_checks.values()):
        raise ValueError("Preflight steps/gradient coverage failed")
    model.eval()
    before=runtime.pose_state(model)
    with torch.no_grad():
        native=sum(model(x,is_train=False)[:4]); fast=runtime.fast_logits(model,x)
    difference=float((native-fast).abs().max())
    if not torch.allclose(native,fast,rtol=1e-5,atol=1e-5):
        raise ValueError("Global-only inference differs from author outputs")
    assert np.array_equal(before["centers"],runtime.pose_state(model)["centers"])
    assert before["count"]==runtime.pose_state(model)["count"]
    record={"state":"passed","created_at":now(),"seed":args.seed,"samples_from":"official_train_only",
            "steps":len(timings),"batch_size":args.batch_size,"precision":"FP32",
            "gradient_checks":gradient_checks,"losses":losses,"source_sha256":source_hash,
            "weights_sha256":sha(args.weights),"global_logits_parity":True,"max_abs_logit_difference":difference,
            "pose_updates_train_only":True,"parameters":sum(p.numel() for p in model.parameters()),
            "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(),
            "mean_step_seconds_after_warmup":float(np.mean(timings[2:])),
            "estimated_train_only_seconds":float(np.mean(timings[2:]))*(len(trainset)//args.batch_size)*args.epochs}
    atomic(args.out/"preflight.json",record)
    print("PREFLIGHT_PASSED",json.dumps(record),flush=True)
    del model,optimizer,loss,native,fast;gc.collect();torch.cuda.empty_cache()
    return record

def benchmark(args, model, testset):
    model.eval(); images=[testset[i][0].unsqueeze(0) for i in range(20)]
    xs=[x.cuda() for x in images]
    forward={"author_full_forward":lambda x:sum(model(x,is_train=False)[:4]),
             "global_only_equivalent":lambda x:runtime.fast_logits(model,x)}
    stats={}
    parity=0.
    with torch.no_grad():
        for x in xs:
            a=forward["author_full_forward"](x); b=forward["global_only_equivalent"](x)
            parity=max(parity,float((a-b).abs().max()))
            if not torch.allclose(a,b,rtol=1e-5,atol=1e-5):raise ValueError("Trained global inference parity failed")
        for name,fn in forward.items():
            for i in range(20):fn(xs[i%len(xs)])
            torch.cuda.synchronize()
            times=[]
            for i in range(100):
                torch.cuda.synchronize();t=time.perf_counter();fn(xs[i%len(xs)]);torch.cuda.synchronize()
                times.append(1000*(time.perf_counter()-t))
            stats[name]={"classifier_only_mean_ms":float(np.mean(times)),
                         "classifier_only_p95_ms":float(np.percentile(times,95))}
            times=[]
            for i in range(100):
                torch.cuda.synchronize();t=time.perf_counter()
                x=testset[i%len(xs)][0].unsqueeze(0).cuda();fn(x);torch.cuda.synchronize()
                times.append(1000*(time.perf_counter()-t))
            stats[name].update({"preprocess_plus_classifier_mean_ms":float(np.mean(times)),
                               "preprocess_plus_classifier_p95_ms":float(np.percentile(times,95))})
    record={"precision":"FP32", "gpu":torch.cuda.get_device_name(),"batch_size":1,
       "warmup_iterations":20,"iterations":100,"unique_samples":20,
       "includes":"RN50 full image-to-label computation, plus redundant parts for native author path",
       "excluded":["YOLO","mask generation","network transfer","video decoding"],
       "input_size":448,"max_abs_logit_difference":parity,"global_logits_parity":True,**stats}
    atomic(args.out/"benchmark.json",record)
    return record

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument("--data-root",type=Path);p.add_argument("--source-root",type=Path)
    p.add_argument("--weights",type=Path);p.add_argument("--out",type=Path)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch-size",type=int,default=16);p.add_argument("--eval-batch-size",type=int,default=8)
    p.add_argument("--workers",type=int,default=8);p.add_argument("--preflight-steps",type=int,default=10)
    p.add_argument("--mode",choices=["run","preflight"],default="run")
    p.add_argument("--resume-file",type=Path)
    args=p.parse_args()
    args.root=args.root.resolve()
    args.source_root=(args.source_root or args.root/"third_party/P2P-Net").resolve()
    args.weights=(args.weights or args.root/"weights/resnet50-19c8e357.pth").resolve()
    args.data_root=(args.data_root or args.root/"data/CUB_200_2011").resolve()
    args.out=(args.out or args.root/f"runs/p2p/seed{args.seed}").resolve()
    args.out.relative_to(args.root)
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/"completed.json").exists():raise ValueError("Completed run exists; refusing to overwrite")
    if (args.out/"history.json").exists() and not args.resume_file:
        raise ValueError("Training history exists; pass explicit resume file or new output directory")
    if args.batch_size!=16 or args.epochs not in (50,300) or args.preflight_steps<4:
        raise ValueError("Author-recipe runs require batch16, epochs50 or300 and at least4 preflight steps")
    args.start=time.monotonic()
    os.environ["TORCH_HOME"]=str(args.root/"weights/.torch_cache")
    os.environ["TMPDIR"]=str(args.root/".runtime/p2p_tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS","4");os.environ.setdefault("MKL_NUM_THREADS","4")
    global torch,np,tqdm,runtime
    import torch, numpy as np
    from tqdm import tqdm
    import p2p_runtime as runtime
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    try:
        author=runtime.load_author(args.source_root)
        args.classes,train_rows,test_rows=runtime.raw_metadata(args.data_root)
        trainset=author.dataset.CUB(str(args.data_root),is_train=True)
        testset=author.dataset.CUB(str(args.data_root),is_train=False)
        assert len(trainset)==5994 and len(testset)==5794
        source_hash={x.name:sha(x) for x in args.source_root.glob("*.py")}
        source_hash.update({"adapter":sha(Path(runtime.__file__)),"trainer":sha(Path(__file__))})
        assert sha(args.weights).startswith("19c8e357")
        cfg={"schema":"p2p_author_recipe_single_gpu_v1","method":"P2P-Net","seed":args.seed,
             "gpu":torch.cuda.get_device_name(),"source_commit":"400ad83459634597ed1f37ec6b9dad1849c5fee6",
             "source_sha256":source_hash,"source_url":"https://github.com/yangxh11/P2P-Net",
             "backbone":"author Resnet.py RN50","pretraining":"ImageNet1k legacy supervised RN50 (19c8e357), no GBDD",
             "weights_path":Path(os.path.relpath(args.weights,args.root)).as_posix(),"weights_sha256":sha(args.weights),
             "metadata_sha256":{n:sha(args.data_root/n) for n in ["images.txt","image_class_labels.txt","train_test_split.txt","classes.txt"]},
             "sample_counts":{"official_train":5994,"train_samples_updated_per_epoch":5984,"official_test":5794},
             "image_size":448,"part_image_size":224,"topn":4,"epochs":args.epochs,"batch_size":args.batch_size,
             "eval_batch_size":args.eval_batch_size,"num_workers":args.workers,"precision":"FP32 (TF32 disabled)",
             "train_transform":"author CUB: Resize550x550, RandomCrop448 padding8, horizontal flip, normalize .5/.5",
             "test_transform":"author CUB: Resize550x550, CenterCrop448, normalize .5/.5; no bbox",
             "optimizer":{"name":"SGD","head_lr":.002,"backbone_lr":.0002,"momentum":.9,"weight_decay":.0005},
             "loss":"author smooth_CE peak [.7,.8,.9,1], image+part+ranking+.1 regularization/stage",
             "prediction_rule":"sum original-image y1+y2+y3+y4 logits",
             "main_selection_policy":f"fixed_last_epoch{args.epochs}",
             "paper_epochs":300,"recipe_variant":"50epoch_short_run" if args.epochs==50 else "author300_single_gpu",
             "author_test_max":"descriptive history only, not selection of this experiment's main checkpoint",
             "retained_author_behaviors":["graph_assign cumulative-permutation behavior",f"author cosine epoch%{args.epochs} resets LR at final epoch",
                                          "BN and Python PR graph-center updates","tail partial train batch skipped"],
             "single_gpu_differences":["original code uses four-card DataParallel with global batch16 (perGPU BN4), singleGPU BN16",
                                        "singleGPU PR updates remove shared-object multithread race",
                                        "workers8 rather than16","torch.no_grad for evaluation"],
             "readme_batch_discrepancy":"README says64; paper section4.1 and executable trainer use global16",
             "author_best_test_selection_difference":"Author selects highest test accuracy; primary here fixed final epoch consistent with prior experiments.",
             "compatibility":"Only exact np.int to np.int64 in isolated effective-source cache; unused distutils trainer bypassed",
             "paper_reported_top1":.902,"initial_state":"no previous experiment checkpoint"}
        import importlib.metadata
        cfg["dependency_versions"]={n:importlib.metadata.version(n) for n in
                                   ["torch","torchvision","numpy","imageio","Pillow","scipy","tqdm"]}
        cfg["runtime_compatibility_metadata"]=author.metadata
        identity_fields={k:cfg[k] for k in ["seed","epochs","batch_size","eval_batch_size","num_workers",
                       "source_commit","source_sha256","weights_sha256","metadata_sha256","main_selection_policy"]}
        args.identity=hashlib.sha256(json.dumps(identity_fields,sort_keys=True).encode()).hexdigest()
        cfg["run_identity"]=args.identity
        if args.resume_file:
            previous=json.loads((args.out/"config.json").read_text())
            if previous.get("run_identity")!=args.identity:raise ValueError("Resume configuration identity mismatch")
        atomic(args.out/"config.json",cfg);atomic(args.out/"classes.json",args.classes)
        with (args.out/"dataset_manifest.jsonl").open("w",encoding="utf-8") as f:
            for r in train_rows+test_rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
        if not args.resume_file:
            pre=preflight(args,author,trainset,source_hash)
            if args.mode=="preflight":
                status(args,"completed","preflight",epoch=0,completed_epochs=0,metrics={"preflight_passed":True})
                return
        seed_all(args.seed)
        model=runtime.make_model(author,args.weights,"cuda");optimizer=optimizer_for(model)
        generator=torch.Generator().manual_seed(args.seed)
        history=[];start_epoch=1
        if args.resume_file:
            receipt=json.loads((args.out/"preflight.json").read_text())
            if receipt["state"]!="passed" or receipt["source_sha256"]!=source_hash:raise ValueError("Preflight source identity mismatch")
            ck=torch.load(args.resume_file,map_location="cpu",weights_only=False)
            if ck.get("run_identity") != args.identity:raise ValueError("Resume checkpoint identity mismatch")
            model.load_state_dict(ck["model"],strict=True);optimizer.load_state_dict(ck["optimizer"])
            runtime.load_pose_state(model,ck["pose_state"]);generator.set_state(ck["loader_generator"])
            random.setstate(ck["python_rng"]);np.random.set_state(ck["numpy_rng"])
            torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
            history=ck["history"];start_epoch=ck["epoch"]+1
        loader=torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,
                  num_workers=args.workers,pin_memory=True,drop_last=True,generator=generator)
        torch.cuda.reset_peak_memory_stats()
        training_started=time.monotonic()
        for epoch in range(start_epoch,args.epochs+1):
            model.train()
            optimizer.param_groups[0]["lr"]=author.utils.cosine_anneal_schedule(epoch,args.epochs,.002)
            optimizer.param_groups[1]["lr"]=author.utils.cosine_anneal_schedule(epoch,args.epochs,.0002)
            sums={};correct1=correct3=correct5=seen=0;epoch_start=time.monotonic()
            for b,(images,labels) in enumerate(tqdm(loader,desc=f"P2P epoch{epoch}/{args.epochs}",dynamic_ncols=True),1):
                x,y=images.cuda(non_blocking=True),labels.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss,components,logits=runtime.forward_loss(author,model,x,y)
                if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite train loss at {epoch}/{b}")
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError(f"Nonfinite gradients at {epoch}/{b}")
                optimizer.step()
                n=len(y);seen+=n
                values={"loss":float(loss),**component_floats(components)}
                for k,v in values.items():sums[k]=sums.get(k,0)+n*v
                top=logits.detach().topk(5,dim=1).indices
                correct1+=int((top[:,0]==y).sum());correct3+=int((top[:,:3]==y[:,None]).any(1).sum())
                correct5+=int((top==y[:,None]).any(1).sum())
                if b==1 or b%5==0 or b==len(loader):
                    elapsed=time.monotonic()-epoch_start
                    done=(epoch-start_epoch)*len(loader)+b
                    eta=(time.monotonic()-training_started)/max(done,1)*((args.epochs-start_epoch+1)*len(loader)-done)
                    status(args,"running","train",epoch=epoch,completed_epochs=epoch-1,batch=b,batches=len(loader),history=history,
                        metrics={"train":{"loss":sums["loss"]/seen,"top1":correct1/seen,"top3":correct3/seen,"top5":correct5/seen},
                                 "batch_loss":values["loss"],"elapsed_seconds":time.monotonic()-args.start,
                                 "samples_per_second":seen/max(elapsed,.01),"eta_seconds":eta,"eta_estimated":True,
                                 "gpu_memory_mb":torch.cuda.max_memory_allocated()/1048576,
                                 "learning_rate":optimizer.param_groups[0]["lr"]})
            assert seen==5984
            row={"epoch":epoch,"train":{"loss":sums["loss"]/seen,"top1":correct1/seen,"top3":correct3/seen,
                  "top5":correct5/seen,"samples":seen,"seconds":time.monotonic()-epoch_start,
                  "components":{k:v/seen for k,v in sums.items() if k!="loss"}},
                 "learning_rate":optimizer.param_groups[0]["lr"],"updated_at":now()}
            # Exact author's evaluation schedule; never choose main checkpoint on test.
            if epoch<5 or epoch%10==0:
                row["test"]=evaluate(args,author,model,testset,test_rows,history,epoch,export_logits=(epoch==args.epochs))
            history.append(row);atomic(args.out/"history.json",history)
            with (args.out/"metrics.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(row)+"\n")
            if epoch%10==0 or epoch in (args.epochs-1,args.epochs):
                save_pt(args.out/"last.pt",checkpoint(model,optimizer,generator,epoch,history,args.identity))
                if epoch==args.epochs-1:save_pt(args.out/f"epoch{args.epochs-1}.pt",checkpoint(model,optimizer,generator,epoch,history,args.identity))
            print("EPOCH_RESULT",json.dumps(row),flush=True)
        save_pt(args.out/"final.pt",checkpoint(model,optimizer,generator,args.epochs,history,args.identity))
        final=history[-1]["test"]
        status(args,"running","benchmark",epoch=args.epochs,completed_epochs=args.epochs,history=history,
               metrics={"train":history[-1]["train"],"test":final,"elapsed_seconds":time.monotonic()-args.start})
        bench=benchmark(args,model,testset)
        best=max((r for r in history if "test" in r),key=lambda r:r["test"]["top1"])
        result={"state":"completed","run_id":args.out.name,"model":"P2P-Net / RN50","seed":args.seed,
                "selected_epoch":args.epochs,"selection_policy":"fixed_last_epoch","metrics":final,"benchmark":bench,
                "parameters":sum(p.numel() for p in model.parameters()),
                "author_test_max_descriptive_only":{"epoch":best["epoch"],"top1":best["test"]["top1"]},
                "training_seconds_this_session":time.monotonic()-training_started}
        atomic(args.out/"result.json",result)
        artifacts={n:{"sha256":sha(args.out/n),"path":n} for n in ["final.pt","last.pt","config.json","classes.json",
                    "dataset_manifest.jsonl","history.json","test_logits.npz","test_confusion.npz","predictions.csv",
                    "classification_report.json","benchmark.json","result.json","preflight.json"]}
        atomic(args.out/"completed.json",{**result,"completed_at":now(),"artifacts":artifacts})
        status(args,"completed","completed",epoch=args.epochs,completed_epochs=args.epochs,history=history,metrics={"train":history[-1]["train"],"test":final})
        print("P2P_COMPLETED",json.dumps(result),flush=True)
    except Exception as exc:
        status(args,"failed","failed",error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise

if __name__=="__main__":
    main()
