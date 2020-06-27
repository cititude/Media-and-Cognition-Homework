import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from terminaltables import  AsciiTable
from models import *
from utils.datasets import *
from utils.utils import *
import sys
from utils.torch_utils import *

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'



# Hyperparameters of finetuned model on coco
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 8.88,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# argparser
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)  
parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
parser.add_argument('--data', type=str, default='data/traffic.data', help='*.data path')
parser.add_argument('--img_size', type=int, default=608)
parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='initial weights path')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--maxdata',default=np.inf,type=int)
parser.add_argument('--lr',default=0.01,type=float)
parser.add_argument('--fix_seed',action="store_true",help='fix all seed to zero')
parser.add_argument('--record_dir',default="./result")
parser.add_argument('--record_file',default="result.txt")
parser.add_argument('--fl_gamma',default=1.5,type=float,help="focal_loss gamma")
parser.add_argument('--save_interval',default=10,type=int)
parser.add_argument('--val_interval',default=10,type=int)
parser.add_argument("--only_val",action="store_true")
parser.add_argument('--only_test',action="store_true")
parser.add_argument('--nms_conf_thresh',default=0.05,type=float)
parser.add_argument('--nms_iou_thresh',default=0.05,type=float)
parser.add_argument('--pred_file',default="pred.json")
opt = parser.parse_args()

print(AsciiTable([[key,vars(opt)[key]] for key in vars(opt)]).table)
device = torch_utils.select_device(opt.device, batch_size=opt.batch_size)
cfg=opt.cfg
data=opt.data
maxdata=opt.maxdata
opt.weights = last if opt.resume else opt.weights
batch_size=opt.batch_size
img_size=opt.img_size
epochs = opt.epochs  
weights = opt.weights  # initial training weights, can be .
nms_conf_thresh=opt.nms_conf_thresh
nms_iou_thresh=opt.nms_iou_thresh

# set print results to both terminal and record_file
record_dir=opt.record_dir
record_file=opt.record_file
record_file=os.path.join(record_dir,record_file)
if not os.path.exists(record_dir):
    os.mkdir(record_dir)
sys.stdout=Logger(record_file)

#update hyp
hyp["lr0"]=opt.lr
hyp["fl_gamma"]=opt.fl_gamma


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=608,
         conf_thres=0.2,
         iou_thres=0.05,  # for nms
         save_json=False,
         model=None,
         dataloader=None,
         path="pred15.json"):
    pred_path=os.path.join(record_dir,path)
    device = next(model.parameters()).device  # get model device

    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    path = data['test']  # path to test images
    names = load_classes(data['names'])  # class names
    predictions={}
    model.eval()
    scale=2048/img_size
    for batch_i, (imgs, _, paths, shapes) in enumerate(tqdm(dataloader)):
        imgs = imgs.to(device).float()
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        with torch.no_grad():
            # Run model
            inf_out, _ = model(imgs)  # inference and training outputs
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres,multi_label=False)  # nms
        for si, pred in enumerate(output):
            objects={}
            objects["objects"]=[]
            if pred is None:
                predictions[os.path.splitext(os.path.basename(paths[si]))[0]]=objects
                continue
            else:
                clip_coords(pred, (height, width))
                for j in range(len(pred)):
                    obj={}
                    obj['bbox']={'xmax':float(pred[j][2]*scale),'xmin':float(pred[j][0]*scale),'ymax':float(pred[j][3]*scale),'ymin':float(pred[j][1]*scale)}
                    obj['category']=names[int(pred[j][5])]
                    obj['score']=pred[j][4].item()
                    objects["objects"].append(obj)
            predictions[os.path.splitext(os.path.basename(paths[si]))[0]]=objects
    predictions={"imgs":predictions}
    json.dump(predictions,fp=open(pred_path,"w"))
    return None
           

def validate(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=608,
         conf_thres=0.3,
         iou_thres=0.05,  # for nms
         save_json=False,
         model=None,
         dataloader=None):

    # Initialize/load model and set device
    device = next(model.parameters()).device  # get model device

    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    model.eval()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float()
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            inf_out, train_out = model(imgs)  # inference and training outputs
            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres,multi_label=False)  # nms
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))


    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps

def train():
    global batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)

    maxdata=opt.maxdata  # load data number(if not debug, set as np.inf)


    record_dir=opt.record_dir
    record_file=opt.record_file
    record_file=os.path.join(record_dir,record_file)

    if opt.fix_seed:  # fix seed
        init_seeds()
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    print("Start loading model")
    model = Darknet(cfg).to(device)
    print("Model loaded successfully")

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  
    optimizer.add_param_group({'params': pg2})  
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0

    if weights.endswith(".weights"): # only load darknet backbone
        load_darknet_weights(model, weights)
    else:
        model.load_state_dict(torch.load(weights)["model"])

    # lr_scheduler
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine lr schedule
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  

    # Dataset
    print("Start loading training data")
    dataset = TrafficDataset(data_dir=train_path, inp_dim=img_size, mode="train",maxdata=maxdata)

    # Dataloader
    batch_size = min(batch_size, len(dataset))
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw=0
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=True,  
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader
    print("Start loading validating data")
    valloader = torch.utils.data.DataLoader(TrafficDataset(data_dir=test_path, inp_dim=img_size, mode="val",maxdata=maxdata),  
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    # Model parameters
    model.nc = nc  
    model.hyp = hyp  
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  

    # Start training
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float()
            targets = targets.to(device)

            # Burn-in
            if ni <= n_burn * 2:
                model.gr = np.interp(ni, [0, n_burn * 2], [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)

                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            loss *= batch_size / 64  # scale loss
            loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.5g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

        sys.stdout.flush()
            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        if (epoch%opt.val_interval==0 or final_epoch) and (not opt.notest):  # Calculate mAP
            results, maps = validate(cfg,
                                data,
                                batch_size=batch_size,
                                img_size=img_size,
                                model=model,
                                save_json=final_epoch,
                                dataloader=valloader,
                                conf_thres=nms_conf_thresh,
                                iou_thres=nms_iou_thresh)

        print(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        if final_epoch or ((epoch+1) % opt.save_interval)==0:
            chkpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        'optimizer': None if final_epoch else optimizer.state_dict()}
            torch.save(chkpt,os.path.join(record_dir,"chkpt_{}.pt".format(epoch)))
            # Save last, best and delete
            if (best_fitness == fi) and not final_epoch:
                torch.save(chkpt, best)
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    # only test 
    if opt.only_test:
        test_dataset=TrafficDataset(mode="test",inp_dim=img_size,maxdata=maxdata)
        test_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,pin_memory=True,collate_fn=test_dataset.collate_fn)
        model=Darknet(cfg).to(device)
        model.load_state_dict(torch.load(opt.weights)["model"])
        test(cfg,data,model=model,dataloader=test_dataloader,conf_thres=nms_conf_thresh,iou_thres=nms_iou_thresh,img_size=img_size,path=opt.pred_file)
        exit()
    
    if opt.only_val:
        val_dataset=TrafficDataset(mode="val",inp_dim=img_size,maxdata=maxdata)
        val_dataloader=torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,pin_memory=True,collate_fn=val_dataset.collate_fn)
        model=Darknet(cfg).to(device)
        model.load_state_dict(torch.load(opt.weights)["model"])
        validate(cfg,data,model=model,dataloader=val_dataloader,conf_thres=nms_conf_thresh,iou_thres=nms_iou_thresh,img_size=img_size)
        exit()

    tb_writer = None
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(record_dir)
    train()  # train normally
