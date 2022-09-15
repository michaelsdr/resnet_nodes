import torch

n_tries = 5
for i in range(1, n_tries):

    for depth in [2, 4, 8, 16, 32, 64]:

        c = torch.load(
            "checkpoint/ckpt_%s_heun_False_backprop_False_seed_%s.pth" % (depth, i)
        )
        log_dicts = c["log_dict"]
        torch.save(
            log_dicts, "results/%s_heun_False_backprop_False_seed_%s.to" % (depth, i)
        )

        c = torch.load(
            "checkpoint/ckpt_%s_heun_True_backprop_False_seed_%s.pth" % (depth, i)
        )
        log_dicts = c["log_dict"]
        torch.save(
            log_dicts, "results/%s_heun_True_backprop_False_seed_%s.to" % (depth, i)
        )

        c = torch.load(
            "checkpoint/ckpt_%s_heun_False_backprop_True_seed_%s.pth" % (depth, i)
        )
        log_dicts = c["log_dict"]
        torch.save(
            log_dicts, "results/%s_heun_False_backprop_True_seed_%s.to" % (depth, i)
        )

        c = torch.load(
            "checkpoint/ckpt_%s_heun_True_backprop_True_seed_%s.pth" % (depth, i)
        )
        log_dicts = c["log_dict"]
        torch.save(
            log_dicts, "results/%s_heun_True_backprop_True_seed_%s.to" % (depth, i)
        )
