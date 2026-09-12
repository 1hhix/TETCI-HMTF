import argparse
import os
import pickle
import time
import warnings

import torch

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from tqdm.auto import tqdm

from hmtf.data.utils import get_dataloader
from routefinder.models import RouteFinderBase
from routefinder.models.baselines.mtpomo import MTPOMO
from routefinder.models.baselines.mvmoe import MVMoE
from routefinder.envs import MTVRPEnv
from hmtf.models import HMTF

try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


def test(
    policy,
    td,
    env,
    num_augment=8,
    augment_fn="dihedral8",  # or symmetric. Default is dihedral8 for reported eval
    num_starts=None,
    device="cuda",
):

    costs_bks = td.get("costs_bks", None)

    with torch.inference_mode():
        with (
            torch.amp.autocast("cuda")
            if "cuda" in str(device)
            else torch.inference_mode()
        ):  # Use mixed precision if supported
            n_start = env.get_num_starts(td) if num_starts is None else num_starts

            if num_augment > 1:
                td = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)(td)

            # Evaluate policy
            out = policy(td, env, phase="test", num_starts=n_start, return_actions=True)

            # Unbatchify reward to [batch_size, num_augment, num_starts].
            reward = unbatchify(out["reward"], (num_augment, n_start))

            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (num_augment, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if num_augment > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                # If costs_bks is available, we calculate the gap to BKS
                if costs_bks is not None:
                    # Rewards are negative costs; normalize against the absolute BKS cost.
                    gap_to_bks = (
                        100
                        * (-max_aug_reward - torch.abs(costs_bks))
                        / torch.abs(costs_bks)
                    )
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": 69420})

            return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="checkpoints/50/hmtf/last.ckpt"
    )
    parser.add_argument(
        "--eal_checkpoint", default=None
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem name: cvrp, vrptw, etc. or all",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50,
        help="Problem size: 50, 100, for automatic loading",
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate. Defaults to all under data/{problem}/ dir",
        default=None,
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--remove-mixed-backhaul",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove mixed backhaul instances. Use --no-remove-mixed-backhaul to keep them.",
    )
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save results to results_EAl/main/{size}/{checkpoint",
    )
    parser.add_argument("--task_embedding", type=str, default="PLM/task_embeddings_USE.pt")
    parser.add_argument("--task_id", type=str, default="PLM/task_id.pt")


    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

    opts = parser.parse_args()
    device = torch.device(f"{opts.device}")

    if opts.datasets is not None:
        data_paths = opts.datasets.split(",")
    else:
        data_paths = []
        for root, _, files in os.walk("data"):
            for file in files:
                if "test" not in root:
                    continue
                if file.endswith(".npz"):
                    if opts.remove_mixed_backhaul and "m" in root:
                        continue
                    if str(opts.size) in file:
                        if file == "50.npz" or file == "100.npz":
                            data_paths.append(os.path.join(root, file))
        assert len(data_paths) > 0, "No datasets found. Check the data directory."
        data_paths = sorted(sorted(data_paths), key=lambda x: len(x))
        print(f"Found {len(data_paths)} datasets on the following paths: {data_paths}")

    print("Loading checkpoint from ", opts.checkpoint)
    if "mvmoe" in opts.checkpoint:
        BaseLitModule = MVMoE
    elif "mtpomo" in opts.checkpoint:
        BaseLitModule = MTPOMO
    elif "rf" in opts.checkpoint:
        BaseLitModule = RouteFinderBase
    else:
        BaseLitModule = HMTF


    model = BaseLitModule.load_from_checkpoint(
        opts.checkpoint, map_location="cpu", strict=False,
    )

    env = MTVRPEnv(load_solutions=False)
    policy = model.policy.to(device).eval()

    results = {}
    task_embedding=torch.load(opts.task_embedding, weights_only=True)
    task_id=torch.load(opts.task_id, weights_only=True)
    for dataset in tqdm(data_paths):
        print(f"Loading {dataset}")
        td_test = env.load_data(dataset)
        dataloader = get_dataloader(td_test, batch_size=opts.batch_size)

        start = time.time()
        res = []
        name=dataset.split('/')[1]
        for batch in tqdm(dataloader):
            batch['task_embedding']=task_embedding[task_id[name]].unsqueeze(0).expand_as(batch['task_embedding'])
            if 'm' in name:
                name_remove_m= name.replace('m', '')
                batch['task_embedding']=task_embedding[task_id[name_remove_m]].unsqueeze(0).expand_as(batch['task_embedding'])
            batch['capacity_route']=batch['capacity_route'].to(torch.bool)
            td_test = env.reset(batch).to(device)
            o = test(policy, td_test, env, device=device)
            res.append(o)
        out = {}
        out["max_aug_reward"] = torch.cat([o["max_aug_reward"] for o in res])
        out["gap_to_bks"] = torch.tensor([o["gap_to_bks"] for o in res],dtype=torch.float)

        inference_time = time.time() - start

        dataset_name = dataset.split("/")[-3].split(".")[0].upper()
        print(
            f"{dataset_name} | Cost: {-out['max_aug_reward'].mean().item():.3f} | Gap: {out['gap_to_bks'].mean().item():.3f}% | Inference time: {inference_time:.3f} s"
        )

        if results.get(dataset_name, None) is None:
            results[dataset_name] = {}
        results[dataset_name]["cost"] = -out["max_aug_reward"].mean().item()
        results[dataset_name]["gap"] = out["gap_to_bks"].mean().item()
        results[dataset_name]["inference_time"] = inference_time

    if opts.save_results:
        checkpoint_name = opts.checkpoint.split("/")[1].split(".")[0]
        savedir = f"results/{checkpoint_name}/{opts.size}/"
        os.makedirs(savedir, exist_ok=True)
        pickle.dump(results, open(savedir + checkpoint_name + ".pkl", "wb"))
    print("\033[0;37;42m DONE\033[0m")
