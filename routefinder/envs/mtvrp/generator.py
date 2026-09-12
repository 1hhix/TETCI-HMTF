from typing import Callable, Tuple, Union

import torch

from rl4co.data.utils import save_tensordict_to_npz
from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.ops import get_distance
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform
import random
log = get_pylogger(__name__)


def get_vehicle_capacity(num_loc: int) -> int:
    """Capacity should be 30 + num_loc/5 if num_loc > 20 as described in Liu et al. 2024 (POMO-MTL).
    For every N over 1000, we add 1 of capacity every 33.3 nodes to align with Ye et al. 2024 (GLOP),
    i.e. 260 at 2K nodes, 350 at 5K nodes and 500 at 10K nodes.
    Note that this serves as a demand scaler.
    """
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) // 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return 30 + extra_cap



TASK_train={
    # 16 VRPs
    "cvrp":[False, False, False, False, True, False, False],
    "ovrp":[True, False, False, False, True, False, False],
    "vrpb":[False, False, False, True, True, False, False],
    "vrpl":[False, False, True, False, True, False, False],
    "vrptw":[False, True, False, False, True, False, False],
    "ovrptw":[True, True, False, False, True, False, False],
    "ovrpb":[True, False, False, True, True, False, False],
    "ovrpl":[True, False, True, False, True, False, False],
    "vrpbl":[False, False, True, True, True, False, False],
    "vrpbtw":[False, True, False, True, True, False, False],
    "vrpltw":[False, True, True, False, True, False, False],
    "ovrpbl":[True, False, True, True, True, False, False],
    "ovrpbtw":[True, True, False, True, True, False, False],
    "ovrpltw":[True, True, True, False, True, False, False],
    "vrpbltw":[False, True, True, True, True, False, False],
    "ovrpbltw":[True, True, True, True, True, False, False],

    # # 4 TSPs
    "tsp":[False, False, False, False, False, False, False],
    "otsp":[True, False, False, False, False, False, False],
    "tsptw":[False, True, False, False, False, False, False],
    "otsptw":[True, True, False, False, False, False, False],

    # # 20  half-asymmetric
    "hcvrp":[False, False, False, False, True, True, False],
    "hovrp":[True, False, False, False, True, True, False],
    "hvrpb":[False, False, False, True, True, True, False],
    "hvrpl":[False, False, True, False, True, True, False],
    "hvrptw":[False, True, False, False, True, True, False],
    "hovrptw":[True, True, False, False, True, True, False],
    "hovrpb":[True, False, False, True, True, True, False],
    "hovrpl":[True, False, True, False, True, True, False],
    "hvrpbl":[False, False, True, True, True, True, False],
    "hvrpbtw":[False, True, False, True, True, True, False],
    "hvrpltw":[False, True, True, False, True, True, False],
    "hovrpbl":[True, False, True, True, True, True, False],
    "hovrpbtw":[True, True, False, True, True, True, False],
    "hovrpltw":[True, True, True, False, True, True, False],
    "hvrpbltw":[False, True, True, True, True, True, False],
    "hovrpbltw":[True, True, True, True, True, True, False],
    "htsp":[False, False, False, False, False, True, False],
    "hotsp":[True, False, False, False, False, True, False],
    "htsptw":[False, True, False, False, False, True, False],
    "hotsptw":[True, True, False, False, False, True, False],
}
TASK_all={
    # 16 VRPs
    "cvrp":[False, False, False, False, True, False, False],
    "ovrp":[True, False, False, False, True, False, False],
    "vrpb":[False, False, False, True, True, False, False],
    "vrpl":[False, False, True, False, True, False, False],
    "vrptw":[False, True, False, False, True, False, False],
    "ovrptw":[True, True, False, False, True, False, False],
    "ovrpb":[True, False, False, True, True, False, False],
    "ovrpl":[True, False, True, False, True, False, False],
    "vrpbl":[False, False, True, True, True, False, False],
    "vrpbtw":[False, True, False, True, True, False, False],
    "vrpltw":[False, True, True, False, True, False, False],
    "ovrpbl":[True, False, True, True, True, False, False],
    "ovrpbtw":[True, True, False, True, True, False, False],
    "ovrpltw":[True, True, True, False, True, False, False],
    "vrpbltw":[False, True, True, True, True, False, False],
    "ovrpbltw":[True, True, True, True, True, False, False],

    # 4 TSPs
    "tsp":[False, False, False, False, False, False, False],
    "otsp":[True, False, False, False, False, False, False],
    "tsptw":[False, True, False, False, False, False, False],
    "otsptw":[True, True, False, False, False, False, False],

    # 20  half-asymmetric
    "hcvrp":[False, False, False, False, True, True, False],
    "hovrp":[True, False, False, False, True, True, False],
    "hvrpb":[False, False, False, True, True, True, False],
    "hvrpl":[False, False, True, False, True, True, False],
    "hvrptw":[False, True, False, False, True, True, False],
    "hovrptw":[True, True, False, False, True, True, False],
    "hovrpb":[True, False, False, True, True, True, False],
    "hovrpl":[True, False, True, False, True, True, False],
    "hvrpbl":[False, False, True, True, True, True, False],
    "hvrpbtw":[False, True, False, True, True, True, False],
    "hvrpltw":[False, True, True, False, True, True, False],
    "hovrpbl":[True, False, True, True, True, True, False],
    "hovrpbtw":[True, True, False, True, True, True, False],
    "hovrpltw":[True, True, True, False, True, True, False],
    "hvrpbltw":[False, True, True, True, True, True, False],
    "hovrpbltw":[True, True, True, True, True, True, False],
    "htsp":[False, False, False, False, False, True, False],
    "hotsp":[True, False, False, False, False, True, False],
    "htsptw":[False, True, False, False, False, True, False],
    "hotsptw":[True, True, False, False, False, True, False],


    # 16  finetune
    "vrpmb":[False, False, False, True, True, False, True],
    "ovrpmb":[True, False, False, True, True, False, True],
    "vrpmbl":[False, False, True, True, True, False, True],
    "vrpmbtw":[False, True, False, True, True, False, True],
    "ovrpmbl":[True, False, True, True, True, False, True],
    "ovrpmbtw":[True, True, False, True, True, False, True],
    "vrpmbltw":[False, True, True, True, True, False, True],
    "ovrpmbltw":[True, True, True, True, True, False, True],

    "hvrpmb":[False, False, False, True, True, True, True],
    "hovrpmb":[True, False, False, True, True, True, True],
    "hvrpmbl":[False, False, True, True, True, True, True],
    "hvrpmbtw":[False, True, False, True, True, True, True],
    "hovrpmbl":[True, False, True, True, True, True, True],
    "hovrpmbtw":[True, True, False, True, True, True, True],
    "hvrpmbltw":[False, True, True, True, True, True, True],
    "hovrpmbltw":[True, True, True, True, True, True, True],
}

class MTVRPGenerator(Generator):
    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        capacity: float = None,
        min_demand: int = 1,
        max_demand: int = 10,
        min_backhaul: int = 1,
        max_backhaul: int = 10,
        scale_demand: bool = True,
        max_time: float = 4.6,
        backhaul_ratio: float = 0.2,
        backhaul_class: int = 1,
        sample_backhaul_class: bool = False,
        max_distance_limit: float = 2.8,  # 2sqrt(2) ~= 2.8
        speed: float = 1.0,
        sample_problem=False,
        subsample=True,
        two_task=False,
        llm_embedding=False,
        llm_model='PLM/task_embeddings_USE',
        **kwargs,
    ) -> None:
        # Location distribution
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

        if capacity is None:
            capacity = get_vehicle_capacity(num_loc)
        self.capacity = capacity
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_backhaul = min_backhaul
        self.max_backhaul = max_backhaul
        self.scale_demand = scale_demand
        self.backhaul_ratio = backhaul_ratio
        assert backhaul_class in (
            1,
            2,
        ), "Backhaul class must be in [1, 2]. We don't use class 0 for efficiency since it is a subset"
        self.backhaul_class = backhaul_class
        self.sample_backhaul_class = sample_backhaul_class

        self.max_time = max_time
        self.max_distance_limit = max_distance_limit
        self.speed = speed
        self.subsample = subsample
        self.sample_problem=sample_problem
        self.task=TASK_all if sample_backhaul_class else TASK_train
        temp=torch.tensor(list(self.task.values()))
        binary_values_row_sum = temp.int().mv(torch.tensor([64,32,16, 8, 4, 2, 1], dtype=torch.int32))
        self.task_id={id.item():name for id,name in zip(binary_values_row_sum,self.task.keys())}
        self.two_task=two_task
        self.llm_embedding=llm_embedding
        self.llm_model=llm_model
    def sample_problem_mask(self,batch_size):
        if not self.sample_problem:
            sampled_keys = random.choices(list(self.task.keys()), k=batch_size//256+1)
            sampled_values = [self.task[key] for key in sampled_keys]
            return torch.tensor(sampled_values, dtype=torch.bool).repeat_interleave(256,dim=0)[:batch_size]
        if self.two_task:
            sampled_keys = random.choices(list(self.task.keys()), k=batch_size//128)
            sampled_values = [self.task[key] for key in sampled_keys]
            return torch.tensor(sampled_values, dtype=torch.bool).repeat_interleave(128,dim=0)
        else:
            sampled_keys = random.choices(list(self.task.keys()), k=batch_size)
            sampled_values = [self.task[key] for key in sampled_keys]
            return torch.tensor(sampled_values, dtype=torch.bool)
    def _generate(self, batch_size) -> TensorDict:
        # Locations
        locs = self.generate_locations(batch_size=batch_size, num_loc=self.num_loc)

        # Vehicle capacity (C, B) - applies to both linehaul and backhaul
        vehicle_capacity = torch.full(
            (*batch_size, 1), self.capacity, dtype=torch.float32
        )
        capacity_original = vehicle_capacity.clone()

        # linehaul demand / delivery (C) and backhaul / pickup demand (B)
        demand_linehaul, demand_backhaul = self.generate_demands(
            batch_size=batch_size, num_loc=self.num_loc
        )

        # Capacity (C)
        capacity_route= self.generate_capacity_route(shape=(*batch_size, 1))
        # Open (O)
        open_route = self.generate_open_route(shape=(*batch_size, 1))

        # Time windows (TW)
        speed = self.generate_speed(shape=(*batch_size, 1))
        time_windows, service_time = self.generate_time_windows(
            locs=locs,
            speed=speed,
        )

        # Distance limit (L)
        distance_limit = self.generate_distance_limit(shape=(*batch_size, 1), locs=locs)

        # scaling
        if self.scale_demand:
            demand_backhaul /= vehicle_capacity
            demand_linehaul /= vehicle_capacity
            vehicle_capacity /= vehicle_capacity

        # task_embedding
        task_embedding = torch.zeros(*batch_size, 512)

        asymmetric_route = self.generate_asymmetric_route(shape=(*batch_size, 1))

        keep_mask=self.sample_problem_mask(batch_size=batch_size[0])
        backhaul_class = torch.full((*batch_size, 1), self.backhaul_class, dtype=torch.float32)
        if self.sample_backhaul_class: # finetune/zeroshot
            assert len(self.task)==56
            backhaul_class[keep_mask[:,-1]]=2
        td = TensorDict(
            {
                "locs": locs,
                "demand_backhaul": demand_backhaul,  # (C)
                "demand_linehaul": demand_linehaul,  # (B)
                "backhaul_class": backhaul_class,  # (B) M
                "distance_limit": distance_limit,  # (L)
                "time_windows": time_windows,  # (TW)
                "service_time": service_time,  # (TW)
                "vehicle_capacity": vehicle_capacity,  # (C)
                "capacity_original": capacity_original,  # unscaled capacity (C)
                "open_route": open_route,  # (O)
                "capacity_route": capacity_route,  # (O)
                "speed": speed,  # common
                "task_id":keep_mask,
                "task_embedding":task_embedding,
                "asymmetric_route":asymmetric_route
            },
            batch_size=batch_size,
        )
        return  self.subsample_problems(td)


    def subsample_problems(self, td):
        keep_mask=td['task_id'].clone()
        td = self._default_open(td, ~keep_mask[:, 0])# O
        td = self._default_time_window(td, ~keep_mask[:, 1])# TW
        td = self._default_distance_limit(td, ~keep_mask[:, 2])# L
        td = self._default_backhaul(td, ~keep_mask[:, 3])# B
        td = self._default_capacity(td, ~keep_mask[:, 4])# C
        td = self._default_distances(td, ~keep_mask[:, 5])# H

        binary_values_row_sum = keep_mask.int().mv(torch.tensor([64,32,16, 8, 4, 2, 1], dtype=torch.int32))
        td['task_id']=binary_values_row_sum
        if self.llm_embedding:
            try:
                task_embedding=torch.load(f'{self.llm_model}.pt',weights_only=True)
                td["task_embedding"]=task_embedding[binary_values_row_sum]
            except:
                task_embedding=torch.load('PLM/task_embeddings_USE.pt',weights_only=True)
                td["task_embedding"]=task_embedding[binary_values_row_sum]
        return td

    @staticmethod
    def _default_capacity(td, remove):
        td["capacity_route"][remove] = False
        return td
    @staticmethod
    def _default_open(td, remove):
        td["open_route"][remove] = False
        return td

    @staticmethod
    def _default_time_window(td, remove):
        default_tw = torch.zeros_like(td["time_windows"])
        default_tw[..., 1] = float("inf")
        td["time_windows"][remove] = default_tw[remove]
        td["service_time"][remove] = torch.zeros_like(td["service_time"][remove])
        return td

    @staticmethod
    def _default_distance_limit(td, remove):
        td["distance_limit"][remove] = float("inf")
        return td

    @staticmethod
    def _default_backhaul(td, remove):
        # by default, where there is a backhaul, linehaul is 0. therefore, we add backhaul to linehaul
        # and set backhaul to 0 where we want to remove backhaul
        td["demand_linehaul"][remove] = (
            td["demand_linehaul"][remove] + td["demand_backhaul"][remove]
        )
        td["demand_backhaul"][remove] = 0
        return td

    @staticmethod
    def _default_distances(td, remove):
        """Generate distance  asymmetric
        """
        td["asymmetric_route"][remove] = False
        return  td

    def generate_locations(self, batch_size, num_loc) -> torch.Tensor:
        """Generate seed locations.

        Returns:
            locs: [B, N+1, 2] where the first location is the depot.
        """
        locs = torch.FloatTensor(*batch_size, num_loc + 1, 2).uniform_(
            self.min_loc, self.max_loc
        )
        return locs

    def generate_demands(self, batch_size: int, num_loc: int) -> torch.Tensor:
        """Classical lineahul demand / delivery from depot (C) and backhaul demand / pickup to depot (B) generation.
        Initialize the demand for nodes except the depot, which are added during reset.
        Demand sampling Following Kool et al. (2019), demands as integers between 1 and 10.
        Generates a slightly different distribution than using torch.randint.

        Returns:
            linehaul_demand: [B, N]
            backhaul_demand: [B, N]
        """
        linehaul_demand = torch.FloatTensor(*batch_size, num_loc).uniform_(
            self.min_demand - 1, self.max_demand - 1
        )
        linehaul_demand = (linehaul_demand.int() + 1).float()
        # Backhaul demand sampling
        backhaul_demand = torch.FloatTensor(*batch_size, num_loc).uniform_(
            self.min_backhaul - 1, self.max_backhaul - 1
        )
        backhaul_demand = (backhaul_demand.int() + 1).float()
        is_linehaul = torch.rand(*batch_size, num_loc) > self.backhaul_ratio
        backhaul_demand = (
            backhaul_demand * ~is_linehaul
        )  # keep only values where they are not linehauls
        linehaul_demand = linehaul_demand * is_linehaul
        return linehaul_demand, backhaul_demand

    def generate_time_windows(
        self,
        locs: torch.Tensor,
        speed: torch.Tensor,
    ) -> torch.Tensor:
        """Generate time windows (TW) and service times for each location including depot.
        We refer to the generation process in "Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization"
        (Liu et al., 2024). Note that another way to generate is from "Learning to Delegate for Large-scale Vehicle Routing" (Li et al, 2021) which
        is used in "MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts" (Zhou et al, 2024). Note that however, in that case
        the distance limit would have no influence when time windows are present, since the tw for depot is the same as distance with speed=1.
        This function can be overridden for that implementation.
        See also https://github.com/RoyalSkye/Routing-MVMoE

        Args:
            locs: [B, N+1, 2] (depot, locs)
            speed: [B]

        Returns:
            time_windows: [B, N+1, 2]
            service_time: [B, N+1]
        """

        batch_size, n_loc = locs.shape[0], locs.shape[1] - 1  # no depot

        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, n_loc)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc)
        d_0i = get_distance(locs[:, 0:1], locs[:, 1:])
        h_max = (self.max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, n_loc)) * d_0i / speed
        tw_end = tw_start + tw_length

        # Depot tw is 0, max_time
        time_windows = torch.stack(
            (
                torch.cat((torch.zeros(batch_size, 1), tw_start), -1),  # start
                torch.cat((torch.full((batch_size, 1), self.max_time), tw_end), -1),
            ),  # en
            dim=-1,
        )
        # depot service time is 0
        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)
        return time_windows, service_time  # [B, N+1, 2], [B, N+1]

    def generate_distance_limit(
        self, shape: Tuple[int, int], locs: torch.Tensor
    ) -> torch.Tensor:
        """Generates distance limits (L).
        The distance lower bound is dist_lower_bound = 2 * max(depot_to_location_distance),
        then the max can be max_lim = min(max_distance_limit, dist_lower_bound + EPS). Ensures feasible yet challenging
        constraints, with each instance having a unique, meaningful limit

        Returns:
            distance_limit: [B, 1]
        """
        max_dist = torch.max(torch.cdist(locs[:, 0:1], locs[:, 1:]).squeeze(-2), dim=1)[0]
        dist_lower_bound = 2 * max_dist +1e-1
        max_distance_limit = torch.maximum(
            torch.full_like(dist_lower_bound, self.max_distance_limit),
            dist_lower_bound,
        )

        # We need to sample from the `distribution` module to get the same distribution with a tensor as input
        return torch.distributions.Uniform(dist_lower_bound, max_distance_limit).sample()[
            ..., None
        ]

    def generate_capacity_route(self, shape: Tuple[int, int]):
        """Generate capacity route flags (C). Here we could have a sampler but we simply return True here so all
        routes are open. Afterwards, we subsample the problems.
        """
        return torch.ones(shape, dtype=torch.bool)

    def generate_asymmetric_route(self, shape: Tuple[int, int]):
        """the distance matrix is asymmetric, i.e., the distance from A to B is not necessarily the same as the distance from B to A.
        """
        return torch.ones(shape, dtype=torch.bool)



    def generate_open_route(self, shape: Tuple[int, int]):
        """Generate open route flags (O). Here we could have a sampler but we simply return True here so all
        routes are open. Afterwards, we subsample the problems.
        """
        return torch.ones(shape, dtype=torch.bool)

    @staticmethod
    def generate_distances(locs):
        """Generate distance  matrix
        """
        return  torch.sqrt(torch.sum((locs[:, :, None, :] - locs[:, None, :, :]) ** 2, dim=-1))



    def generate_speed(self, shape: Tuple[int, int]):
        """We simply generate the speed as constant here"""
        # in this version, the speed is constant but this class may be overridden
        return torch.full(shape, self.speed, dtype=torch.float32)

    def generate_backhaul_class(self, shape: Tuple[int, int], sample: bool = False):
        """Generate backhaul class (B) for each node. If sample is True, we sample the backhaul class
        otherwise, we return the same class for all nodes.
        - Backhaul class 1: classic backhaul (VRPB), linehauls must be served before backhauls in a route (every customer is either, not both)
        - Backhaul class 2: mixed backhaul (VRPMPD or VRPMB), linehauls and backhauls can be served in any order (every customer is either, not both)
        """
        if sample:
            return torch.randint(1, 3, shape, dtype=torch.float32)
        else:
            return torch.full(shape, self.backhaul_class, dtype=torch.float32)

    @staticmethod
    def save_data(td: TensorDict, path, compress: bool = False):
        save_tensordict_to_npz(td, path)


