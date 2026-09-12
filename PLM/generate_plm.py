import torch
import tensorflow_hub as hub

# USE
model = hub.load("./universal-sentence-encoder/tensorFlow2/large/2")

task_PLM = {
    # 16 VRPs
    'cvrp': 'The basic Vehicle Routing Problem where vehicles deliver goods to a set of customers starting and ending at the depot. No additional constraints are considered.',
    'ovrp': 'Open Vehicle Routing Problem. Vehicles do not need to return to the depot after serving the last customer.',
    'vrpb': 'Vehicle Routing Problem with Backhauls. Vehicles serve both linehaul customers (deliveries) and backhaul customers (pickups), with linehaul deliveries made before backhaul pickups.',
    'vrpl': 'Vehicle Routing Problem with Duration Limits. There is a limit on the total travel time or distance of each route to balance the workload across vehicles.',
    'vrptw': 'Vehicle Routing Problem with Time Windows. Each customer has a time window during which service must occur. Vehicles must arrive within these time frames.',
    'ovrptw': 'Open Vehicle Routing Problem with Time Windows. Vehicles must respect the time windows for customers and are not required to return to the depot.',
    'ovrpb': 'Open Vehicle Routing Problem with Backhauls. Vehicles serve linehaul and backhaul customers without returning to the depot.',
    'ovrpl': 'Open Vehicle Routing Problem with Duration Limits. Vehicles do not return to the depot and must complete their routes within a specified time or distance.',
    'vrpbl': 'Vehicle Routing Problem with Backhauls and Duration Limits. Vehicles must serve linehaul customers before backhaul customers and complete their routes within a maximum travel time or distance.',
    'vrpbtw': 'Vehicle Routing Problem with Backhauls and Time Windows. Vehicles must respect the time windows for each customer and ensure linehaul deliveries occur before backhaul pickups.',
    'vrpltw': 'Vehicle Routing Problem with Duration Limits and Time Windows. Routes must be completed within a maximum time limit, and vehicles must arrive at customers within their specific time windows.',
    'ovrpbl': 'Open Vehicle Routing Problem with Backhauls and Duration Limits. Vehicles serve linehaul and backhaul customers, with a maximum allowed travel time, and do not need to return to the depot.',
    'ovrpbtw': 'Open Vehicle Routing Problem with Backhauls and Time Windows. Vehicles must respect the time windows and linehaul-before-backhaul rules without needing to return to the depot.',
    'ovrpltw': 'Open Vehicle Routing Problem with Duration Limits and Time Windows. Vehicles must complete routes within a time limit, respect time windows, and do not return to the depot.',
    'vrpbltw': 'Vehicle Routing Problem with Backhauls, Duration Limits, and Time Windows. Vehicles must serve linehaul customers first, complete the route within a time limit, and respect customer time windows.',
    'ovrpbltw': 'Open Vehicle Routing Problem with Backhauls, Duration Limits, and Time Windows. Vehicles respect all constraints and do not need to return to the depot after completing the route.',

    # 4 TSPs
    'tsp': 'Traveling Salesman Problem. A single vehicle must visit each customer exactly once and return to the starting location. No additional constraints apply.',
    'otsp': 'Open Traveling Salesman Problem. A single vehicle must visit each customer exactly once but does not need to return to the starting location.',
    'tsptw': 'Traveling Salesman Problem with Time Windows. A single vehicle must visit each customer exactly once within specified time windows and return to the starting location.',
    'otsptw': 'Open Traveling Salesman Problem with Time Windows. A single vehicle must visit each customer exactly once within specified time windows but does not need to return to the starting location.',

    # 20 Asymmetric Versions
    'hcvrp': 'Asymmetric Capacitated Vehicle Routing Problem. The distance from A to B is not necessarily the same as from B to A. Vehicles deliver goods to customers starting and ending at the depot with no additional constraints.',
    'hovrp': 'Asymmetric Open Vehicle Routing Problem. Vehicles serve customers and do not need to return to the depot, with asymmetric distances.',
    'hvrpb': 'Asymmetric Vehicle Routing Problem with Backhauls. Vehicles serve linehaul and backhaul customers with asymmetric distances, ensuring linehaul deliveries occur before backhaul pickups.',
    'hvrpl': 'Asymmetric Vehicle Routing Problem with Duration Limits. Vehicles have an asymmetric distance matrix and must complete routes within a maximum time or distance.',
    'hvrptw': 'Asymmetric Vehicle Routing Problem with Time Windows. Vehicles must adhere to customer time windows with asymmetric distances between locations.',
    'hovrptw': 'Asymmetric Open Vehicle Routing Problem with Time Windows. Vehicles serve customers without needing to return to the depot, with asymmetric distances and time windows constraints.',
    'hovrpb': 'Asymmetric Open Vehicle Routing Problem with Backhauls. Vehicles serve linehaul and backhaul customers with asymmetric distances and do not return to the depot.',
    'hovrpl': 'Asymmetric Open Vehicle Routing Problem with Duration Limits. Vehicles complete routes within a specified time or distance with asymmetric distances and do not return to the depot.',
    'hvrpbl': 'Asymmetric Vehicle Routing Problem with Backhauls and Duration Limits. Vehicles serve linehaul before backhaul customers, have asymmetric distances, and must complete routes within a time or distance limit.',
    'hvrpbtw': 'Asymmetric Vehicle Routing Problem with Backhauls and Time Windows. Vehicles serve linehaul before backhaul customers, respect customer time windows, and travel with asymmetric distances.',
    'hvrpltw': 'Asymmetric Vehicle Routing Problem with Duration Limits and Time Windows. Routes are constrained by a maximum time, with asymmetric distances and customer time windows.',
    'hovrpbl': 'Asymmetric Open Vehicle Routing Problem with Backhauls and Duration Limits. Vehicles serve linehaul and backhaul customers, respect the maximum allowed time, have asymmetric distances, and do not return to the depot.',
    'hovrpbtw': 'Asymmetric Open Vehicle Routing Problem with Backhauls and Time Windows. Vehicles respect time windows and linehaul-before-backhaul rules, with asymmetric distances, and do not return to the depot.',
    'hovrpltw': 'Asymmetric Open Vehicle Routing Problem with Duration Limits and Time Windows. Vehicles serve customers within a maximum time, with time windows and asymmetric distances, and do not return to the depot.',
    'hvrpbltw': 'Asymmetric Vehicle Routing Problem with Backhauls, Duration Limits, and Time Windows. Vehicles respect linehaul-before-backhaul, a maximum time limit, time windows, and asymmetric distances.',
    'hovrpbltw': 'Asymmetric Open Vehicle Routing Problem with Backhauls, Duration Limits, and Time Windows. Vehicles respect all constraints, travel with asymmetric distances, and do not return to the depot after completing the route.',
    'htsp': 'Asymmetric Traveling Salesman Problem. A single vehicle must visit each customer exactly once and return to the starting location, with asymmetric distances between locations.',
    'hotsp': 'Asymmetric Open Traveling Salesman Problem. A single vehicle must visit each customer exactly once with asymmetric distances and does not need to return to the starting location.',
    'htsptw': 'Asymmetric Traveling Salesman Problem with Time Windows. A single vehicle visits each customer within specified time windows and returns to the start, with asymmetric distances.',
    'hotsptw': 'Asymmetric Open Traveling Salesman Problem with Time Windows. A single vehicle visits each customer exactly once within time windows, with asymmetric distances, and does not need to return to the starting location.',

    ## 16 Finetune
    'vrpmb': 'Vehicle Routing Problem with Mixed Backhauls and Backhauls. Vehicles can mix linehaul and backhaul customers without a strict sequence, but must respect capacity and backhaul demand.',
    'ovrpmb': 'Open Vehicle Routing Problem with Mixed Backhauls and Backhauls. Vehicles can mix linehaul and backhaul customers without needing to return to the depot.',
    'vrpmbl': 'Vehicle Routing Problem with Mixed Backhauls, Backhauls, and Duration Limits. Routes must respect duration limits and capacity constraints while serving mixed customer types.',
    'vrpmbtw': 'Vehicle Routing Problem with Mixed Backhauls, Backhauls, and Time Windows. Vehicles can mix linehaul and backhaul customers while respecting time windows and capacity limits.',
    'ovrpmbl': 'Open Vehicle Routing Problem with Mixed Backhauls, Backhauls, and Duration Limits. Vehicles must respect a time limit, can mix customers, and do not return to the depot.',
    'ovrpmbtw': 'Open Vehicle Routing Problem with Mixed Backhauls, Backhauls, and Time Windows. Vehicles mix linehaul and backhaul customers, respect time windows, and do not return to the depot.',
    'vrpmbltw': 'Vehicle Routing Problem with Mixed Backhauls, Backhauls, Duration Limits, and Time Windows. Routes must adhere to duration and time window constraints, while mixing linehaul and backhaul customers.',
    'ovrpmbltw': 'Open Vehicle Routing Problem with Mixed Backhauls, Backhauls, Duration Limits, and Time Windows. Vehicles respect all constraints and do not return to the depot after completing the route.',

    'hvrpmb': 'Asymmetric Vehicle Routing Problem with Mixed Backhauls and Backhauls. Vehicles can mix linehaul and backhaul customers without a strict sequence, but must respect capacity and backhaul demand, with asymmetric distances.',
    'hovrpmb': 'Asymmetric Open Vehicle Routing Problem with Mixed Backhauls and Backhauls. Vehicles can mix linehaul and backhaul customers without needing to return to the depot, with asymmetric distances.',
    'hvrpmbl': 'Asymmetric Vehicle Routing Problem with Mixed Backhauls, Backhauls, and Duration Limits. Routes must respect duration limits and capacity constraints while serving mixed customer types, with asymmetric distances.',
    'hvrpmbtw': 'Asymmetric Vehicle Routing Problem with Mixed Backhauls, Backhauls, and Time Windows. Vehicles can mix linehaul and backhaul customers while respecting time windows and capacity limits, with asymmetric distances.',
    'hovrpmbl': 'Asymmetric Open Vehicle Routing Problem with Mixed Backhauls, Backhauls, and Duration Limits. Vehicles must respect a time limit, can mix customers, and do not return to the depot, with asymmetric distances.',
    'hovrpmbtw': 'Asymmetric Open Vehicle Routing Problem with Mixed Backhauls, Backhauls, and Time Windows. Vehicles mix linehaul and backhaul customers, respect time windows, and do not return to the depot, with asymmetric distances.',
    'hvrpmbltw': 'Asymmetric Vehicle Routing Problem with Mixed Backhauls, Backhauls, Duration Limits, and Time Windows. Routes must adhere to duration and time window constraints while mixing linehaul and backhaul customers, with asymmetric distances.',
    'hovrpmbltw': 'Asymmetric Open Vehicle Routing Problem with Mixed Backhauls, Backhauls, Duration Limits, and Time Windows. Vehicles respect all constraints and do not return to the depot after completing the route, with asymmetric distances.'
}
def embed(input):
  return torch.tensor(model(input).numpy())
task_embeddings=torch.zeros(128,512)
task_embeddings_plm = embed(list(task_PLM.values()))
task_id=torch.load('PLM/task_id.pt', weights_only=True)
for i,key in enumerate(task_PLM):
    task_embeddings[task_id[key]]=task_embeddings_plm[i]
torch.save(task_embeddings,'task_embeddings_.pt')
