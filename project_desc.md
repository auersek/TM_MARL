
# Final Year Project: Multi-Agent Reinforcement Learning for Cooperative Real-Time Control in Trackmania

**Imperial College London — MEng Electronic and Information Engineering**

**Supervisor:** Dr. Dan Goodman

**Original FYP:** Learning to Vote When It Matters

**Proposed extension:** Cooperative Multi-Agent Reinforcement Learning in Trackmania

**Project status:** Confirmed FYP. Research architecture and experimental methodology to be developed.

---

# 1. Project Overview

The aim of this project is to investigate how multiple artificial intelligence agents with different information, expertise, and potentially conflicting interests can learn to communicate and cooperate to make effective decisions under time constraints.

The project extends the original _Learning to Vote When It Matters_ research proposal by applying multi-agent reinforcement learning (MARL) to the racing simulator Trackmania.

Instead of training a single neural network to control a car using all available information, the proposed system distributes information among multiple specialised agents that must collectively control one vehicle.

Each agent has access to a different subset of the environment's information. For example:

- **Vision agent:** Receives camera images and learns to recognise corners, obstacles, track boundaries, and visual information.
    
- **Telemetry agent:** Receives the vehicle's position, velocity, heading, angular velocity, and other physical measurements.
    
- **Track/strategy agent:** Receives information about the track layout, upcoming turns, checkpoints, and potentially the optimal racing line.
    

Because no individual agent necessarily has complete information, successful driving may require communication and cooperation.

The agents must collectively determine steering, acceleration, and braking actions while operating under real-time constraints.

The central research question is:

**Can multiple agents with complementary information learn an effective communication and collective decision-making strategy under time constraints, and how does their performance compare with a single agent that has access to all information?**

The broader objective is to investigate whether cooperative, consensus-based, democratic-like, or hierarchical decision-making emerges when agents have shared goals, different expertise, and limited time to coordinate.

Trackmania provides a practical environment in which cooperation can be evaluated quantitatively through racing performance.

Importantly, the goal is not simply to develop a racing AI. The research contribution lies in understanding the mechanisms and limitations of distributed intelligent decision-making.

---

# 2. Connection to the Original FYP

The original project, _Learning to Vote When It Matters_, investigates how groups of agents with different information, interests, and expertise can work together.

Possible collective decision-making mechanisms include:

- Majority voting.
    
- Expertise-weighted voting.
    
- Consensus through communication.
    
- Learned decision aggregation.
    
- Hierarchical decision-making.
    

The original research also considers whether time pressure and shared interests encourage agents to cooperate or adopt particular decision-making structures.

The Trackmania extension maps naturally onto these objectives.

|Original research objective|Trackmania implementation|
|---|---|
|Different information|Vision, telemetry, and track-data observations|
|Different expertise|Agents specialised in perception, vehicle dynamics, or strategy|
|Communication|Agents exchange learned or predefined messages|
|Collective decision-making|Agents jointly determine vehicle controls|
|Time constraints|Decisions must be produced within control deadlines|
|Shared interests|All agents benefit from completing a race quickly|
|Conflicting interests|Agents may prioritise safety, stability, or speed differently|
|Strong consequences|Poor decisions cause crashes or lost racing time|
|Emergent behaviour|Agents may learn specialisation, voting, trust, or hierarchy|

One important distinction is that cooperation, consensus, and democracy are not equivalent.

A system could achieve high performance by letting one agent control everything. This would demonstrate successful control, but not necessarily decentralised or democratic decision-making.

The project must therefore evaluate both the quality of decisions and the structure of the decision-making process.

---

# 3. Proposed System Architecture

## 3.1 General Architecture

The system consists of a Trackmania simulation, multiple agent policies, a communication mechanism, and a collective decision-making mechanism.

**Conceptual architecture diagram (text description):**

At the top is the Trackmania simulation, containing the car, track geometry, physics, and game state.

The simulation provides different observations to three parallel agents:

1. The vision agent receives RGB camera images.
    
2. The telemetry agent receives physical vehicle-state measurements.
    
3. The strategy agent receives track geometry or waypoint information.
    

Each agent processes its observations through an appropriate neural network.

The vision agent might use a convolutional neural network (CNN) or pretrained visual encoder.

The telemetry and strategy agents might use multilayer perceptrons (MLPs).

Below these three agents is a communication layer. Depending on the architecture, agents may exchange messages, predictions, confidence estimates, or learned latent representations.

A collective decision mechanism combines the agents' information and recommendations to produce a single control action.

The final control action consists of steering, throttle, and braking commands.

These commands are executed in Trackmania, causing the environment to transition to a new state.

The agents then receive updated observations and rewards.

During training, reinforcement learning algorithms update the agents' policies using the outcomes of these interactions.

This process repeats over many simulation steps and training episodes.

## 3.2 Possible Agent Inputs

### Vision Agent

Possible observations:

- Current RGB camera image.
    
- Recent image frames.
    
- Visible track boundaries.
    
- Upcoming corners.
    
- Obstacles and barriers.
    
- Relative orientation of the vehicle to the track.
    

Possible architecture:

RGB frames → CNN or visual encoder → Feature representation → Policy network.

The agent could learn to estimate the direction and severity of upcoming corners or directly propose control actions.

### Telemetry Agent

Possible observations:

- Vehicle position.
    
- Linear velocity.
    
- Angular velocity.
    
- Heading.
    
- Acceleration.
    
- Lateral velocity or slip-related measurements.
    
- Previous control actions.
    

Possible architecture:

Telemetry vector → MLP → Feature representation → Policy network.

This agent could become particularly useful for understanding vehicle dynamics, stability, and appropriate steering or braking responses.

### Track/Strategy Agent

Possible observations:

- Track layout.
    
- Upcoming waypoint coordinates.
    
- Relative direction to the next checkpoint.
    
- Distance to corners.
    
- Track curvature.
    
- Information about an optimal racing line, if available.
    

Possible architecture:

Track/waypoint representation → MLP or sequence encoder → Policy network.

This agent could contribute longer-term strategic information unavailable from the current camera frame.

The precise observations would depend on the available Trackmania integration and the chosen experimental setup.

---

# 4. Mathematical Formulation

The system can be modelled as a partially observable multi-agent sequential decision-making problem.

If all agents share the same reward, a cooperative Decentralised Partially Observable Markov Decision Process (Dec-POMDP) provides a relevant mathematical framework.

If agents have different reward functions, a partially observable stochastic game is a more general formulation.

## 4.1 Environment State

Let the complete environment state at time t be:

`s_t = (p_t, v_t, θ_t, ω_t, T, ...)`

Where:

- `p_t` = Vehicle position.
    
- `v_t` = Vehicle velocity.
    
- `θ_t` = Vehicle orientation.
    
- `ω_t` = Angular velocity.
    
- `T` = Track information.
    

Additional variables might include wheel-contact states, checkpoint progress, and other physical quantities.

## 4.2 Partial Observations

Each agent receives an observation derived from the complete state:

`o_t^i ~ O_i(. | s_t)`

For example:

- Vision: `o_t^vision = RGB image`
    
- Telemetry: `o_t^telemetry = (position, velocity, orientation, ...)`
    
- Strategy: `o_t^strategy = track/waypoint information`
    

The different observation functions ensure that no agent necessarily has access to all relevant information.

This creates a need for information sharing.

## 4.3 Agent Policies

Each agent has a decision-making policy:

`π_i(a_i | o_i, h_i, m_i)`

Where:

- `π_i` = Agent i's policy.
    
- `o_i` = Current local observation.
    
- `h_i` = Observation history or internal memory.
    
- `m_i` = Messages received from other agents.
    
- `a_i` = Agent's proposed action.
    

The policy could be implemented using a neural network.

If temporal information is important, recurrent neural networks, gated recurrent units (GRUs), or another memory mechanism could be incorporated.

## 4.4 Vehicle Control Actions

The final control action is:

`a_t = (δ_t, τ_t, b_t)`

Where:

- `δ_t ∈ [-1, 1]` = Steering.
    
- `τ_t ∈ [0, 1]` = Throttle.
    
- `b_t ∈ [0, 1]` = Brake.
    

These ranges are a proposed normalised control representation.

A collective decision function combines the agents' recommendations:

`a_t = F(a_t^1, a_t^2, ..., a_t^N, m_t)`

The function F might be fixed or learned.

The simulator applies the action, advances its physics, and produces the next state.

## 4.5 Reinforcement Learning Objective

The goal of reinforcement learning is to learn policies that maximise expected cumulative reward.

A standard objective is:

`J(π) = E[Σ γ^t r_t]`

Where:

- `r_t` = Reward at time t.
    
- `γ` = Discount factor.
    
- `π` = Policy being optimised.
    

In a multi-agent system, agents may maximise a shared team reward, individual rewards, or a combination of both.

---

# 5. Possible Multi-Agent Architectures

Several architectures could be investigated. These are alternatives rather than components that must all be implemented.

## Architecture A: Independent Policies with Fixed Aggregation

This is the simplest multi-agent baseline.

Each agent independently observes its own information and proposes vehicle controls.

The proposed actions are combined using a predetermined mechanism.

**Diagram description:**

Three parallel observation-processing branches are shown.

Each branch contains an input, an agent policy network, and a proposed control action.

The three proposed actions feed into a fixed aggregation block.

The aggregation block outputs the final steering, throttle, and braking commands.

There is no explicit communication between the agent branches.

### Simple Averaging

For steering:

`δ_final = (δ_1 + δ_2 + ... + δ_N) / N`

Equivalent averaging could be applied to throttle and braking.

### Weighted Aggregation

Alternatively:

`a_final = Σ w_i a_i`

Subject to:

`Σ w_i = 1` and `w_i ≥ 0`.

The weights could represent agent reliability or expertise.

For example, the vision agent might receive greater influence near an unfamiliar corner, while telemetry may be more important during a slide.

Advantages:

- Simple implementation.
    
- Useful baseline.
    
- Relatively easy to interpret.
    

Disadvantages:

- No explicit communication.
    
- Conflicting actions may cancel each other.
    
- Fixed weights may fail when expertise depends on the situation.
    

Majority voting could also be used if control proposals were discretised into choices such as left, right, straight, brake, or accelerate.

For continuous controls, weighted averaging is generally a more natural starting point.

---

## Architecture B: Learned Communication Between Agents

In this architecture, agents learn what information to communicate.

Each agent contains both a control policy and a communication mechanism.

An agent could generate a message:

`m_i = g_i(o_i, h_i)`

Where `g_i` is a learnable communication function.

Messages may be continuous vectors or discrete symbols.

**Diagram description:**

Three agent encoders process vision, telemetry, and track data independently.

Their encoded observations enter a message-passing network.

Messages are exchanged between agents for one or more communication rounds.

Each agent updates its internal representation using the received messages.

The updated representations are then used to produce action proposals or a collective control decision.

The final action is executed in Trackmania.

### Example

The vision agent detects a sharp corner.

It communicates a learned message representing the corner's direction and severity.

The telemetry agent combines that message with the car's current speed.

The agents collectively determine an appropriate steering and braking action.

The message does not necessarily need to be interpretable by humans.

An important research question is whether agents learn useful communication protocols naturally through reinforcement learning.

Possible experiments:

- Communication versus no communication.
    
- Fixed messages versus learned messages.
    
- One communication round versus multiple rounds.
    
- Small versus large message vectors.
    
- Communication cost versus driving performance.
    

This architecture is particularly closely aligned with the original FYP.

---

## Architecture C: Complementary Perception and Control Expertise

This architecture focuses on agents with different capabilities rather than simply different observations.

For example:

**Perception agent:**

- Has strong visual information.
    
- Understands track geometry.
    
- Detects upcoming corners.
    
- Does not directly control the car.
    

**Control agent:**

- Understands steering, throttle, and braking.
    
- Learns vehicle dynamics.
    
- Has limited information about upcoming track geometry.
    
- Executes the final control action.
    

**Diagram description:**

The environment sends visual observations to a perception agent.

The perception agent converts these observations into messages or high-level instructions.

These messages are transmitted to a control agent.

The control agent combines the messages with its own observations and produces the final vehicle controls.

The environment executes those controls and returns new observations and rewards.

This allows the project to investigate **cross-expertise communication**.

The central question is whether agents with complementary capabilities can learn a communication protocol enabling them to solve a task that neither could solve effectively alone.

Possible extensions include swapping expertise between agents, reducing communication bandwidth, or introducing deliberate misunderstandings between agents.

This approach studies distributed problem-solving, although it does not necessarily involve voting.

---

## Architecture D: Hierarchical or Leader-Based Decision-Making

Another possibility is to use a hierarchical structure.

Multiple lower-level agents process different kinds of information.

A higher-level coordinating agent receives their outputs and selects the final control action.

**Diagram description:**

Three specialised agents appear at the lower level.

All three send their observations, features, or proposed actions to a higher-level coordination agent.

The coordinator combines this information and produces steering, braking, and throttle commands.

There is no need for the lower-level agents to agree directly with one another.

The hierarchy may be explicitly designed, or some hierarchical influence may emerge through learning.

The project could investigate whether strict time pressure encourages a group to adopt leader-based decision-making.

A hierarchy might reduce communication requirements and improve response speed.

However, it could also make the system dependent on one agent and reduce the influence of other agents.

A useful comparison would be decentralised learned communication versus a centralised coordinator using the same observation information.

---

# 6. Reinforcement Learning Approaches

The project could use several reinforcement learning algorithms depending on the complexity of the environment and the control architecture.

## 6.1 Q-Learning

Q-learning estimates the expected discounted return of taking an action in a particular state:

`Q(s, a)`

It is particularly suitable for small environments with discrete states and actions.

An initial simplified racing environment could use actions such as:

- Steer left.
    
- Steer right.
    
- Go straight.
    
- Accelerate.
    
- Brake.
    

Q-learning could be used to establish a basic multi-agent experiment before introducing continuous control or neural networks.

## 6.2 Deep Q-Networks (DQN)

DQN approximates Q-values using a neural network.

It is useful for more complex observations but is most directly applicable to discrete action spaces.

Continuous steering would require discretisation or another algorithm.

## 6.3 Proximal Policy Optimisation (PPO)

PPO is a policy-gradient algorithm that directly learns a policy.

It is well suited to many continuous-control problems.

The policy can output a probability distribution over actions, rather than simply selecting the action with the largest Q-value.

PPO uses a clipped optimisation objective to limit excessively large policy updates.

## 6.4 Independent PPO

Each agent has its own PPO policy and learns from experience.

This is relatively straightforward conceptually.

However, when multiple agents learn simultaneously, changes in one agent's policy alter the environment experienced by the others.

This produces the multi-agent non-stationarity problem.

## 6.5 Multi-Agent PPO (MAPPO)

MAPPO extends PPO to cooperative multi-agent settings.

A common approach is Centralised Training, Decentralised Execution (CTDE).

During training, a centralised critic can use additional global information to evaluate actions.

During execution, agents use only their local observations and permitted communication.

This could be particularly useful for the Trackmania project.

## 6.6 Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic method used for continuous control.

It incorporates an entropy objective that encourages exploration.

SAC is relevant because Trackmania is fundamentally a continuous-control problem, and existing Trackmania RL frameworks have used SAC-based approaches.

## 6.7 Imitation Learning

Another possibility is to pretrain agents using demonstrations.

For example, a successful single-agent driving policy could generate trajectories.

The multi-agent system might learn basic control behaviour from these trajectories before being trained to communicate.

This could reduce the difficulty of simultaneously learning basic driving and cooperative behaviour from scratch.

---

# 7. Centralised Training, Decentralised Execution

CTDE is an important concept for this project.

During training, the learning algorithm may access:

- The complete vehicle state.
    
- All agents' observations.
    
- All proposed actions.
    
- Global rewards.
    
- Communication messages.
    

A centralised critic can use this information to estimate the quality of collective decisions.

However, during execution, individual agents would only be allowed to access their own observations and permitted messages.

**Diagram description:**

The training system contains multiple separate policy networks and one centralised critic.

The critic receives global environment information and potentially the actions of all agents.

It provides learning signals used to update the individual policies.

During deployment, the centralised critic is removed from the decision-making process.

Each agent independently observes its assigned inputs, communicates where allowed, and contributes to the final control decision.

This allows training to benefit from global information while preserving decentralised execution.

It is important to distinguish the training critic from a central controller that directly chooses all actions. A centralised critic does not necessarily control the agents during execution.

---

# 8. Reward Functions and Shared Interests

Reward design is a central research challenge.

If the reward function is poorly designed, agents may optimise undesirable behaviour.

For example, rewarding consensus too strongly could encourage agents to agree quickly even when their decisions are incorrect.

## 8.1 Shared Team Reward

A simple team reward might be:

`R_team = α × progress - β × crash_penalty - γ × time_cost`

Where:

- Progress rewards moving towards the next checkpoint.
    
- Crash penalty discourages collisions.
    
- Time cost encourages faster completion.
    

All agents receive the same reward.

This encourages a common objective.

A shared reward is likely the simplest initial setting.

## 8.2 Individual Reward Functions

To investigate partially conflicting interests, agents could have separate objectives.

Examples:

**Vision agent:** Rewarded for collision avoidance and staying on the track.

**Telemetry agent:** Rewarded for stable handling and avoiding loss of control.

**Strategy agent:** Rewarded for minimising lap time and maintaining progress.

The complete reward could be:

`R_i = λ R_team + (1 - λ) R_i_individual`

Where:

`0 ≤ λ ≤ 1`

When λ = 1, agents only receive the shared team reward.

When λ = 0, agents only receive their individual rewards.

Intermediate values create partially aligned interests.

By varying λ, the project could investigate whether common interests encourage cooperation and how different incentives affect the collective decision mechanism.

This could also reveal whether agents learn strategic or misleading communication when their incentives diverge.

Such results should be treated as experimental findings rather than assumed in advance.

---

# 9. Time-Constrained Decision-Making

Time pressure is a defining feature of the original research project.

Trackmania naturally introduces the need for repeated, rapid control decisions.

The system could impose several kinds of constraints.

## 9.1 Communication-Round Limits

Agents may exchange only a limited number of messages before selecting a control action.

For example:

- Zero communication rounds.
    
- One communication round.
    
- Two communication rounds.
    
- Four communication rounds.
    

The project could evaluate whether additional communication improves performance and whether the benefit eventually saturates.

## 9.2 Communication Bandwidth

Each agent could transmit only a limited number of values or bits.

This encourages the communication policy to encode relevant information efficiently.

A communication bottleneck may encourage agents to specialise or develop compact representations.

## 9.3 Communication Costs

Messages could incur a penalty.

For example:

`R = R_driving - η × communication_cost`

This creates an incentive to communicate only when the expected benefit exceeds the cost.

## 9.4 Real-Time Deadlines

The total decision-making process could have an actual execution deadline.

For example, a 20 Hz controller has a 50 ms control period.

Within this period, the system must:

1. Collect observations.
    
2. Run agent inference.
    
3. Exchange any required messages.
    
4. Combine the agent outputs.
    
5. Produce the final control action.
    

The experiment could investigate how different architectures behave under tighter deadlines.

A simulated limit on communication rounds is not the same as a real wall-clock time limit. Ideally, the project would distinguish between these two constraints.

---

# 10. Experimental Baselines

Appropriate baselines are essential to determine whether multi-agent cooperation provides a measurable advantage.

Possible configurations include:

|Configuration|Purpose|
|---|---|
|Random policy|Establish basic chance-level performance|
|Single agent with all information|Centralised information baseline|
|Single vision-only agent|Evaluate visual control capability|
|Multiple agents, no communication|Evaluate information separation|
|Multiple agents, simple averaging|Test fixed action aggregation|
|Multiple agents, weighted voting|Test expertise-based aggregation|
|Multiple agents, learned communication|Test learned cooperation|
|Multiple agents, constrained communication|Investigate time pressure|
|Multiple agents, conflicting rewards|Investigate mixed incentives|
|Hierarchical controller|Compare decentralised and leader-based control|

The single-agent centralised baseline is particularly important.

A centralised agent with access to all information may have an advantage because it does not need communication.

The research should investigate whether distributed agents can approach that performance, and what costs or benefits arise from decentralisation.

Comparisons should control for model capacity, information access, training budget, and environment conditions as far as practical.

---

# 11. Evaluation Metrics

The project needs to measure more than lap time.

A system could complete races successfully without demonstrating meaningful cooperation.

Evaluation should therefore cover several dimensions.

## 11.1 Driving Performance

- Lap completion rate.
    
- Average lap time.
    
- Median lap time.
    
- Distance progressed before failure.
    
- Number of crashes.
    
- Checkpoints reached.
    
- Performance across different tracks.
    
- Generalisation to unseen track layouts.
    

## 11.2 Communication Efficiency

- Number of messages exchanged.
    
- Message size.
    
- Communication bandwidth.
    
- Number of communication rounds.
    
- Communication overhead.
    
- Total decision latency.
    
- Performance under communication restrictions.
    

## 11.3 Cooperation and Consensus

- Agreement between action proposals.
    
- Sensitivity of group decisions to individual agents.
    
- Degree of influence held by each agent.
    
- Use of expertise in relevant situations.
    
- Performance when individual agents are removed.
    
- Performance when messages are corrupted or withheld.
    
- Whether communication improves outcomes relative to no communication.
    

Agreement on continuous control actions would need a suitable definition, such as agreement within a steering tolerance, rather than exact equality.

## 11.4 Training Performance

- Total environment steps.
    
- Training wall-clock time.
    
- Sample efficiency.
    
- Learning stability.
    
- Variance across random seeds.
    
- Sensitivity to hyperparameters.
    
- Computational and memory requirements.
    

## 11.5 Emergent Behaviour

Potential behaviours to investigate include:

- Specialisation.
    
- Expertise-weighted influence.
    
- Learned trust.
    
- Efficient information sharing.
    
- Hierarchical control.
    
- Consensus-seeking.
    
- Dominance by a single agent.
    
- Communication only when necessary.
    

These behaviours should be measured using clearly defined experiments rather than inferred from racing performance alone.

---

# 12. Ablation Studies

Ablation experiments help determine which parts of the system actually contribute to success.

For example, a trained three-agent system could be evaluated under several modifications.

### Remove the Vision Agent

Does the car still navigate unfamiliar corners?

If performance collapses, the visual information may be important.

### Remove the Telemetry Agent

Can the remaining agents compensate for the missing vehicle-state measurements?

This may reveal whether the telemetry agent provides genuinely useful information.

### Remove the Strategy Agent

Does the system still handle complex track layouts?

If lap times increase significantly, the strategy agent may be contributing useful longer-term information.

### Disable Communication

Does the group still perform well?

A significant deterioration would provide evidence that communication is useful.

### Replace Learned Communication with Fixed Messages

Does learned communication offer an advantage over predefined information exchange?

### Randomise Messages

If replacing messages with random data has little effect, the trained agents may not actually be using the communication channel.

### Restrict Communication Time

Does the learned strategy change under tighter deadlines?

These experiments can help determine whether the agents are genuinely cooperating or simply relying on one dominant policy.

---

# 13. Technical Challenges

## 13.1 Multi-Agent Non-Stationarity

In ordinary reinforcement learning, the environment is generally assumed to have stationary transition dynamics.

In multi-agent learning, the behaviour of other agents can change during training.

For example, the vision agent might learn to rely on messages from the telemetry agent.

However, as the telemetry agent changes its policy, the meaning or reliability of those messages may also change.

This makes training more difficult.

Possible approaches include CTDE, stabilised policy updates, controlled training schedules, or temporarily freezing certain policies.

## 13.2 Multi-Agent Credit Assignment

If a car successfully navigates a corner, which agent contributed most to that outcome?

The vision agent might have detected the corner, the telemetry agent might have identified excessive speed, and the strategy agent might have recommended braking early.

If every agent receives the same reward, determining the contribution of each agent may be difficult.

Possible approaches include counterfactual baselines, difference rewards, centralised critics, and value-function decomposition.

The project may investigate these techniques depending on its final scope.

## 13.3 Reward Exploitation

Agents may discover behaviours that maximise the specified reward without accomplishing the intended objective.

For example:

- Agents may prioritise consensus over correct decisions.
    
- A strategy agent may encourage aggressive driving that improves short-term progress but causes frequent crashes.
    
- Agents may stop communicating if another agent can produce acceptable rewards alone.
    

Careful reward design and evaluation are therefore essential.

## 13.4 Learning Communication

Agents may fail to establish useful communication.

They may ignore messages, send redundant information, or develop protocols that only work with specific training partners.

Learned messages may also be difficult for humans to interpret.

Communication performance should be evaluated by its effect on decisions, not simply whether messages are being transmitted.

## 13.5 Computational Complexity

Training multiple agents to drive competently may require substantial simulation time.

Vision-based policies can be particularly expensive.

If all agents learn driving and communication simultaneously from scratch, the project may become difficult to train.

Possible simplifications include:

- Starting with fewer agents.
    
- Using smaller observation spaces.
    
- Reusing pretrained visual encoders.
    
- Pretraining a driving policy.
    
- Beginning with fixed communication.
    
- Using a simplified racing environment for preliminary experiments.
    

## 13.6 Defining Democratic Behaviour

Consensus alone does not establish democracy.

A group could reach perfect consensus by following a single dominant agent.

If the project investigates democratic-like behaviour, it should define measurable properties such as:

- Distribution of decision-making influence.
    
- Responsiveness to relevant expertise.
    
- Participation in communication.
    
- Ability of different agents to affect outcomes.
    
- Collective decision quality.
    
- Degree of agreement.
    

This distinction is central to the original research motivation.

---

# 14. Practical Implementation

## 14.1 Trackmania Reinforcement Learning Framework

A useful open-source starting point is TMRL.

GitHub:

[https://github.com/trackmania-rl/tmrl](https://github.com/trackmania-rl/tmrl)

Documentation:

[https://tmrl.readthedocs.io/en/latest/](https://tmrl.readthedocs.io/en/latest/)

TMRL provides infrastructure for reinforcement learning in Trackmania 2020.

Relevant features include:

- Interaction with the game environment.
    
- Collection of observations.
    
- Vehicle-control interfaces.
    
- Reinforcement learning training infrastructure.
    
- Image-based, LIDAR-style, and telemetry-related observations, depending on configuration.
    

The project would need to adapt or extend the framework to support multiple policies controlling a single vehicle.

Potential engineering tasks include:

1. Creating separate observation spaces.
    
2. Building individual agent policy interfaces.
    
3. Implementing message passing.
    
4. Implementing action aggregation.
    
5. Managing synchronised control steps.
    
6. Calculating team and individual rewards.
    
7. Logging communication and behaviour.
    
8. Evaluating trained systems on selected tracks.
    

The exact available observations and integration requirements would need to be verified during development.

## 14.2 Possible Software Stack

|Component|Technology|
|---|---|
|Programming language|Python|
|Neural network framework|PyTorch|
|Racing environment|Trackmania 2020|
|RL integration|TMRL|
|Multi-agent environment interface|Custom wrapper, potentially PettingZoo|
|Environment conventions|Gymnasium|
|Numerical processing|NumPy|
|Reinforcement learning|PPO, MAPPO, SAC, or simpler methods|
|Experiment tracking|TensorBoard or another logger|
|Data analysis|pandas, Matplotlib|
|Version control|Git|

The project does not inherently require custom hardware or FPGA development.

Most of the technical work is likely to involve machine learning, reinforcement learning, simulator integration, and experimental analysis.

---

# 15. Proposed Development Plan

The following is a possible implementation sequence.

## Phase 1: Literature Review and Environment Setup

Objectives:

- Study single-agent reinforcement learning.
    
- Review multi-agent reinforcement learning.
    
- Investigate cooperative and mixed-interest systems.
    
- Understand learned communication mechanisms.
    
- Study the original research on collective voting and expertise.
    
- Install and test TMRL.
    
- Establish a reliable simulator-control interface.
    

Deliverable:

A working environment capable of collecting observations, sending controls, and logging racing performance.

## Phase 2: Single-Agent Baseline

Develop or reuse a single-agent driving policy.

Initially, the agent may receive all available observations.

Measure:

- Lap completion rate.
    
- Lap time.
    
- Training efficiency.
    
- Generalisation to different tracks.
    

Deliverable:

A reproducible centralised control baseline.

## Phase 3: Distributed Information and Fixed Aggregation

Introduce multiple agents with different observation spaces.

Initially, use a shared reward function and simple aggregation.

Possible methods:

- Mean action aggregation.
    
- Fixed expertise-weighted aggregation.
    
- Discrete voting.
    
- Predetermined communication messages.
    

Deliverable:

A functioning multi-agent system that collectively controls one car.

## Phase 4: Learned Communication

Replace or extend fixed communication with a learned communication mechanism.

Possible approaches:

- Continuous message vectors.
    
- Discrete message symbols.
    
- One-round message passing.
    
- Multi-round communication.
    
- Learned importance weighting.
    

Compare against the previous baselines.

Deliverable:

Evidence about whether learned communication improves collective control.

## Phase 5: Time-Constrained Communication

Introduce limits on:

- Communication rounds.
    
- Message size.
    
- Communication cost.
    
- Decision latency.
    

Evaluate how these restrictions affect driving performance and decision-making structure.

Deliverable:

An analysis of the trade-off between communication, time pressure, and collective performance.

## Phase 6: Conflicting Interests and Expertise

Introduce different agent reward functions or vary their reliability and expertise.

Evaluate whether the system learns to:

- Communicate selectively.
    
- Trust particular agents.
    
- Redistribute influence dynamically.
    
- Develop stable cooperative strategies.
    
- Adopt hierarchical decision-making.
    

Deliverable:

An experimental investigation into how shared interests and expertise affect cooperation.

## Phase 7: Final Analysis

Perform ablation studies, repeated trials, statistical comparisons, and behavioural analysis.

Evaluate the strongest architectures on tracks or conditions that were not used during training.

Deliverable:

A final research report explaining the effects of information distribution, learned communication, time pressure, and incentives on collective control.

The project would not necessarily need to complete every phase. A rigorous study of learned communication and time constraints could be sufficient for a strong research contribution.

---

# 16. Possible Research Hypotheses

The project could investigate several hypotheses.

**Hypothesis 1: Communication improves performance when relevant information is distributed across agents.**

Compare learned communication against no communication.

**Hypothesis 2: Learned aggregation outperforms simple averaging or fixed voting when agents have different expertise.**

Compare learned decision weights against equal and predefined weights.

**Hypothesis 3: Communication constraints encourage more efficient information exchange.**

Vary communication bandwidth and the number of communication rounds.

**Hypothesis 4: Excessively strict time constraints reduce collective decision quality or encourage hierarchical behaviour.**

Measure decision quality and agent influence under different deadlines.

**Hypothesis 5: Shared rewards encourage cooperation, while conflicting incentives alter communication strategies.**

Vary the balance between team and individual rewards.

**Hypothesis 6: Agents can learn to rely on whichever expertise is most relevant to the current situation.**

Examine whether agent influence changes between straights, corners, and unstable driving conditions.

**Hypothesis 7: Decentralised agents can approach the performance of a centralised policy through communication.**

Compare multi-agent control against a single agent receiving all observations.

These hypotheses are not guaranteed to be true. Discovering limitations or contradictory results would still provide useful research findings.

---

# 17. Main Research Questions

The project's final focus could be expressed through the following research questions.

### Primary question

Can multiple reinforcement learning agents with complementary observations learn to communicate and coordinate effectively to control a vehicle under real-time constraints?

### Secondary questions

1. Does learned communication improve performance compared with fixed aggregation or no communication?
    
2. How does the amount of permitted communication affect collective decision quality and latency?
    
3. Can agents learn when another agent possesses more relevant expertise?
    
4. How do partially conflicting rewards affect cooperation and decision-making?
    
5. Does time pressure encourage consensus, hierarchy, or selective communication?
    
6. Can a decentralised system approach the performance of a centralised controller with access to all observations?
    
7. How can collective decision-making influence be measured and interpreted?
    

For a manageable FYP, it would be sensible to select one primary question and a small number of secondary questions rather than attempting to answer all of them.

---

# 18. Relevant Background Topics

The project will require familiarity with several areas of machine learning and mathematical modelling.

## Reinforcement Learning

- Markov Decision Processes (MDPs).
    
- Partially Observable Markov Decision Processes (POMDPs).
    
- States, observations, actions, rewards, and policies.
    
- Q-learning.
    
- Policy gradients.
    
- Actor-critic methods.
    
- PPO.
    
- SAC.
    
- Exploration versus exploitation.
    
- Reward shaping.
    

## Multi-Agent Reinforcement Learning

- Cooperative MARL.
    
- General-sum stochastic games.
    
- Decentralised partial observability.
    
- Centralised Training, Decentralised Execution.
    
- MAPPO.
    
- Non-stationarity.
    
- Multi-agent credit assignment.
    
- Independent versus joint policies.
    
- Parameter sharing.
    

## Communication and Collective Intelligence

- Majority voting.
    
- Weighted voting.
    
- Consensus mechanisms.
    
- Learned communication.
    
- Message passing.
    
- Communication bottlenecks.
    
- Emergent specialisation.
    
- Expertise estimation.
    
- Social influence.
    
- Hierarchical decision-making.
    

## Experimental Machine Learning

- Baseline comparisons.
    
- Ablation studies.
    
- Statistical variation across random seeds.
    
- Generalisation.
    
- Training stability.
    
- Sample efficiency.
    
- Model capacity and information-access controls.
    
- Evaluation of emergent behaviour.
    

## Real-Time Control

- Continuous action spaces.
    
- Vehicle dynamics.
    
- Steering and braking control.
    
- Control frequency.
    
- Decision latency.
    
- Partial observations.
    
- Temporal state estimation.
    

---

# 19. Related Research and Resources

## Original Multi-Agent Project

_Learning to Vote When It Matters_

Imperial College London, supervised by Dr. Dan Goodman.

The underlying research question concerns how agents with different information and expertise make collective decisions under time pressure.

## Mertzani et al. (2022)

[https://pubmed.ncbi.nlm.nih.gov/36200809/](https://pubmed.ncbi.nlm.nih.gov/36200809/)

Relevant to distributed knowledge aggregation, social influence, expertise, and multi-agent coordination.

## Learning to Communicate with Deep Multi-Agent Reinforcement Learning

Foerster et al., NeurIPS 2016.

[https://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning](https://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning)

Introduces RIAL and DIAL, two approaches for learning communication between agents.

## A First Introduction to Cooperative Multi-Agent Reinforcement Learning

Christopher Amato, 2024.

[https://arxiv.org/abs/2405.06161](https://arxiv.org/abs/2405.06161)

Useful background on cooperative MARL, CTDE, policy gradients, and multi-agent learning methods.

## TrackMania Reinforcement Learning (TMRL)

[https://github.com/trackmania-rl/tmrl](https://github.com/trackmania-rl/tmrl)

Open-source framework for reinforcement learning in Trackmania 2020.

---

# 20. Expected Research Contribution

The primary goal is not simply to train a neural network capable of racing in Trackmania.

The contribution is to investigate **how intelligent agents with distributed information and expertise learn to coordinate their behaviour under time constraints**.

A successful project could provide evidence about:

- Whether learned communication improves cooperative control.
    
- How agent specialisation influences decision quality.
    
- Whether learned decision aggregation outperforms simple voting.
    
- How communication limitations change collective performance.
    
- Whether agents develop expertise-dependent trust.
    
- How shared and individual incentives affect cooperation.
    
- Whether decision-making becomes more hierarchical under time pressure.
    

The project could also establish circumstances in which decentralisation is disadvantageous.

For example, a single centralised policy may outperform the multi-agent system because communication introduces overhead or because agents fail to coordinate effectively.

This would not make the research unsuccessful. Understanding when distributed decision-making works, and when it does not, is itself a useful contribution.

---

# 21. Recommended Initial Scope

The initial system should be deliberately simple.

A reasonable starting configuration would involve:

- Three agents.
    
- Separate observation spaces for vision, telemetry, and track information.
    
- One shared team reward.
    
- A fixed control frequency.
    
- Continuous steering, throttle, and braking.
    
- A centralised single-agent baseline.
    
- A simple fixed aggregation baseline.
    
- A learned communication architecture.
    
- A small number of selected tracks.
    

The first objective should be to demonstrate that a multi-agent controller can reliably perform a basic driving task.

The second objective should be to establish whether communication improves performance.

Once these results are reproducible, the project can introduce time constraints, different incentives, and more sophisticated communication strategies.

This sequence minimises the risk of simultaneously attempting to solve too many difficult problems.

The main research experiments should ideally change one important variable at a time.

For example, once the agents can drive successfully, hold the policies and observations as comparable as possible while varying communication bandwidth.

This makes it easier to identify the actual effect of the experimental condition.

---

# 22. Final Project Summary

_Multi-Agent Reinforcement Learning for Cooperative Real-Time Control in Trackmania_ investigates how multiple specialised artificial intelligence agents can learn to communicate and cooperate to control a single racing vehicle.

Unlike conventional single-agent reinforcement learning, the proposed system distributes observations and potentially expertise across several agents. One agent may receive visual information, another vehicle telemetry, and another knowledge of the track layout.

Because the agents possess incomplete information, they must combine their knowledge to generate effective steering, acceleration, and braking decisions.

The project will investigate different collective decision-making strategies, ranging from simple voting and action averaging to learned communication protocols and potentially hierarchical coordination.

Reinforcement learning methods will be used to train the agents, potentially using centralised training with decentralised execution.

A central focus will be understanding the effects of communication constraints, time pressure, expertise, and partially shared interests on group behaviour.

The system will be evaluated against appropriate single-agent and multi-agent baselines using racing performance, communication efficiency, decision latency, and measures of agent influence.

The broader aim is to understand whether useful collective decision-making structures can emerge through learning, and under what conditions agents with distributed information can outperform simpler cooperation mechanisms.

**Central research objective: Determine whether, and under what conditions, multiple agents with complementary information can learn effective real-time collective decision-making through communication, while maintaining competitive control performance under time constraints.**