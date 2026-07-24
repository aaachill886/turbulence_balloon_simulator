# Stage 3 Policy Training

This folder contains the Stage 3 policy-learning pipeline.

## Goal
Train a policy network that maps a state vector to control actions.

## Inputs
- state vector constructed from logs / benchmark CSVs
- position, velocity, predicted wind, uncertainty
- waypoint geometry
- wind gradients
- environment parameters
- previous control action

## Outputs
- 6-axis action vector:
  - thrust_x_pos, thrust_x_neg
  - thrust_y_pos, thrust_y_neg
  - thrust_z_pos, thrust_z_neg

## Scripts
- `model.py`: policy network definition
- `train_policy.py`: BC training from logged CSVs

## Recommended data source
For Stage 3, use logged runs that include:
- `pos_*`
- `vel_*`
- `obs_u_*`
- `pred_p_*`
- `predSigma`
- `grad_*`
- `cfg_*`
- `waypointPos_*`
- `thrust_*`
- `episode_id`

